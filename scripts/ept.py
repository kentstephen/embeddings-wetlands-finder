"""A small reader for Entwine Point Tile (EPT) point clouds, as USGS publishes 3DEP lidar (s3://usgs-lidar-public,
one cloud per lidar project, EPSG:3857). Finds the octree nodes that touch a box and reads them one at a time,
so a caller can fold points into a grid and let them go.
"""
import io
import json
import threading
import time

import numpy as np

ROOT = "https://usgs-lidar-public.s3.us-west-2.amazonaws.com"
R_EARTH = 6378137.0


_tl = threading.local()


def get(url, tries=5):
    """The body of a URL, over a connection each thread keeps open (a new TLS handshake per node costs more than
    the node)."""
    import urllib3
    if not hasattr(_tl, "http"):
        _tl.http = urllib3.PoolManager(maxsize=2, timeout=urllib3.Timeout(connect=10, read=120))
    for attempt in range(tries):
        try:
            r = _tl.http.request("GET", url)
            if r.status != 200:
                raise OSError(f"HTTP {r.status} {url}")
            return r.data
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(2 + 4 * attempt)


def to_3857(lon, lat):
    return R_EARTH * np.radians(lon), R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def to_lonlat(x, y):
    return np.degrees(x / R_EARTH), np.degrees(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2)


def info(project):
    return json.loads(get(f"{ROOT}/{project}/ept.json"))


def nodes(project, meta, box3857):
    """Octree nodes (key, point count) whose cube touches the box (x0, y0, x1, y1), every depth."""
    bx0, by0, _, bx1, by1, _ = meta["bounds"]
    span = bx1 - bx0
    x0, y0, x1, y1 = box3857
    out, cache = [], {}

    def touches(key):
        d, x, y, _ = (int(v) for v in key.split("-"))
        s = span / 2 ** d
        return bx0 + x * s < x1 and bx0 + (x + 1) * s > x0 and by0 + y * s < y1 and by0 + (y + 1) * s > y0

    def walk(page):
        if page not in cache:
            cache[page] = json.loads(get(f"{ROOT}/{project}/ept-hierarchy/{page}.json"))
        h = cache[page]
        stack = [page]
        while stack:
            key = stack.pop()
            n = h.get(key)
            if n is None or not touches(key):
                continue
            if n == -1:  # the subtree is in its own page
                walk(key)
                continue
            out.append((key, n))
            d, x, y, z = (int(v) for v in key.split("-"))
            for i in range(8):
                stack.append(f"{d + 1}-{2 * x + (i & 1)}-{2 * y + (i >> 1 & 1)}-{2 * z + (i >> 2 & 1)}")

    walk("0-0-0-0")
    seen = {}
    for k, n in out:
        seen[k] = n
    return sorted(seen.items())


def fetch_node(project, key):
    return get(f"{ROOT}/{project}/ept-data/{key}.laz")


def read_node(project, key, raw=None):
    """One node's points: x, y, z (float64, EPSG:3857 and the cloud's vertical unit), intensity, return number,
    number of returns, classification, scan angle (degrees), and the bytes fetched."""
    import laspy
    if raw is None:
        raw = fetch_node(project, key)
    las = laspy.read(io.BytesIO(raw))
    ang = np.asarray(las.scan_angle, np.float32) * 0.006 if "scan_angle" in las.point_format.dimension_names \
        else np.asarray(las.scan_angle_rank, np.float32)
    return (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z), np.asarray(las.intensity),
            np.asarray(las.return_number), np.asarray(las.number_of_returns), np.asarray(las.classification), ang, len(raw))
