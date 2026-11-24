"""NWI polygons for a box from the PMTiles copy on Source Cooperative (cboettig/wetlands, nwi-v2.pmtiles).

The USFWS map service answers HTTP 500 when a box touches a very large polygon (a bay, a big swamp), since it sends
the whole polygon. The zoom 14 tiles hold every polygon cut at the tile's edge, so a box costs a few dozen small
range reads whatever it touches. Vertices are on a 4096 grid per tile, about half a meter at zoom 14.

    polys = nwi_tiles.read(box)   # [(shapely geometry in lon/lat, type code)], larger pieces first

Needs the pmtiles and mapbox-vector-tile packages.
"""
import functools
import gzip
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import embedding_signal as E

URL = "https://data.source.coop/cboettig/wetlands/nwi-v2.pmtiles"
Z = 14


@functools.lru_cache(maxsize=512)
def _get(off, n):
    def go():
        r = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off + n - 1}"})
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.read()
    return E.again(go)


def _tile(xy):
    import mapbox_vector_tile
    import shapely
    from pmtiles.reader import Reader
    from shapely.geometry import shape
    x, y = xy
    b = Reader(_get).get(Z, x, y)
    if not b:
        return []
    d = mapbox_vector_tile.decode(gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)
    out, n = [], 2 ** Z
    for lyr in d.values():
        ext = lyr["extent"]

        def to_lonlat(c, ext=ext):  # tile units, y up, to degrees
            lon = (x + c[:, 0] / ext) / n * 360 - 180
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1 - c[:, 1] / ext) / n))))
            return np.column_stack([lon, lat])
        for f in lyr["features"]:
            if f["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
                continue
            g = shapely.transform(shape(f["geometry"]), to_lonlat)
            if not g.is_empty:
                out.append((g, E.TYPES.get(f["properties"].get("WETLAND_TYPE"), 8)))
    return out


def read(box):
    """NWI polygon pieces that touch the box (lon0, lat0, lon1, lat1), as (geometry, type code), larger first."""
    lon0, lat0, lon1, lat1 = box
    n = 2 ** Z
    tx = lambda lon: int(np.floor((lon + 180) / 360 * n))  # noqa: E731
    ty = lambda lat: int(np.floor((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n))  # noqa: E731
    tiles = [(x, y) for x in range(tx(lon0), tx(lon1) + 1) for y in range(ty(lat1), ty(lat0) + 1)]
    with ThreadPoolExecutor(8) as ex:
        out = [p for part in ex.map(_tile, tiles) for p in part]
    return sorted(out, key=lambda t: -t[0].area)
