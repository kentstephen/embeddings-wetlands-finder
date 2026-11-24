# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""How much of a finer wetland map does the saved head find, by wetland size?

NWI taught the head and NWI leaves out most small wetlands, so agreement with NWI cannot say what the head misses.
Vermont publishes two finer layers (the Significant Wetlands Inventory and the Wetlands Advisory Layer, which adds
town maps, site visits and the state's own screen digitizing). Here each Vermont wetland in three 10 km squares the
head was never taught on is one object, and the question is asked per object, by size:

  - is it in NWI at all (half its pixels or more inside an NWI polygon)?
  - does the head find it: is its mean score (wetland plus open water probability) over a cutoff?
  - how does it rank against dry ground: the share of background pixels in the same square that score below it.

Background is every pixel outside both maps and at least two pixels from either. The share of background over each
cutoff is the price of that cutoff. Nothing is fitted here and nothing is saved.

    uv run scripts/small_wetlands.py
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import nwi_tiles as N  # noqa: E402

VT = "https://anrmaps.vermont.gov/arcgis/rest/services/Open_Data/OPENDATA_ANR_WATER_SP_NOCACHE_v2/MapServer"
LAYERS = {193: "Significant Wetlands Inventory", 194: "Wetlands Advisory Layer"}
SQUARES = {
    "Montpelier, VT": (-72.635, 44.245),
    "Middlebury, VT": (-73.185, 43.995),
    "Franklin County, VT": (-72.885, 44.795),
}
YEAR = 2024
HEAD = Path(__file__).resolve().parent.parent / "models" / "wetland-water-head-alphaearth.npz"
CUTS = (0.08, 0.30, 0.50)
SIZES = ((0, 0.1), (0.1, 0.5), (0.5, 2), (2, 10), (10, 1e9))  # acres
SQM_PER_ACRE = 4046.86
# Vermont's server leaves out an intermediate certificate; the system's trust store finds it, Python's bundle does not
import ssl  # noqa: E402

import truststore  # noqa: E402

TLS = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def vt_polys(layer, box):
    """Every polygon of a Vermont layer that touches the box, as shapely geometries in lon, lat."""
    from shapely.geometry import shape
    from shapely.validation import make_valid
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode(dict(geometry=",".join(f"{v:.6f}" for v in box), geometryType="esriGeometryEnvelope", inSR=4326,
                                        outSR=4326, spatialRel="esriSpatialRelIntersects", outFields="OBJECTID", resultOffset=off,
                                        f="geojson"))
        d = E.again(lambda: json.load(urllib.request.urlopen(f"{VT}/{layer}/query?{q}", timeout=120, context=TLS)))
        fs = d.get("features", [])
        for f in fs:
            if f.get("geometry"):
                g = make_valid(shape(f["geometry"]))
                if not g.is_empty:
                    out.append(g)
        if len(fs) < 1000:
            return out
        off += len(fs)


def objects(polys, box):
    """Touching and overlapping polygons joined into single wetlands, clipped to the box: (geometry, acres)."""
    from pyproj import Transformer
    from shapely.geometry import box as bbox
    from shapely.ops import transform, unary_union
    to_m = Transformer.from_crs(4326, 32618, always_xy=True).transform
    u = unary_union([p.buffer(0) for p in polys]).intersection(bbox(*box))
    parts = [g for g in getattr(u, "geoms", [u]) if g.geom_type == "Polygon" and not g.is_empty]
    return [(g, transform(to_m, g).area / SQM_PER_ACRE) for g in parts]


def main():
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy import ndimage
    E.guard(6)
    t0 = time.time()
    h = np.load(HEAD)
    hnd = E.open_model("AlphaEarth")
    rows = []          # one per wetland: square, acres, in NWI, mean score, rank against background
    back = {}          # square -> background scores
    for name, (lon, lat) in SQUARES.items():
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        y0, y1, x0, x1 = rc
        shape_ = (y1 - y0, x1 - x0)
        tr = Affine(E.RES, 0, E.X0 + x0 * E.RES, 0, -E.RES, E.Y0 - y0 * E.RES)
        x, ok = E.read_model("AlphaEarth", hnd, YEAR, rc)
        z = np.maximum(x.reshape(-1, x.shape[-1]) @ h["W1"] + h["b1"], 0) @ h["W2"] + h["b2"]
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        wet = (p[:, 1] + p[:, 2]).reshape(shape_)
        del x, z, p
        polys = [g for layer in LAYERS for g in vt_polys(layer, box)]
        obs = sorted(objects(polys, box), key=lambda t: -t[1])
        # larger first, so a small wetland keeps its pixels where shapes touch; every pixel a shape touches counts
        ids = rasterize([(g, i + 1) for i, (g, _) in enumerate(obs)], out_shape=shape_, transform=tr, fill=0, dtype="int32",
                        all_touched=True)
        nwi = E.burn(N.read(box), rc) > 0
        idx = np.arange(1, len(obs) + 1)
        n_px = ndimage.sum(np.ones(shape_), ids, idx)
        mean = ndimage.mean(wet, ids, idx)
        in_nwi = ndimage.mean(nwi.astype(np.float32), ids, idx) >= 0.5
        near = ndimage.binary_dilation((ids > 0) | nwi, iterations=2)
        b = np.sort(wet[~near & ok])
        back[name] = b
        for i, (_, acres) in enumerate(obs):
            if n_px[i] > 0:
                rows.append((name, acres, bool(in_nwi[i]), float(mean[i]), float(np.searchsorted(b, mean[i]) / len(b))))
        print(f"  {name}: {len(polys)} polygons, {len(obs)} wetlands, {int(nwi.sum() / nwi.size * 1000) / 10}% of pixels in NWI, "
              f"{int((ids > 0).sum() / ids.size * 1000) / 10}% in the Vermont maps, {time.time() - t0:.0f} s", flush=True)

    sq, acres, in_nwi, mean, rank = (np.array(c) for c in zip(*rows))
    print(f"\nVermont wetlands in {len(SQUARES)} squares the head was not taught on, {YEAR}. Score: wetland plus open water probability,"
          f" mean over the wetland.\n")
    print("  Background (outside both maps, two pixels clear): share of pixels over each cutoff")
    allb = np.concatenate(list(back.values()))
    print("    " + ", ".join(f"over {c:.2f}: {(allb >= c).mean() * 100:.1f}%" for c in CUTS) + f"   ({len(allb):,} pixels)")

    def table(title, keep):
        print(f"\n  {title}")
        print(f"    {'acres':>10} {'count':>6} {'in NWI':>7} " + " ".join(f"{'>=' + format(c, '.2f'):>7}" for c in CUTS) + f" {'rank':>6}")
        for lo, hi in SIZES:
            k = keep & (acres >= lo) & (acres < hi)
            if not k.any():
                continue
            lab = f"{lo:g} to {hi:g}" if hi < 1e8 else f"{lo:g}+"
            print(f"    {lab:>10} {k.sum():>6} {in_nwi[k].mean() * 100:>6.0f}% "
                  + " ".join(f"{(mean[k] >= c).mean() * 100:>6.0f}%" for c in CUTS) + f" {np.median(rank[k]):>6.2f}")

    table("All Vermont wetlands, by size. Found = mean score at or over the cutoff. Rank = median share of background scoring lower.",
          np.ones(len(acres), bool))
    table("Only those NWI does not have", ~in_nwi)
    table("Only those NWI has", in_nwi)
    for name in SQUARES:
        table(name, sq == name)
    import psutil
    print(f"\n{time.time() - t0:.0f} s, memory of the process at the end {psutil.Process().memory_info().rss / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
