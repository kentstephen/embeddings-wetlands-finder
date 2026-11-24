# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Do the yearly embeddings already hold what leaf-off satellite imagery shows?

The embeddings are made from a whole year of Sentinel-2 (and radar), so scenes from the months when the trees are
bare are part of what they summarize. If so, handing the head a raw leaf-off Sentinel-2 picture of the same year
should add nothing on top of an embedding. A summer picture is run beside it as a control: if both add the same
small amount, the gain is not about the season.

For each square, Sentinel-2 L2A scenes of the embedding year are read from Microsoft Planetary Computer for two
seasons, leaf-off (January to April 20 and November 5 to December 31) and leaf-on (June to September 15). Cloud,
cloud shadow and snow are dropped with the scene classification layer, and the median of what is left gives nine
numbers per pixel and season: blue, green, red, near infrared and two shortwave infrared bands, NDVI, NDMI and
MNDWI. Same squares, labels, held-out blocks and small head as terrain_adds.py.

Nothing is cached. Peak memory about 9 GB; the process stops itself at 12 GB.

    uv run scripts/leafoff_adds.py [--year 2024] [--scenes 12]
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import pc  # noqa: E402
import terrain_adds as T  # noqa: E402

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
CLEAR = (4, 5, 6, 7)  # scene classes kept: vegetation, bare ground, water, unclassified
SEASONS = {"leaf-off": ("{y}-01-01/{y}-04-20", "{y}-11-05/{y}-12-31"), "leaf-on": ("{y}-06-01/{y}-09-15",)}


def read_scene(it, LON, LAT):
    """One scene at the given points: (points, 6) reflectance, NaN where not clear or outside the scene."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    def band(key):
        with rasterio.open(pc.signed("sentinel-2-l2a", it["assets"][key]["href"])) as r:
            X, Y = Transformer.from_crs(4326, r.crs, always_xy=True).transform(LON, LAT)
            c = np.floor((X - r.transform.c) / r.transform.a).astype(np.int64)
            rw = np.floor((Y - r.transform.f) / r.transform.e).astype(np.int64)
            m = (c >= 0) & (c < r.width) & (rw >= 0) & (rw < r.height)
            out = np.zeros(LON.shape, np.uint16)
            if m.any():
                c0, r0 = c[m].min(), rw[m].min()
                a = r.read(1, window=Window(c0, r0, c[m].max() - c0 + 1, rw[m].max() - r0 + 1))
                out[m] = a[rw[m] - r0, c[m] - c0]
            return out

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"):
        scl = band("SCL")
        ok = np.isin(scl, CLEAR)
        if ok.mean() < 0.05:
            return None
        off = 1000 if float(it["properties"].get("s2:processing_baseline", "0")) >= 4 else 0
        q = np.stack([band(b) for b in BANDS], -1)
    f = (q.astype(np.float32) - off) / 10000
    f[~ok | (q[..., 0] == 0)] = np.nan
    return f


def composite(box, rc, year, season, most):
    """Nine numbers per pixel from the median of the season's clear looks, a mask of pixels with any, and notes."""
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)
    items = []
    for span in SEASONS[season]:
        items += pc.search("sentinel-2-l2a", box, datetime=span.format(y=year), query={"eo:cloud_cover": {"lt": 50}})
    items = sorted(items, key=lambda i: i["properties"]["eo:cloud_cover"])[:most]
    with ThreadPoolExecutor(6) as ex:
        got = [(i["properties"]["datetime"][:10], f) for i, f in
               zip(items, ex.map(lambda i: E.again(lambda: read_scene(i, LON, LAT)), items)) if f is not None]
    stack = np.stack([f for _, f in got])
    n = np.isfinite(stack[..., 0]).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(stack, 0)
    del stack
    b, g, r, nir, s1, _ = np.moveaxis(med, -1, 0)
    idx = [(nir - r) / (nir + r + 1e-4), (nir - s1) / (nir + s1 + 1e-4), (g - s1) / (g + s1 + 1e-4)]
    F = np.nan_to_num(np.concatenate([med, np.stack(idx, -1)], -1)).astype(np.float32)
    note = (f"{len(got)} scenes ({', '.join(sorted(d for d, _ in got))}), {np.mean(n > 0):.1%} of pixels seen clear, "
            f"median {np.median(n):.0f} clear looks")
    return F, n > 0, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--scenes", type=int, default=12)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    S = E.load_squares()
    for name, s in S.items():
        t = time.time()
        z, W, N, crs, _ = T.read_dem(s["box"])
        s["T"], s["x_ok"] = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
        del z
        for season in SEASONS:
            s[season], ok, note = composite(s["box"], s["rc"], a.year, season, a.scenes)
            s["x_ok"] = s["x_ok"] & ok
            print(f"  {name}, {season}: {note}", flush=True)
        print(f"  {name}: read in {time.time() - t:.0f} s, peak {peak():.1f} GB", flush=True)

    cols = ["terrain", "leaf-off", "leaf-on"]
    for m in E.MODELS:
        cols += [m, m + " + leaf-off", m + " + leaf-on", m + " + terrain", m + " + terrain + leaf-off"]
    R = {t: {c: {} for c in cols} for t in T.TESTS}

    def run(col, name, X, ok):
        for t, v in T.score(X, S[name], ok, np.random.default_rng(0)).items():
            R[t][col][name] = v

    for name, s in S.items():
        ok = s["x_ok"]
        for c in ("terrain", "leaf-off", "leaf-on"):
            run(c, name, s["T" if c == "terrain" else c][ok], ok)
    print("  terrain and the two seasons done", flush=True)
    for model in E.MODELS:
        t0 = time.time()
        hnd = E.open_model(model)
        for name, s in S.items():
            X, ok = E.read_model(model, hnd, a.year, s["rc"])
            ok &= s["x_ok"]
            X = X[ok]
            cat = lambda *k: np.concatenate([X] + [s[i][ok] for i in k], 1)  # noqa: E731
            run(model, name, X, ok)
            run(model + " + leaf-off", name, cat("leaf-off"), ok)
            run(model + " + leaf-on", name, cat("leaf-on"), ok)
            run(model + " + terrain", name, cat("T"), ok)
            run(model + " + terrain + leaf-off", name, cat("T", "leaf-off"), ok)
            del X
        print(f"  {model} done, {time.time() - t0:.0f} s, peak {peak():.1f} GB", flush=True)

    print("\nAUC (and, at 80 % of wetland found, the share of flagged pixels that are wetland). Small head, scored on "
          "1 km blocks it was not taught on.")
    for t in T.TESTS:
        print(f"\n{t}")
        for name in S:
            if name not in R[t]["terrain"]:
                continue
            print(f"  {name}")
            for c in cols:
                au, pr = R[t][c][name]
                print(f"    {c:36} {au:.3f} ({pr:.0%})")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
