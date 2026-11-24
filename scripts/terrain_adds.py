# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Does lidar terrain add to the embeddings for finding wetlands?

Same three squares, labels and held-out blocks as embedding_signal.py. Terrain comes from the USGS 3DEP 1 m
elevation models (bare earth made from the lidar point clouds), read at 2 m and turned into eleven numbers per
10 m pixel: slope, curvature, roughness inside the pixel, height against the mean of the surroundings at four
distances (10 to 300 m), height above the lowest ground nearby at three (30 to 300 m), and depth below the level
a depression would fill to.

A small head (one hidden layer of 64) is taught on seven inputs: terrain alone, each embedding alone, and each
embedding with terrain. Scored on 1 km blocks it was not taught on (three folds).

Nothing is cached. One embedding is held at a time. Peak memory about 7 GB; the process stops itself at 12 GB.

    uv run scripts/terrain_adds.py [--year 2024]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace, minimum_filter, uniform_filter
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402

TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
STEP, FOLDS, TRAIN_MAX = 2.0, 3, 60_000
TERRAIN = ["slope", "curvature", "roughness", "rel 10 m", "rel 30 m", "rel 100 m", "rel 300 m", "above low 30 m",
           "above low 100 m", "above low 300 m", "depression depth"]


def read_dem(box):
    """3DEP 1 m elevation over the box on a 2 m grid in the tiles' UTM zone: array (NaN none), west, north, CRS,
    and the lidar projects used. Newer tiles are read first; older ones only fill what is still empty."""
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    for attempt in range(6):  # the catalog sometimes answers with a short error instead of the tile list
        d = E.get_json(TNM, {"datasets": "Digital Elevation Model (DEM) 1 meter", "bbox": ",".join(f"{v:.5f}" for v in box),
                             "prodFormats": "GeoTIFF", "max": 100})
        if d.get("items"):
            break
        time.sleep(5 + 10 * attempt)
    else:
        raise RuntimeError(f"3DEP catalog gave no tiles for {box}: {str(d)[:200]}")
    items = d["items"]
    items.sort(key=lambda i: i["publicationDate"], reverse=True)
    z = crs = None
    used = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"):
        for it in items:
            with rasterio.open("/vsicurl/" + it["downloadURL"]) as r:
                if z is None:
                    crs = r.crs
                    xs, ys = Transformer.from_crs(4326, crs, always_xy=True).transform(
                        [box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
                    W, N = np.floor(min(xs) / STEP) * STEP - 400, np.ceil(max(ys) / STEP) * STEP + 400  # 400 m margin
                    w, h = int((max(xs) + 400 - W) / STEP) + 1, int((N - (min(ys) - 400)) / STEP) + 1
                    z = np.full((h, w), np.nan, np.float32)
                if r.crs != crs:
                    continue
                b = r.bounds
                x0, x1 = max(b.left, W), min(b.right, W + w * STEP)
                y0, y1 = max(b.bottom, N - h * STEP), min(b.top, N)
                if x1 - x0 < 2 * STEP or y1 - y0 < 2 * STEP:
                    continue
                c0, r0 = int(round((x0 - W) / STEP)), int(round((N - y1) / STEP))
                nc, nr = int((x1 - x0) / STEP), int((y1 - y0) / STEP)
                nc, nr = min(nc, w - c0), min(nr, h - r0)
                tgt = z[r0:r0 + nr, c0:c0 + nc]
                if not np.isnan(tgt).any():
                    continue
                a = r.read(1, window=from_bounds(x0, y1 - nr * STEP, x0 + nc * STEP, y1, r.transform),
                           out_shape=(nr, nc), resampling=Resampling.average, boundless=True, fill_value=np.nan).astype(np.float32)
                if r.nodata is not None:
                    a[a == r.nodata] = np.nan
                a[a < -1000] = np.nan
                m = np.isnan(tgt) & ~np.isnan(a)
                if m.any():
                    tgt[m] = a[m]
                    used.append(it["title"].split(" ", 5)[-1] if it["title"].startswith("USGS 1 Meter") else it["title"])
    return z, W, N, crs, sorted(set(used))


def terrain(z):
    """Eleven terrain numbers on a 10 m grid (h / 5, w / 5, 11) with the same north-west corner: each is worked out
    at 2 m and averaged over the 10 m cell."""
    from skimage.morphology import reconstruction
    hole = np.isnan(z)
    if hole.any():  # nearest ground into the holes, so the filters have something to read
        i = distance_transform_edt(hole, return_distances=False, return_indices=True)
        z = z[tuple(i)]
    n = lambda m: max(int(round(m / STEP)), 1)  # noqa: E731
    zs = gaussian_filter(z, 2)
    gy, gx = np.gradient(zs, STEP)
    out = [np.log1p(100 * np.hypot(gx, gy)), laplace(zs) / STEP ** 2]
    d = z - uniform_filter(z, n(10))
    out.append(np.log1p(100 * np.sqrt(np.maximum(uniform_filter(d * d, n(10)), 0))))
    del gx, gy, d
    for r in (10, 30, 100, 300):
        out.append(z - uniform_filter(z, 2 * n(r) + 1))
    for r in (30, 100, 300):
        out.append(np.log1p(np.maximum(z - minimum_filter(z, 2 * n(r) + 1), 0)))
    seed = np.full_like(z, z.max())
    seed[0], seed[-1], seed[:, 0], seed[:, -1] = z[0], z[-1], z[:, 0], z[:, -1]
    out.append(np.log1p(np.maximum(reconstruction(seed, z, method="erosion").astype(np.float32) - z, 0)))
    k = n(10)
    for i, f in enumerate(out):  # one at a time, so only one full-size copy is alive
        out[i] = np.nan_to_num(uniform_filter(f.astype(np.float32), k)[k // 2::k, k // 2::k])
    return np.stack(out, -1)


def on_aef(F, W, N, crs, rc):
    """The terrain numbers under the center of every AlphaEarth pixel, and whether the pixel is inside the grid."""
    from pyproj import Transformer
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)
    X, Y = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
    c, r = np.floor((X - W) / 10).astype(int), np.floor((N - Y) / 10).astype(int)
    ok = (c >= 0) & (c < F.shape[1]) & (r >= 0) & (r < F.shape[0])
    return F[np.clip(r, 0, F.shape[0] - 1), np.clip(c, 0, F.shape[1] - 1)], ok


def head():
    return make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=40, early_stopping=True, n_iter_no_change=4,
                                                         batch_size=512, random_state=0))


def held_out(X, y, fold, inside, rng):
    p = np.full(len(y), np.nan, np.float32)
    for k in range(FOLDS):
        tr, te = np.flatnonzero(inside & (fold != k)), np.flatnonzero(inside & (fold == k))
        if len(te) and 0 < y[tr].sum() < len(tr):
            if len(tr) > TRAIN_MAX:
                tr = rng.choice(tr, TRAIN_MAX, replace=False)
            p[te] = head().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


TESTS = ("vegetated wetland vs upland", f"the same, within {E.EDGE_M} m of the NWI line",
         "forested/shrub wetland vs upland, tree cover only")


def score(X, s, ok, rng):
    """AUC and the share of flagged pixels that are wetland at 80 % found, for each test."""
    y_all, f, edge, cov = s["lab"][ok], s["fold"][ok] % FOLDS, s["edge"][ok], s["cover"][ok]
    up, veg = y_all == 0, np.isin(y_all, E.VEGETATED)
    use = veg | up
    p = held_out(X, veg, f, use, rng)
    near = use & (edge <= E.EDGE_M)
    out = {TESTS[0]: (E.auc(veg[use], p[use]), E.prec_at_recall(veg[use], p[use])),
           TESTS[1]: (E.auc(veg[near], p[near]), E.prec_at_recall(veg[near], p[near]))}
    m = (cov == 10) & ((y_all == 2) | up)
    y = y_all == 2
    if y[m].sum() >= 500:
        p2 = held_out(X, y, f, m, rng)
        out[TESTS[2]] = (E.auc(y[m], p2[m]), E.prec_at_recall(y[m], p2[m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    S = E.load_squares()
    for name, s in S.items():
        t = time.time()
        z, W, N, crs, used = read_dem(s["box"])
        s["T"], s["t_ok"] = on_aef(terrain(z), W, N, crs, s["rc"])
        print(f"  terrain {name}: {z.shape[0]} x {z.shape[1]} cells of {STEP:.0f} m, {np.isnan(z).mean():.1%} empty, "
              f"{time.time() - t:.0f} s, peak {peak():.1f} GB; lidar projects: {'; '.join(used)}", flush=True)
        del z

    cols = ["terrain"] + [c for m in E.MODELS for c in (m, m + " + terrain")]
    R = {t: {c: {} for c in cols} for t in TESTS}
    for name, s in S.items():
        ok = s["t_ok"]
        for t, v in score(s["T"][ok], s, ok, np.random.default_rng(0)).items():
            R[t]["terrain"][name] = v
    print("  terrain alone done", flush=True)
    for model in E.MODELS:
        t0 = time.time()
        hnd = E.open_model(model)
        for name, s in S.items():
            X, ok = E.read_model(model, hnd, a.year, s["rc"])
            ok &= s["t_ok"]
            X = X[ok]
            for t, v in score(X, s, ok, np.random.default_rng(0)).items():
                R[t][model][name] = v
            X = np.concatenate([X, s["T"][ok]], 1)
            for t, v in score(X, s, ok, np.random.default_rng(0)).items():
                R[t][model + " + terrain"][name] = v
            del X
            print(f"  {model} {name} done, {time.time() - t0:.0f} s, peak {peak():.1f} GB", flush=True)

    print("\nAUC (and, at 80 % of wetland found, the share of flagged pixels that are wetland). Small head, scored on "
          "1 km blocks it was not taught on.")
    for t in TESTS:
        print(f"\n{t}")
        for name in S:
            if name not in R[t]["terrain"]:
                continue
            print(f"  {name}")
            for c in cols:
                au, pr = R[t][c][name]
                print(f"    {c:24} {au:.3f} ({pr:.0%})")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
