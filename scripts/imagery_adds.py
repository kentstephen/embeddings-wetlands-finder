# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Does aerial imagery add to the embeddings and the lidar terrain for finding wetlands?

Same three squares, labels, held-out blocks and small head as terrain_adds.py. Imagery is the USDA NAIP aerial
photography (0.3 to 0.6 m, red, green, blue and near infrared, summer, leaves on), read from Microsoft Planetary
Computer at 2 m and turned into eleven numbers per 10 m pixel: the mean of each band, the spread of each band
inside the pixel (texture), the mean and spread of NDVI, and the mean of NDWI.

The head is taught on imagery alone, terrain with imagery, and each embedding with terrain and with terrain and
imagery. NAIP is flown state by state on different days with different cameras and is not calibrated, so its
numbers are comparable inside a square, not between squares.

Nothing is cached. Peak memory about 8 GB; the process stops itself at 12 GB.

    uv run scripts/imagery_adds.py [--year 2024]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import pc  # noqa: E402
import terrain_adds as T  # noqa: E402

STEP = 2.0
IMAGERY = ["red", "green", "blue", "near infrared", "red spread", "green spread", "blue spread",
           "near infrared spread", "NDVI", "NDVI spread", "NDWI"]


def read_naip(box):
    """NAIP over the box on a 2 m grid in the photos' UTM zone: (4, h, w) float32 in 0 to 1 (NaN none), west, north,
    CRS, and the flight days used. The newest flight is read first; older ones only fill what is still empty."""
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    items = sorted(pc.search("naip", box), key=lambda i: i["properties"]["datetime"], reverse=True)
    z = crs = None
    used = set()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"):
        for it in items:
            if z is not None and not np.isnan(z[0]).any():
                break
            with rasterio.open(pc.signed("naip", it["assets"]["image"]["href"])) as r:
                if r.count < 4:
                    continue
                if z is None:
                    crs = r.crs
                    xs, ys = Transformer.from_crs(4326, crs, always_xy=True).transform(
                        [box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
                    W, N = np.floor(min(xs) / STEP) * STEP - 20, np.ceil(max(ys) / STEP) * STEP + 20
                    w, h = int((max(xs) + 20 - W) / STEP) + 1, int((N - (min(ys) - 20)) / STEP) + 1
                    z = np.full((4, h, w), np.nan, np.float32)
                if r.crs != crs:
                    continue
                b = r.bounds
                x0, x1 = max(b.left, W), min(b.right, W + w * STEP)
                y0, y1 = max(b.bottom, N - h * STEP), min(b.top, N)
                if x1 - x0 < 2 * STEP or y1 - y0 < 2 * STEP:
                    continue
                c0, r0 = int(round((x0 - W) / STEP)), int(round((N - y1) / STEP))
                nc, nr = min(int((x1 - x0) / STEP), w - c0), min(int((y1 - y0) / STEP), h - r0)
                tgt = z[:, r0:r0 + nr, c0:c0 + nc]
                if not np.isnan(tgt[0]).any():
                    continue
                a = r.read([1, 2, 3, 4], window=from_bounds(x0, y1 - nr * STEP, x0 + nc * STEP, y1, r.transform),
                           out_shape=(4, nr, nc), resampling=Resampling.average, boundless=True, fill_value=0)
                m = np.isnan(tgt[0]) & (a.max(0) > 0)
                if m.any():
                    tgt[:, m] = a[:, m].astype(np.float32) / 255
                    used.add(it["properties"]["datetime"][:10])
    return z, W, N, crs, sorted(used)


def imagery(z):
    """Eleven imagery numbers on a 10 m grid (h / 5, w / 5, 11) and a mask of cells with a photo under them."""
    k = int(round(10 / STEP))
    have = ~np.isnan(z[0])
    z = np.nan_to_num(z)
    r, g, b, n = z
    ndvi, ndwi = (n - r) / (n + r + 1e-3), (g - n) / (g + n + 1e-3)
    mean = lambda f: uniform_filter(f, k)[k // 2::k, k // 2::k]  # noqa: E731
    spread = lambda f: np.sqrt(np.maximum(mean(f * f) - mean(f) ** 2, 0))  # noqa: E731
    F = [mean(v) for v in z] + [spread(v) for v in z] + [mean(ndvi), spread(ndvi), mean(ndwi)]
    return np.stack(F, -1).astype(np.float32), mean(have.astype(np.float32)) > 0.99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    S = E.load_squares()
    for name, s in S.items():
        t = time.time()
        z, W, N, crs, _ = T.read_dem(s["box"])
        s["T"], t_ok = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
        del z
        z, W, N, crs, days = read_naip(s["box"])
        F, have = imagery(z)
        s["I"], i_ok = T.on_aef(F, W, N, crs, s["rc"])
        i_ok &= T.on_aef(have, W, N, crs, s["rc"])[0]
        s["x_ok"] = t_ok & i_ok
        print(f"  {name}: imagery flown {', '.join(days)}, {z.shape[1]} x {z.shape[2]} cells of {STEP:.0f} m, "
              f"{i_ok.mean():.1%} of pixels have a photo, {time.time() - t:.0f} s, peak {peak():.1f} GB", flush=True)
        del z

    cols = ["terrain", "imagery", "terrain + imagery"]
    for m in E.MODELS:
        cols += [m + " + terrain", m + " + terrain + imagery"]
    R = {t: {c: {} for c in cols} for t in T.TESTS}

    def run(col, name, X, ok):
        for t, v in T.score(X, S[name], ok, np.random.default_rng(0)).items():
            R[t][col][name] = v

    for name, s in S.items():
        ok = s["x_ok"]
        run("terrain", name, s["T"][ok], ok)
        run("imagery", name, s["I"][ok], ok)
        run("terrain + imagery", name, np.concatenate([s["T"][ok], s["I"][ok]], 1), ok)
    print("  terrain and imagery done", flush=True)
    for model in E.MODELS:
        t0 = time.time()
        hnd = E.open_model(model)
        for name, s in S.items():
            X, ok = E.read_model(model, hnd, a.year, s["rc"])
            ok &= s["x_ok"]
            X = np.concatenate([X[ok], s["T"][ok]], 1)
            run(model + " + terrain", name, X, ok)
            X = np.concatenate([X, s["I"][ok]], 1)
            run(model + " + terrain + imagery", name, X, ok)
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
                print(f"    {c:34} {au:.3f} ({pr:.0%})")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
