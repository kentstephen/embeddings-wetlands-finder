# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""Past wetland or not: do the embeddings say which crop, which kind of natural vegetation, which land cover?
And does the shape of the ground add to them for any of those?

Three answer keys, each a 30 m map read at the center of every 10 m embedding pixel:

  crop         USDA Cropland Data Layer 2021 (Planetary Computer), crop classes only: corn, soybeans, rice, cotton, ...
  vegetation   LANDFIRE existing vegetation type 2023 (USGS image service), natural types only: the ecological
               system, such as a bottomland hardwood forest or a longleaf pine woodland
  land cover   Annual NLCD 2023, all classes

Inputs compared on the same pixels: AlphaEarth, TESSERA and OlmoEarth of the map's year, the eleven terrain numbers
from 10 m elevation alone, and AlphaEarth with them.

Squares are the wetland landscapes of many_squares.py and the ordinary ones of water_head.py, dealt into five folds;
every square is scored by a head that saw no pixel of it. A head is taught the classes that at least two of its
squares have. Per held-out square, among the pixels of classes the head was taught:

  right        the share whose class the head names, beside the share the square's commonest class would get
  ranked       each class against the others (AUC), averaged over the classes with enough pixels

These maps are models themselves, 30 m, and partly made from the same satellites, so agreement with them is not
ground truth; a class that changes inside a 30 m cell is mislabeled at 10 m. Nothing is saved.

    uv run scripts/cover_kinds.py [--workers 2] [--limit 4]
"""
import argparse
import io
import sys
import time
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import fine_labels as F  # noqa: E402
import many_squares as M  # noqa: E402
import pc  # noqa: E402
import terrain_adds as T  # noqa: E402
import water_head as W  # noqa: E402

MODELS = ("AlphaEarth", "TESSERA", "OlmoEarth")
KEYS = {"crop": 2021, "vegetation": 2023, "land cover": 2023}  # the year each map is of
PER_CLASS = 400     # pixels kept per class in a square
HAVE = 150          # pixels a class needs in a square to be kept there
KEEP = 6_000        # pixels kept per square and answer key
TEACH = 2_500       # pixels taught from each square
FOLDS = 5
MIN = 100           # pixels a class needs in a held-out square, and the rest too, to be ranked
NLCD = ("https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/"
        "c1/v1/cu/mosaic/Annual_NLCD_LndCov_{}_CU_C1V1.tif")
EVT = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2023/LF2023_EVT_CONUS/ImageServer/exportImage"
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3")


def grid(rc):
    y0, y1, x0, x1 = rc
    return np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)


def nearest(src, LON, LAT, out):
    """The raster's value under each point, written into `out` where it is still 0 and the raster has one."""
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    X, Y = Transformer.from_crs(4326, src.crs.to_wkt(), always_xy=True).transform(LON.ravel(), LAT.ravel())
    pad = 3 * abs(src.transform.a)
    w = from_bounds(X.min() - pad, Y.min() - pad, X.max() + pad, Y.max() + pad, src.transform).round_offsets().round_lengths()
    a = E.again(lambda: src.read(1, window=w, boundless=True, fill_value=0))
    tr = src.window_transform(w)
    r, c = np.floor((Y - tr.f) / tr.e).astype(int), np.floor((X - tr.c) / tr.a).astype(int)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    v = np.zeros(LON.size, np.int32)
    v[ok] = a[r[ok], c[ok]]
    o = out.ravel()
    m = (o == 0) & (v > 0)
    o[m] = v[m]


def read_crop(box, rc):
    """Cropland Data Layer 2021 class per pixel, 0 where it is not a crop."""
    import rasterio
    LON, LAT = grid(rc)
    out = np.zeros(LON.shape, np.int32)
    with rasterio.Env(**ENV):
        for it in pc.search("usda-cdl", box, datetime="2021-01-01/2021-12-31"):
            if "cropland" in it["assets"]:
                with rasterio.open(pc.signed("usda-cdl", it["assets"]["cropland"]["href"])) as s:
                    nearest(s, LON, LAT, out)
    # crops only: the classes past 80 below 196 are forest, shrub, water, developed and the like
    out[~(((out >= 1) & (out <= 61) | (out >= 66) & (out <= 77) | (out >= 204)) & (out != 37))] = 0
    return out


def read_vegetation(box, rc):
    """LANDFIRE existing vegetation type 2023 per pixel, 0 where it is water, developed or farmed."""
    import rasterio
    LON, LAT = grid(rc)
    w = int(np.ceil((box[2] - box[0]) * 111_320 * np.cos(np.radians(0.5 * (box[1] + box[3]))) / 30))
    h = int(np.ceil((box[3] - box[1]) * 111_320 / 30))
    q = urllib.parse.urlencode(dict(bbox=",".join(f"{v:.6f}" for v in box), bboxSR=4326, imageSR=4326, size=f"{w},{h}", format="tiff",
                                    pixelType="U16", interpolation="RSP_NearestNeighbor", f="image"))
    raw = E.again(lambda: urllib.request.urlopen(f"{EVT}?{q}", timeout=180).read())
    out = np.zeros(LON.shape, np.int32)
    with rasterio.open(io.BytesIO(raw)) as s:
        nearest(s, LON, LAT, out)
    out[(out == 7292) | ((out >= 7295) & (out <= 7300)) | ((out >= 7960) & (out <= 7999)) | (out > 20000)] = 0
    return out


def read_cover(box, rc):
    import rasterio
    LON, LAT = grid(rc)
    out = np.zeros(LON.shape, np.int32)
    with rasterio.Env(**ENV), rasterio.open(NLCD.format(2023)) as s:
        nearest(s, LON, LAT, out)
    out[out >= 250] = 0
    return out


READ = {"crop": read_crop, "vegetation": read_vegetation, "land cover": read_cover}


def gather(name, lon, lat):
    """Kept pixels of one square for each answer key: class and every input. A string says why it was dropped."""
    warnings.filterwarnings("ignore")
    E.guard(8)
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        z, Wx, Nn, crs, _ = F.dem10(box)
        if np.isnan(z).mean() > 0.6:
            return "10 m elevation missing over most of the square"
        Tn, ok = T.on_aef(T.terrain(z), Wx, Nn, crs, rc)
        del z
        ok &= np.isfinite(Tn).all(-1)
        X, miss = {}, []
        for y in sorted(set(KEYS.values())):
            for m in MODELS:
                try:
                    a, k = E.read_model(m, E.open_model(m), y, rc)
                    if k.mean() < 0.5:
                        raise RuntimeError(f"data on {k.mean():.0%} of the square")
                    ok &= k
                    X[m, y] = a.astype(np.float16)
                    del a
                except Exception as e:  # noqa: BLE001
                    miss.append(f"{m} {y}: {type(e).__name__} {str(e)[:50]}")
        rng = np.random.default_rng(0)
        out, said = {}, []
        for key, year in KEYS.items():
            try:
                lab = READ[key](box, rc)
            except Exception as e:  # noqa: BLE001
                said.append(f"{key}: {type(e).__name__} {str(e)[:60]}")
                continue
            v, n = np.unique(lab[ok & (lab > 0)], return_counts=True)
            rows = [rng.choice(np.flatnonzero((ok & (lab == c)).ravel()), min(PER_CLASS, k), replace=False) for c, k in zip(v, n) if k >= HAVE]
            if not rows:
                said.append(f"{key}: no class with {HAVE} pixels")
                continue
            rows = np.concatenate(rows)
            rows = np.sort(rng.choice(rows, min(KEEP, len(rows)), replace=False))
            pick = lambda a: a.reshape(lab.size, -1)[rows]  # noqa: E731
            out[key] = dict(y=lab.ravel()[rows], X={m: pick(X[m, year]) for m in MODELS if (m, year) in X}, classes=int((n >= HAVE).sum()))
            out[key]["X"]["terrain"] = pick(Tn).astype(np.float32)
        return dict(keys=out, miss=miss + said, secs=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}: {str(e)[:160]}"


def feats(g, parts):
    return np.hstack([g["X"][p].astype(np.float32) for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    F.ELEV = "10m"
    peak = E.guard(10)
    t0 = time.time()
    C = {**M.CANDIDATES, **W.ORDINARY}
    if a.limit:
        C = {**dict(list(M.CANDIDATES.items())[9:9 + a.limit]), **dict(list(W.ORDINARY.items())[5:5 + a.limit])}
    G = {}
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(gather, n, lon, lat): n for n, (lon, lat, _) in C.items()}
        for i, j in enumerate(as_completed(jobs)):
            n, g = jobs[j], j.result()
            if isinstance(g, str):
                print(f"  dropped {n}: {g}", flush=True)
                continue
            G[n] = g
            print(f"  {i + 1} of {len(C)}, {n} ({C[n][2]}): " + ", ".join(f"{k} {v['classes']} classes" for k, v in g["keys"].items())
                  + (f"; {'; '.join(g['miss'])}" if g["miss"] else "") + f"; {g['secs']:.0f} s, {(time.time() - t0) / 60:.0f} min", flush=True)
    INPUTS = {"AlphaEarth": ["AlphaEarth"], "TESSERA": ["TESSERA"], "OlmoEarth": ["OlmoEarth"], "terrain": ["terrain"],
              "AlphaEarth + terrain": ["AlphaEarth", "terrain"]}
    for key in KEYS:
        names = sorted(n for n in G if key in G[n]["keys"] and all(m in G[n]["keys"][key]["X"] for m in MODELS))
        print(f"\n{key}: {len(names)} squares with this map and all three embeddings", flush=True)
        if len(names) < FOLDS:
            continue
        fold = {n: i % FOLDS for i, n in enumerate(names)}
        R = {inp: {} for inp in INPUTS}
        base = {}
        for k in range(FOLDS):
            tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
            seen = {}
            for n in tr:
                for c in np.unique(G[n]["keys"][key]["y"]):
                    seen[c] = seen.get(c, 0) + 1
            taught = np.array(sorted(c for c, v in seen.items() if v >= 2))
            rng = np.random.default_rng(0)
            pick = {}
            for n in tr:
                i = np.flatnonzero(np.isin(G[n]["keys"][key]["y"], taught))
                pick[n] = np.sort(rng.choice(i, min(TEACH, len(i)), replace=False))
            for inp, parts in INPUTS.items():
                m = T.head().fit(np.concatenate([feats(G[n]["keys"][key], parts)[pick[n]] for n in tr]),
                                 np.concatenate([G[n]["keys"][key]["y"][pick[n]] for n in tr]))
                cl = list(m.classes_)
                for n in te:
                    g = G[n]["keys"][key]
                    u = np.isin(g["y"], cl)
                    if u.sum() < 2 * MIN:
                        continue
                    y, P = g["y"][u], m.predict_proba(feats(g, parts)[u])
                    aucs = [E.auc(y == c, P[:, j]) for j, c in enumerate(cl) if (y == c).sum() >= MIN and (y != c).sum() >= MIN]
                    R[inp][n] = (float((np.array(cl)[P.argmax(1)] == y).mean()), float(np.mean(aucs)) if aucs else np.nan, len(aucs))
                    base[n] = (float(np.unique(y, return_counts=True)[1].max() / len(y)), int(len(np.unique(y))), float(u.mean()))
            print(f"  fold {k + 1} of {FOLDS} done ({len(taught)} classes taught), {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
        sc = sorted(base)
        if not sc:
            continue
        med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan  # noqa: E731
        print(f"  {len(sc)} squares scored; a median square has {int(np.median([base[n][1] for n in sc]))} taught classes, "
              f"{np.median([base[n][2] for n in sc]):.0%} of its kept pixels in them, and its commonest holds {np.median([base[n][0] for n in sc]):.0%}")
        print(f"  {'input':24}{'right':>8}{'ranked':>8}   against AlphaEarth, square by square: right, ranked (better in)")
        for inp in INPUTS:
            acc = np.array([R[inp][n][0] if n in R[inp] else np.nan for n in sc])
            auc = np.array([R[inp][n][1] if n in R[inp] else np.nan for n in sc])
            line = f"  {inp:24}{med(acc):>8.1%}{med(auc):>8.3f}"
            if inp != "AlphaEarth":
                da = acc - np.array([R["AlphaEarth"][n][0] for n in sc])
                du = auc - np.array([R["AlphaEarth"][n][1] for n in sc])
                du = du[np.isfinite(du)]
                line += f"   {np.nanmedian(da):+.1%} ({(da > 0).sum()} of {len(da)}), " + (f"{np.median(du):+.3f} ({(du > 0).sum()} of {len(du)})" if len(du) else "none")
            print(line)
        print("  each square: classes, commonest, then right for " + ", ".join(INPUTS))
        for n in sc:
            print(f"    {n:30}{base[n][1]:>4}{base[n][0]:>6.0%}  " + "".join(f"{R[i][n][0]:>7.0%}" if n in R[i] else f"{'':>7}" for i in INPUTS) + f"   {C[n][2]}")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
