# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "laspy[lazrs]", "urllib3"]
# ///
"""Do the raw lidar points add to the embeddings and the lidar terrain for finding wetlands?

Same three squares, labels and held-out blocks as embedding_signal.py and terrain_adds.py. Every point of the
3DEP point cloud over each square is read (USGS public EPT, one lidar project per square) and folded into numbers
per 10 m embedding pixel, node by node, so only a few nodes' points are in memory at a time:

  density     points per pixel (log), share classed ground, share classed water
  returns     share of pulses with a single return, mean number of returns per pulse
  intensity   mean and spread of log intensity of ground returns, mean of first returns, and the share of ground
              returns in ten intensity bands (bands set from a sample of the project's own points)
  height      share of returns in eight bands of height above the 1 m elevation model (under 0.15 m to over 20 m),
              mean height of first returns
  scan angle  mean absolute angle, so a head can allow for intensity falling off to the side

A small head (one hidden layer of 64) is taught on terrain, points, both, and each embedding alone, with terrain,
and with terrain and points. Scored on 1 km blocks it was not taught on (three folds).

Nothing is cached: about 6 billion points (about 42 GB) are read over the network each run, by --workers
processes. Peak memory about 8 GB; the process stops itself at 12 GB.

    uv run scripts/lpc_adds.py [--year 2024] [--workers 8] [--half-km 5] [--only "Lee, NH"]
"""
import argparse
import sys
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import ept  # noqa: E402
import terrain_adds as T  # noqa: E402

PROJECT = {"Delmarva, MD": "DE_Statewide_1_B23", "Stutsman County, ND": "ND_3DEPProcessing_5_D22",
           "Lee, NH": "NH_Coastal_1_2019"}
H_EDGES = np.array([0.15, 0.5, 1, 2, 5, 10, 20], np.float32)  # eight height bands, m
NI, NH = 10, len(H_EDGES) + 1
FETCHES, BATCH = 4, 16  # fetches at a time inside a worker, nodes handed to a worker at a time
SCAL = ["n", "n_first", "n_single", "sum_returns", "n_ground", "n_water", "sum_logi_ground", "sum_logi2_ground",
        "sum_logi_first", "sum_h_first", "n_h_first", "sum_angle"]
NAMES = (["density", "ground share", "water share", "single-return share", "returns per pulse", "ground intensity",
          "ground intensity spread", "first-return intensity", "first-return height", "scan angle"]
         + [f"intensity band {i}" for i in range(NI)] + [f"height band {i}" for i in range(NH)])
_tl = threading.local()
_job = {}


def _init(*args):
    """A worker process keeps the square's elevation model and grid for all its nodes."""
    _job["args"] = args


def _batch(keys):
    """A batch of nodes in a worker: fetched a few at a time over open connections, then folded one by one."""
    from concurrent.futures import ThreadPoolExecutor
    project, rc, box, dem, W, N, crs, i_lo, i_hi = _job["args"]
    if "pool" not in _job:
        _job["pool"] = ThreadPoolExecutor(FETCHES)
    out = []
    for key, raw in zip(keys, _job["pool"].map(lambda k: _try(ept.fetch_node, project, k), keys)):
        try:
            if isinstance(raw, Exception):
                raise raw
            out.append(node_stats(project, key, rc, box, dem, W, N, crs, i_lo, i_hi, raw))
        except Exception as e:  # noqa: BLE001
            out.append(repr(e))
    return out


def _try(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        return e


def node_stats(project, key, rc, box, dem, W, N, crs, i_lo, i_hi, raw=None):
    """One node folded onto the pixels it touches: (row0, col0, scalars (K, h, w), intensity bands, height bands,
    points used, bytes fetched), or None when no point falls in the square."""
    from pyproj import Transformer
    x, y, z, inten, rn, nr, cls, ang, nbytes = ept.read_node(project, key, raw)
    keep = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3]) & (cls != 7) & (cls != 18)  # noise out
    if not keep.any():
        return None, 0, nbytes
    x, y, z, inten, rn, nr, cls, ang = (v[keep] for v in (x, y, z, inten, rn, nr, cls, ang))
    y0, y1, x0, x1 = rc
    lon, lat = ept.to_lonlat(x, y)
    col = np.floor((lon - E.X0) / E.RES).astype(np.int64) - x0
    row = np.floor((E.Y0 - lat) / E.RES).astype(np.int64) - y0
    ok = (col >= 0) & (col < x1 - x0) & (row >= 0) & (row < y1 - y0)
    if not ok.any():
        return None, 0, nbytes
    if not ok.all():
        x, y, z, inten, rn, nr, cls, ang, col, row = (v[ok] for v in (x, y, z, inten, rn, nr, cls, ang, col, row))
    if not hasattr(_tl, "tf") or _tl.crs != crs:
        _tl.tf, _tl.crs = Transformer.from_crs(3857, crs, always_xy=True), crs
    ux, uy = _tl.tf.transform(x, y)
    dc, dr = np.floor((ux - W) / T.STEP).astype(np.int64), np.floor((N - uy) / T.STEP).astype(np.int64)
    inside = (dc >= 0) & (dc < dem.shape[1]) & (dr >= 0) & (dr < dem.shape[0])
    hag = np.full(len(z), np.nan, np.float32)
    hag[inside] = z[inside] - dem[dr[inside], dc[inside]]
    r0, c0 = row.min(), col.min()
    h, w = row.max() - r0 + 1, col.max() - c0 + 1
    li = (row - r0) * w + (col - c0)
    size = h * w
    cnt = lambda m, wt=None: np.bincount(li[m], None if wt is None else wt[m], size)  # noqa: E731
    logi = np.log2(inten.astype(np.float32) + 1)
    first, ground, hv = rn == 1, cls == 2, np.isfinite(hag)
    S = np.stack([np.bincount(li, minlength=size), cnt(first), cnt(first & (nr == 1)), cnt(first, nr.astype(np.float32)),
                  cnt(ground), cnt(cls == 9), cnt(ground, logi), cnt(ground, logi * logi), cnt(first, logi),
                  cnt(first & hv, np.clip(hag, -5, 80)), cnt(first & hv), np.bincount(li, np.abs(ang), size)]).reshape(len(SCAL), h, w)
    ib = np.clip(((logi[ground] - i_lo) / (i_hi - i_lo) * NI).astype(np.int64), 0, NI - 1)
    HI = np.bincount(li[ground] * NI + ib, minlength=size * NI).reshape(h, w, NI)
    hb = np.searchsorted(H_EDGES, hag[hv])
    HH = np.bincount(li[hv] * NH + hb, minlength=size * NH).reshape(h, w, NH)
    return (int(r0), int(c0), S, HI, HH), len(x), nbytes


def tree_gb():
    """Memory of this process and its workers."""
    import psutil
    p = psutil.Process()
    return (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 1e9


def read_points(name, s, dem, W, N, crs, threads, peak):
    """Per-pixel point numbers (h, w, len(NAMES)) for a square, and a mask of pixels that had any point."""
    project = PROJECT[name]
    meta = ept.info(project)
    x0, y0 = ept.to_3857(s["box"][0], s["box"][1])
    x1, y1 = ept.to_3857(s["box"][2], s["box"][3])
    box = (x0, y0, x1, y1)
    t0 = time.time()
    ns = ept.nodes(project, meta, box)
    total = sum(n for _, n in ns)
    print(f"  {name}: {project}, {len(ns):,} nodes, {total / 1e6:,.0f} M points in them, found in {time.time() - t0:.0f} s", flush=True)
    # intensity bands from a sample of this project's own points (1st to 99th percentile of log intensity)
    mid = [k for k, n in ns if n > 5000][:: max(len(ns) // 40, 1)][:40]
    li = np.concatenate([np.log2(ept.read_node(project, k)[3].astype(np.float32) + 1) for k in mid])
    i_lo, i_hi = (float(v) for v in np.percentile(li, [1, 99]))
    i_hi = max(i_hi, i_lo + 0.5)
    h, w = s["lab"].shape
    S, HI, HH = np.zeros((len(SCAL), h, w)), np.zeros((h, w, NI), np.int32), np.zeros((h, w, NH), np.int32)
    done = used = got = failed = 0
    t0 = last = time.time()
    with ProcessPoolExecutor(threads, initializer=_init, initargs=(project, s["rc"], box, dem, W, N, crs, i_lo, i_hi)) as ex:
        keys = [k for k, _ in ns]
        it = iter([keys[i:i + BATCH] for i in range(0, len(keys), BATCH)])
        live = set()

        def fill():
            while len(live) < threads * 2:
                k = next(it, None)
                if k is None:
                    return
                live.add(ex.submit(_batch, k))
        fill()
        while live:
            ready, _ = wait(live, return_when=FIRST_COMPLETED)
            live -= ready
            fill()
            for res in (r for f in ready for r in f.result()):
                done += 1
                if isinstance(res, str):
                    failed += 1
                    if failed <= 5:
                        print(f"    node failed: {res}", flush=True)
                    continue
                out, n, nb = res
                used, got = used + n, got + nb
                if out is not None:
                    r0, c0, a, b, c = out
                    hh, ww = a.shape[1:]
                    S[:, r0:r0 + hh, c0:c0 + ww] += a
                    HI[r0:r0 + hh, c0:c0 + ww] += b
                    HH[r0:r0 + hh, c0:c0 + ww] += c
            if time.time() - last > 60 or not live:
                last, el = time.time(), time.time() - t0
                print(f"    {done:,}/{len(ns):,} nodes, {used / 1e6:,.0f} M points used, {got / 1e9:.1f} GB at {got / el / 1e6:.1f} MB/s, "
                      f"{el / 60:.0f} min, about {el / done * (len(ns) - done) / 60:.0f} min left, memory {tree_gb():.1f} GB with workers, failed {failed}", flush=True)
    g = dict(zip(SCAL, S))
    n, nf, ng = np.maximum(g["n"], 1), np.maximum(g["n_first"], 1), np.maximum(g["n_ground"], 1)
    mi = g["sum_logi_ground"] / ng
    F = [np.log1p(g["n"]), g["n_ground"] / n, g["n_water"] / n, g["n_single"] / nf, g["sum_returns"] / nf, mi,
         np.sqrt(np.maximum(g["sum_logi2_ground"] / ng - mi * mi, 0)), g["sum_logi_first"] / nf,
         g["sum_h_first"] / np.maximum(g["n_h_first"], 1), g["sum_angle"] / n]
    F = np.concatenate([np.stack(F, -1), HI / np.maximum(HI.sum(-1, keepdims=True), 1),
                        HH / np.maximum(HH.sum(-1, keepdims=True), 1)], -1).astype(np.float32)
    ok = g["n"] > 0
    hgt = HH.sum((0, 1))
    print(f"  {name}: {used / 1e6:,.0f} M points on the square, {ok.mean():.1%} of pixels have points, median {np.median(g['n'][ok]):.0f} a pixel, "
          f"{(g['n_ground'] > 0).mean():.1%} have ground returns; returns under 0.15 m above the elevation model {hgt[0] / hgt.sum():.0%}, "
          f"over 5 m {hgt[5:].sum() / hgt.sum():.0%}; intensity bands {2 ** i_lo:.0f} to {2 ** i_hi:.0f}; failed nodes {failed}", flush=True)
    return np.nan_to_num(F), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--half-km", type=float, default=5.0)
    ap.add_argument("--only", action="append")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    E.HALF_KM = a.half_km
    if a.only:
        E.SQUARES = {k: v for k, v in E.SQUARES.items() if k in a.only}
    S = E.load_squares()
    for name, s in S.items():
        z, W, N, crs, used = T.read_dem(s["box"])
        s["T"], t_ok = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
        hole = np.isnan(z)
        if hole.any():
            from scipy.ndimage import distance_transform_edt
            z = z[tuple(distance_transform_edt(hole, return_distances=False, return_indices=True))]
        s["P"], p_ok = read_points(name, s, z, W, N, crs, a.workers, peak)
        s["x_ok"] = t_ok & p_ok
        del z

    cols = ["terrain", "points", "terrain + points"]
    for m in E.MODELS:
        cols += [m, m + " + terrain", m + " + terrain + points"]
    R = {t: {c: {} for c in cols} for t in T.TESTS}

    def run(col, name, X, ok):
        t0 = time.time()
        for t, v in T.score(X, S[name], ok, np.random.default_rng(0)).items():
            R[t][col][name] = v
        print(f"  {name}: {col} done, {time.time() - t0:.0f} s, memory {peak():.1f} GB", flush=True)

    for name, s in S.items():
        ok = s["x_ok"]
        run("terrain", name, s["T"][ok], ok)
        run("points", name, s["P"][ok], ok)
        run("terrain + points", name, np.concatenate([s["T"][ok], s["P"][ok]], 1), ok)
    for model in E.MODELS:
        hnd = E.open_model(model)
        for name, s in S.items():
            X, ok = E.read_model(model, hnd, a.year, s["rc"])
            ok &= s["x_ok"]
            X = X[ok]
            run(model, name, X, ok)
            X = np.concatenate([X, s["T"][ok]], 1)
            run(model + " + terrain", name, X, ok)
            X = np.concatenate([X, s["P"][ok]], 1)
            run(model + " + terrain + points", name, X, ok)
            del X

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
                print(f"    {c:32} {au:.3f} ({pr:.0%})")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
