# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""Inside a wetland, do the embeddings say how long it stays wet, and what kind it is?

Every other script here asks wetland or not. NWI's code for a polygon says more: its class (emergent, forested,
shrub) and its water regime, how long water stands in a typical year. This script asks whether a small head can
read those from a pixel, in squares it was never taught on, and which input reads them best.

Only vegetated freshwater (palustrine) wetland pixels are used. Water regimes are grouped by how long the ground
is wet:

  temporary       A, J   flooded briefly in the growing season
  saturated       B, D   wet to the surface, seldom flooded
  seasonal        C, E   flooded for long stretches, dry by the end of most growing seasons
  semipermanent   F, G, H  flooded through the growing season in most years, or always

Inputs compared on the same pixels: AlphaEarth, TESSERA and OlmoEarth alone, the eleven terrain numbers from 10 m
elevation alone, and AlphaEarth with them. With --s2 a spring and a late-summer Sentinel-2 composite (nine numbers
each, leafoff_adds.composite) are read too, alone and beside AlphaEarth, to see whether a look at two seasons adds
what a yearly embedding blends.

Squares are the candidates of many_squares.py, dealt into six folds; every square is scored by heads that saw no
pixel of it. Scores, per held-out square, then the median over squares:

  regime        each regime against the others it shares the square with (AUC), averaged
  one class     the same within emergent pixels alone and within forested pixels alone, so the head cannot lean on
                forest and marsh having different usual regimes
  neighbors     temporary against seasonal, and seasonal against semipermanent
  kind          emergent, forested, shrub, each against the others (AUC), averaged

Nothing is fitted ahead and nothing is saved. Squares are read by --workers processes of up to 8 GB each.

    uv run scripts/water_regime.py [--workers 2] [--limit 4] [--s2] [--only "Yazoo, MS"]
"""
import argparse
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import fine_labels as F  # noqa: E402
import many_squares as M  # noqa: E402
import nwi_tiles as N  # noqa: E402
import terrain_adds as T  # noqa: E402

MODELS = ("AlphaEarth", "TESSERA", "OlmoEarth")
YEAR = 2024
PER_CELL = 2_500    # pixels kept per class and regime in a square
TEACH = 6_000       # pixels taught from each square, spread evenly over the regimes it has
FOLDS = 6
MIN = 150           # pixels a group needs in a held-out square to be scored
REGIME = {"A": 1, "J": 1, "B": 2, "D": 2, "C": 3, "E": 3, "F": 4, "G": 4, "H": 4}
RNAME = {1: "temporary", 2: "saturated", 3: "seasonal", 4: "semipermanent"}
CLASS = {"EM": 1, "FO": 2, "SS": 3}
CNAME = {1: "emergent", 2: "forested", 3: "shrub"}
CODE = re.compile(r"^P([A-Z]{2})\d*(?:/(?:[A-Z]{2})?\d*)*([A-Z])?")


def parse(code):
    """(class, regime group) of a palustrine NWI code, 0 where it has none we use: PFO1/SS1C is forested, seasonal."""
    m = CODE.match(code or "")
    if not m:
        return 0, 0
    return CLASS.get(m.group(1), 0), REGIME.get(m.group(2) or "", 0)


def gather(name, lon, lat, s2):
    """Kept pixels of one square: class, regime and every input. A string says why a square was dropped."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    warnings.filterwarnings("ignore")
    E.guard(8)
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        y0, y1, x0, x1 = rc
        shp = (y1 - y0, x1 - x0)
        tr = Affine(E.RES, 0, E.X0 + x0 * E.RES, 0, -E.RES, E.Y0 - y0 * E.RES)
        polys = N.read_codes(box)
        # larger pieces first, so a small polygon keeps its pixels inside a large one; 255 marks a polygon we do not use
        shapes = [(g, (lambda c, r: c * 8 + r if c and r else 255)(*parse(code))) for g, _, code in polys]
        lab = rasterize(shapes, out_shape=shp, transform=tr, fill=0, dtype="uint8") if shapes else np.zeros(shp, np.uint8)
        cls, reg = np.where(lab < 255, lab // 8, 0), np.where(lab < 255, lab % 8, 0)
        ok = (cls > 0) & (reg > 0)
        if ok.sum() < 2 * MIN:
            return f"{int(ok.sum())} vegetated palustrine pixels with a regime"
        X, miss = {}, []
        for m in MODELS:
            try:
                a, k = E.read_model(m, E.open_model(m), YEAR, rc)
                if k[ok].mean() < 0.5:
                    raise RuntimeError(f"data on {k[ok].mean():.0%} of the pixels")
                ok &= k
                X[m] = a.astype(np.float16)
                del a
            except Exception as e:  # noqa: BLE001
                miss.append(f"{m}: {type(e).__name__} {str(e)[:60]}")
        z, Wx, Nn, crs, _ = F.dem10(box)
        if np.isnan(z).mean() > 0.6:
            return "10 m elevation missing over most of the square"
        Tn, k = T.on_aef(T.terrain(z), Wx, Nn, crs, rc)
        del z
        ok &= k & np.isfinite(Tn).all(-1)
        X["terrain"] = Tn.astype(np.float32)
        if s2:
            import leafoff_adds as L
            parts = []
            for season in ("leaf-off", "leaf-on"):
                a, k, _ = L.composite(box, rc, YEAR, season, 12)
                ok &= k
                parts.append(a)
            X["two seasons"] = np.concatenate(parts, -1).astype(np.float32)
        rng = np.random.default_rng(0)
        rows = []
        for c in CNAME:
            for r in RNAME:
                i = np.flatnonzero((ok & (cls == c) & (reg == r)).ravel())
                rows.append(rng.choice(i, min(PER_CELL, len(i)), replace=False))
        rows = np.sort(np.concatenate(rows))
        if len(rows) < 2 * MIN:
            return "too few pixels with every input"
        pick = lambda a: a.reshape(lab.size, -1)[rows].squeeze()  # noqa: E731
        return dict(cls=pick(cls).astype(np.uint8), reg=pick(reg).astype(np.uint8), X={k: pick(v) for k, v in X.items()},
                    miss=miss, secs=time.time() - t0,
                    count={(c, r): int((ok & (cls == c) & (reg == r)).sum()) for c in CNAME for r in RNAME})
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}: {str(e)[:160]}"


def feats(g, parts):
    return np.hstack([g["X"][p].astype(np.float32).reshape(len(g["reg"]), -1) for p in parts])


def teach(G, names, parts, key, groups):
    """A head on TEACH pixels of each square, the same number from each group the square has."""
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for n in names:
        g = G[n]
        have = [k for k in groups if (g[key] == k).sum() >= 20]
        for k in have:
            i = np.flatnonzero(g[key] == k)
            i = np.sort(rng.choice(i, min(TEACH // len(have), len(i)), replace=False))
            Xs.append(feats(g, parts)[i])
            ys.append(g[key][i])
    return T.head().fit(np.concatenate(Xs), np.concatenate(ys))


def ovr(y, P, classes, keep=None):
    """Each group against the others in a square (AUC), averaged over the groups with enough pixels on both sides."""
    keep = np.ones(len(y), bool) if keep is None else keep
    out = []
    for j, k in enumerate(classes):
        a, b = keep & (y == k), keep & (y != k)
        if a.sum() >= MIN and b.sum() >= MIN:
            out.append(E.auc(y[keep] == k, P[keep, j]))
    return float(np.mean(out)) if out else np.nan


def pair(y, P, classes, lo, hi, keep=None):
    """One group against another (AUC): the head's probability of the second over the two."""
    keep = (np.ones(len(y), bool) if keep is None else keep) & np.isin(y, (lo, hi))
    if (keep & (y == lo)).sum() < MIN or (keep & (y == hi)).sum() < MIN or lo not in classes or hi not in classes:
        return np.nan
    a, b = P[keep, list(classes).index(lo)], P[keep, list(classes).index(hi)]
    return E.auc(y[keep] == hi, b / (a + b + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only")
    ap.add_argument("--s2", action="store_true", help="read spring and late-summer Sentinel-2 too (slow)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    F.ELEV = "10m"
    peak = E.guard(10)
    t0 = time.time()
    C = {n: v for n, v in M.CANDIDATES.items() if not a.only or n in a.only.split(";")}
    if a.limit:
        C = dict(list(C.items())[:a.limit])
    G = {}
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(gather, n, lon, lat, a.s2): n for n, (lon, lat, _) in C.items()}
        for i, j in enumerate(as_completed(jobs)):
            n, g = jobs[j], j.result()
            if isinstance(g, str):
                print(f"  dropped {n}: {g}", flush=True)
                continue
            G[n] = g
            c = g["count"]
            print(f"  {i + 1} of {len(C)}, {n} ({C[n][2]}): " + ", ".join(f"{RNAME[r]} {sum(c[(k, r)] for k in CNAME):,}" for r in RNAME)
                  + (f"; missing {'; '.join(g['miss'])}" if g["miss"] else "") + f"; {g['secs']:.0f} s, {(time.time() - t0) / 60:.0f} min", flush=True)
    names = sorted(G)
    INPUTS = {"AlphaEarth": ["AlphaEarth"], "TESSERA": ["TESSERA"], "OlmoEarth": ["OlmoEarth"], "terrain": ["terrain"],
              "AlphaEarth + terrain": ["AlphaEarth", "terrain"], "all three": list(MODELS)}
    if a.s2:
        INPUTS.update({"two seasons": ["two seasons"], "AlphaEarth + two seasons": ["AlphaEarth", "two seasons"]})
    full = [n for n in names if all(m in G[n]["X"] for m in MODELS)]
    print(f"\n{len(names)} squares kept of {len(C)}; {len(full)} have all three embeddings. Scored on those {len(full)}, {YEAR}, "
          f"each by heads that never saw it ({FOLDS} folds).", flush=True)
    if len(full) < 2:
        return
    K = min(FOLDS, len(full))
    fold = {n: i % K for i, n in enumerate(full)}
    TASKS = ["regime", "regime, emergent only", "regime, forested only", "temporary vs seasonal", "seasonal vs semipermanent",
             "seasonal vs semipermanent, emergent", "temporary vs seasonal, forested", "kind"]
    R = {inp: {t: {} for t in TASKS} for inp in INPUTS}
    for k in range(K):
        tr, te = [n for n in full if fold[n] != k], [n for n in full if fold[n] == k]
        for inp, parts in INPUTS.items():
            mr, mk = teach(G, tr, parts, "reg", RNAME), teach(G, tr, parts, "cls", CNAME)
            for n in te:
                g = G[n]
                x = feats(g, parts)
                P, cl = mr.predict_proba(x), list(mr.classes_)
                y, c = g["reg"], g["cls"]
                r = R[inp]
                r["regime"][n] = ovr(y, P, cl)
                r["regime, emergent only"][n] = ovr(y, P, cl, c == 1)
                r["regime, forested only"][n] = ovr(y, P, cl, c == 2)
                r["temporary vs seasonal"][n] = pair(y, P, cl, 1, 3)
                r["seasonal vs semipermanent"][n] = pair(y, P, cl, 3, 4)
                r["seasonal vs semipermanent, emergent"][n] = pair(y, P, cl, 3, 4, c == 1)
                r["temporary vs seasonal, forested"][n] = pair(y, P, cl, 1, 3, c == 2)
                r["kind"][n] = ovr(c, mk.predict_proba(x), list(mk.classes_))
        print(f"  fold {k + 1} of {K} done, {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)

    base = "AlphaEarth"
    print("\nMedian AUC over held-out squares (squares that could be scored). Chance is .500.")
    print(f"  {'':38}" + "".join(f"{i[:22]:>24}" for i in INPUTS))
    for t in TASKS:
        cells = []
        for inp in INPUTS:
            v = np.array([R[inp][t][n] for n in full])
            cells.append(f"{np.nanmedian(v):.3f} ({np.isfinite(v).sum()})" if np.isfinite(v).any() else "none")
        print(f"  {t:38}" + "".join(f"{c:>24}" for c in cells))
    print(f"\nAgainst {base}, square by square: median change, and squares where the input is better")
    for t in TASKS:
        cells = []
        for inp in INPUTS:
            if inp == base:
                cells.append("")
                continue
            d = np.array([R[inp][t][n] - R[base][t][n] for n in full])
            d = d[np.isfinite(d)]
            cells.append(f"{np.median(d):+.3f}, {(d > 0).sum()} of {len(d)}" if len(d) else "none")
        print(f"  {t:38}" + "".join(f"{c:>24}" for c in cells))
    print("\nEach square, regime AUC: " + ", ".join(i for i in INPUTS))
    for n in full:
        print(f"  {n:28}" + "".join(f"{R[i]['regime'][n]:>8.3f}" for i in INPUTS) + f"   {C[n][2]}")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
