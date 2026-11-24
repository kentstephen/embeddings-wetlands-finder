# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile"]
# ///
"""A head for a map: upland, vegetated wetland and open water from the AlphaEarth embedding alone.

The earlier heads left open water out of both sides, so they were never shown a lake. This one is taught three
classes from NWI polygons (read from the tiles on Source Cooperative, see nwi_tiles.py): upland (no polygon), vegetated wetland (emergent, forested/shrub, tidal) and
open water (pond, lake, riverine, tidal deepwater). It reads no lidar, so it can be run anywhere the embedding is.

Squares are the candidates of many_squares.py, and a second group of ordinary landscapes that were not picked for
their wetlands (desert, city, cropland, mountains, large water), so the head is also taught and scored on ground
with little or no wetland. No square is dropped for its wetland share or its elevation coverage.

Squares are dealt into six folds and every square is scored by a head that saw no pixel of it. Reported per
held-out square, in 2024:

  - AUC of vegetated wetland against upland, and the share of flagged pixels that are wetland at 80 % found, from
    the three-class head and from a two-class head taught without water (the earlier setup, to see what adding
    water costs)
  - AUC of open water against everything else
  - each class's share called correctly when every pixel takes its most probable class
  - how the pixels called water line up with ESA WorldCover's permanent water class (a second answer key)

Then the head taught on every square is saved as plain weights (models/wetland-water-head-alphaearth.npz), with
the standardization folded into the first layer:

    h = np.maximum(x @ W1 + b1, 0);  z = h @ W2 + b2;  p = np.exp(z - z.max(1, keepdims=True));  p /= p.sum(1, keepdims=True)

x is the 64 AlphaEarth numbers as read by embedding_signal.read_model; the columns of p are upland, vegetated
wetland, open water. Teaching pixels are spread over four years of embeddings against the one set of NWI labels.

Nothing fetched is cached. Squares are read by --workers processes of up to 6 GB each.

    uv run scripts/water_head.py [--workers 1] [--limit 6] [--no-save]
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import many_squares as M  # noqa: E402
import nwi_tiles as N  # noqa: E402
import terrain_adds as T  # noqa: E402

MODEL = "AlphaEarth"
YEARS = (2018, 2020, 2022, 2024)
YEAR = 2024
KEEP = 50_000          # pixels kept per square
TEACH = 6_000          # pixels taught from each square
FOLDS = 6
WATER = (3, 4, 5, 7)   # NWI pond, lake, riverine, tidal deepwater
CLASSES = ("upland", "vegetated wetland", "open water")
MIN = 300              # pixels of a class a square needs before that class is scored in it
OUT = Path(__file__).resolve().parent.parent / "models" / "wetland-water-head-alphaearth.npz"

# Landscapes not picked for wetlands. name: (center lon, center lat, landscape)
ORDINARY = {
    "Sonoran Desert, AZ": (-112.60, 32.90, "desert"),
    "Great Basin, NV": (-116.80, 39.50, "sagebrush basin"),
    "Phoenix, AZ": (-112.07, 33.50, "desert city"),
    "Chicago suburbs, IL": (-87.95, 41.85, "city"),
    "Atlanta, GA": (-84.40, 33.80, "city under trees"),
    "Corn Belt, IL": (-88.90, 40.40, "drained cropland"),
    "High Plains, KS": (-100.90, 38.50, "irrigated cropland"),
    "Central Valley orchards, CA": (-119.60, 36.40, "irrigated orchards"),
    "Front Range, CO": (-105.65, 40.05, "high mountains"),
    "Cascades, WA": (-121.40, 47.50, "conifer mountains"),
    "Ozarks, MO": (-91.60, 37.20, "upland hardwood forest"),
    "Ridge and Valley, VA": (-79.60, 37.90, "forested ridges"),
    "Hill Country, TX": (-99.20, 30.30, "dry woodland"),
    "Pine plantations, AL": (-87.30, 32.40, "pine plantations"),
    "Lake Michigan shore, WI": (-87.75, 43.40, "large lake shore"),
    "Chesapeake shore, MD": (-76.35, 38.75, "tidal shore"),
}


def gather(name, lon, lat, kind):
    """Everything kept from one square, or a string saying why it was dropped."""
    warnings.filterwarnings("ignore")
    E.guard(6)
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        lab = E.burn(N.read(box), rc)
        cls = np.full(lab.shape, 255, np.uint8)  # 255: NWI "other", left out
        cls[lab == 0] = 0
        cls[np.isin(lab, E.VEGETATED)] = 1
        cls[np.isin(lab, WATER)] = 2
        cover = E.read_worldcover(rc)
        ok = cls != 255
        hnd = E.open_model(MODEL)
        X = {}
        for y in YEARS:
            a, k = E.read_model(MODEL, hnd, y, rc)
            ok &= k
            X[y] = a.astype(np.float16)
            del a
        rows = np.flatnonzero(ok.ravel())
        if len(rows) < 10_000:
            return name, "almost no pixels with an embedding and a label"
        share = [float((cls.ravel()[rows] == c).mean()) for c in range(3)]
        rows = np.sort(np.random.default_rng(0).choice(rows, min(KEEP, len(rows)), replace=False))
        pick = lambda a: a.reshape(lab.size, -1)[rows].squeeze()  # noqa: E731
        return name, dict(kind=kind, share=share, y=pick(cls), cover=pick(cover), X={y: pick(v) for y, v in X.items()},
                          secs=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return name, f"failed: {type(e).__name__}: {str(e)[:160]}"


def teach(G, names, rng, water=True):
    """A head taught on TEACH pixels of each named square, spread evenly over the years. Without water it is the
    two-class head of the earlier tests."""
    Xs, ys = [], []
    for n in names:
        g = G[n]
        r = np.arange(len(g["y"])) if water else np.flatnonzero(g["y"] < 2)
        r = rng.choice(r, min(TEACH, len(r)), replace=False)
        for part, y in zip(np.array_split(r, len(YEARS)), YEARS):
            Xs.append(g["X"][y][np.sort(part)].astype(np.float32))
            ys.append(g["y"][np.sort(part)])
    return T.head().fit(np.concatenate(Xs), np.concatenate(ys))


def plain(m):
    """The fitted head as two plain layers, standardization folded into the first."""
    sc, mlp = m.steps[0][1], m.steps[1][1]
    W1 = (mlp.coefs_[0] / sc.scale_[:, None]).astype(np.float32)
    b1 = (mlp.intercepts_[0] - (sc.mean_ / sc.scale_) @ mlp.coefs_[0]).astype(np.float32)
    return W1, b1, mlp.coefs_[1].astype(np.float32), mlp.intercepts_[1].astype(np.float32)


def run_plain(W1, b1, W2, b2, x):
    z = np.maximum(x @ W1 + b1, 0) @ W2 + b2
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def med(v, f=".3f"):
    v = np.array([x for x in v if np.isfinite(x)])
    return f"{np.median(v):{f}} ({v.min():{f}} to {v.max():{f}}), {len(v)} squares" if len(v) else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    C = {**M.CANDIDATES, **ORDINARY}
    if a.limit:  # a pilot: some of each group
        C = {**dict(list(M.CANDIDATES.items())[:a.limit]), **dict(list(ORDINARY.items())[-a.limit:])}
    G, t0 = {}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(gather, n, *v) for n, v in C.items()]):
            name, g = f.result()
            if isinstance(g, str):
                print(f"  dropped {name}: {g}", flush=True)
                continue
            G[name] = g
            print(f"  {name} ({g['kind']}): upland {g['share'][0]:.0%}, wetland {g['share'][1]:.1%}, water {g['share'][2]:.1%}, "
                  f"{g['secs']:.0f} s; {len(G)} of {len(C)} after {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
    names = [n for n in C if n in G]
    K = min(FOLDS, len(names))
    fold = {n: i % K for i, n in enumerate(names)}
    print(f"\n{len(names)} squares kept of {len(C)}, {K} folds, memory {peak():.1f} GB", flush=True)

    R, conf, wc = {}, np.zeros((3, 3)), np.zeros(3)
    for k in range(K):
        tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
        m3 = teach(G, tr, np.random.default_rng(0))
        m2 = teach(G, tr, np.random.default_rng(0), water=False)
        for n in te:
            g = G[n]
            X, y = g["X"][YEAR].astype(np.float32), g["y"]
            P = m3.predict_proba(X)
            P = np.stack([P[:, list(m3.classes_).index(c)] if c in m3.classes_ else np.zeros(len(y)) for c in range(3)], 1)
            land = y < 2
            scored = (y == 1).sum() >= MIN and (y == 0).sum() >= MIN
            wscored = (y == 2).sum() >= MIN
            call = P.argmax(1)
            cm = np.array([[((y == i) & (call == j)).sum() for j in range(3)] for i in range(3)], float)
            conf += cm
            w80, cw = g["cover"] == 80, call == 2
            wc += ((w80 & cw).sum(), w80.sum(), cw.sum())
            R[n] = dict(
                auc3=E.auc(y[land] == 1, P[land, 1]) if scored else np.nan,
                pr3=E.prec_at_recall(y[land] == 1, P[land, 1]) if scored else np.nan,
                auc2=E.auc(y[land] == 1, m2.predict_proba(X[land])[:, 1]) if scored else np.nan,
                pr2=E.prec_at_recall(y[land] == 1, m2.predict_proba(X[land])[:, 1]) if scored else np.nan,
                aucw=E.auc(y == 2, P[:, 2]) if wscored else np.nan,
                right=[cm[i, i] / cm[i].sum() if cm[i].sum() >= MIN else np.nan for i in range(3)],
                false_wet=cm[0, 1] / max(cm[0].sum(), 1), false_water=cm[0, 2] / max(cm[0].sum(), 1))
        print(f"  fold {k + 1} of {K} done, {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)

    groups = (("wetland landscapes", [n for n in names if n in M.CANDIDATES]),
              ("ordinary landscapes", [n for n in names if n in ORDINARY]))
    print(f"\nHeld-out squares, {YEAR}. Median (lowest to highest), and the number of squares with enough pixels to score.")
    for title, ns in groups:
        if not ns:
            continue
        v = lambda key: [R[n][key] for n in ns]  # noqa: E731
        print(f"\n  {title} ({len(ns)} squares)")
        print(f"    wetland vs upland AUC, three classes   {med(v('auc3'))}")
        print(f"    wetland vs upland AUC, taught no water {med(v('auc2'))}")
        print(f"    flagged that is wetland, three classes {med(v('pr3'), '.0%')}")
        print(f"    flagged that is wetland, no water      {med(v('pr2'), '.0%')}")
        print(f"    open water vs the rest, AUC            {med(v('aucw'))}")
        for i, c in enumerate(CLASSES):
            print(f"    {c} called {c:18}  {med([R[n]['right'][i] for n in ns], '.0%')}")
        print(f"    upland called wetland                  {med(v('false_wet'), '.1%')}")
        print(f"    upland called water                    {med(v('false_water'), '.1%')}")

    print("\n  All held-out pixels together, most probable class. Rows: NWI. Columns: called upland, wetland, water.")
    for i, c in enumerate(CLASSES):
        print(f"    {c:18}" + "".join(f"{x:>8.1%}" for x in conf[i] / conf[i].sum()) + f"   ({conf[i].sum() / conf.sum():.1%} of pixels)")
    print(f"  Of the pixels called a class, the share NWI has in that class: "
          + ", ".join(f"{c} {conf[i, i] / conf[:, i].sum():.0%}" for i, c in enumerate(CLASSES)))
    print(f"  WorldCover permanent water: {wc[0] / max(wc[1], 1):.0%} of it is called water; "
          f"{wc[0] / max(wc[2], 1):.0%} of what is called water is WorldCover water")

    print("\n  Each square: wetland share, water share, wetland AUC (flagged that is wetland), water AUC, upland called wetland")
    for title, ns in groups:
        for n in ns:
            r, s = R[n], G[n]["share"]
            print(f"    {n:28}{s[1]:>6.1%}{s[2]:>7.1%}{r['auc3']:>8.3f} ({r['pr3']:>4.0%}){r['aucw']:>8.3f}{r['false_wet']:>8.1%}")

    if not a.no_save and not a.limit:
        m = teach(G, names, np.random.default_rng(0))
        W = plain(m)
        x = np.concatenate([G[n]["X"][YEAR][:200] for n in names]).astype(np.float32)
        gap = float(np.abs(run_plain(*W, x) - m.predict_proba(x)).max())
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, W1=W[0], b1=W[1], W2=W[2], b2=W[3], inputs=np.array([f"{MODEL} {i}" for i in range(x.shape[1])]),
                 classes=np.array(CLASSES), years=np.array(YEARS), taught_on=np.array(names),
                 label="NWI: upland (no polygon), vegetated wetland (emergent, forested/shrub, tidal), "
                       "open water (pond, lake, riverine, tidal deepwater)")
        print(f"\n  saved {OUT.relative_to(OUT.parent.parent)} ({OUT.stat().st_size / 1e3:.0f} KB), taught on {len(names)} squares; "
              f"largest gap between the plain weights and the fitted head on {len(x):,} pixels: {gap:.1e}")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
