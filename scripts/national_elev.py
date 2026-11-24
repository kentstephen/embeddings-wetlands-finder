# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""Does the national 10 m elevation model help one head taught across the country?

fine_labels.py showed that eleven terrain numbers from 3DEP's 1/3 arc-second elevation roughly double the small
wetlands a head finds, when the head is taught inside the state it is scored in. Here the three-class head of
water_head.py (upland, vegetated wetland, open water; NWI labels; the same 76 squares of the lower 48) is taught
twice, on AlphaEarth alone and on AlphaEarth with those terrain numbers, and scored two ways:

  1. Pixels against NWI in squares held out whole (six folds over the 76 squares).
  2. Wetlands of four states' finer maps as objects, in the held-out squares fine_labels.py uses, by heads taught
     on all 76 squares. None of the 76 overlaps a square it is scored on (checked at the start).

    uv run scripts/national_elev.py              # everything, about an hour
    uv run scripts/national_elev.py --limit 3    # a pilot: a few squares, one state square, nothing saved
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import fine_labels as F  # noqa: E402
import many_squares as M  # noqa: E402
import nwi_tiles as N  # noqa: E402
import terrain_adds as T  # noqa: E402
import water_head as W  # noqa: E402

YEARS = W.YEARS
YEAR = W.YEAR
KEEP = W.KEEP
TEACH = W.TEACH
FOLDS = W.FOLDS
MIN = W.MIN
OUT = Path(__file__).resolve().parent.parent / "models" / "wetland-water-head-alphaearth-elev10m.npz"


def gather(lon, lat, kind):
    """KEEP pixels of one square: class from NWI, AlphaEarth for each year, the eleven terrain numbers from the 10 m
    elevation. Only pixels with all of them are kept, so both heads see the same pixels. A string says why a square
    was dropped."""
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        z, Wx, Nn, crs, _ = F.dem10(box)
        gap = float(np.isnan(z).mean())
        if gap > 0.6:
            return f"10 m elevation missing over {gap:.0%}"
        Tn, ok = T.on_aef(T.terrain(z), Wx, Nn, crs, rc)
        del z
        ok &= np.isfinite(Tn).all(-1)
        lab = E.burn(N.read(box), rc)
        cls = np.full(lab.shape, 255, np.uint8)  # 255: NWI "other", left out
        cls[lab == 0] = 0
        cls[np.isin(lab, E.VEGETATED)] = 1
        cls[np.isin(lab, W.WATER)] = 2
        ok &= cls != 255
        hnd = E.open_model("AlphaEarth")
        X = {}
        for y in YEARS:
            a, k = E.read_model("AlphaEarth", hnd, y, rc)
            ok &= k
            X[y] = a.astype(np.float16)
            del a
        rows = np.flatnonzero(ok.ravel())
        if len(rows) < 10_000:
            return "almost no pixels with an embedding, a label and elevation"
        share = [float((cls.ravel()[rows] == c).mean()) for c in range(3)]
        rows = np.sort(np.random.default_rng(0).choice(rows, min(KEEP, len(rows)), replace=False))
        pick = lambda a: a.reshape(lab.size, -1)[rows].squeeze()  # noqa: E731
        return dict(kind=kind, share=share, gap=gap, y=pick(cls), T=pick(Tn).astype(np.float32),
                    X={y: pick(v) for y, v in X.items()}, secs=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}: {str(e)[:160]}"


def feats(g, year, elev):
    x = g["X"][year].astype(np.float32)
    return np.hstack([x, g["T"]]) if elev else x


def teach(G, names, elev):
    """A three-class head taught on TEACH pixels of each named square, spread evenly over the years. The same pixels
    are drawn with and without elevation."""
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for n in names:
        g = G[n]
        r = rng.choice(len(g["y"]), min(TEACH, len(g["y"])), replace=False)
        for part, y in zip(np.array_split(r, len(YEARS)), YEARS):
            part = np.sort(part)
            Xs.append(feats(g, y, elev)[part])
            ys.append(g["y"][part])
    return T.head().fit(np.concatenate(Xs), np.concatenate(ys))


def probs(m, x, n):
    P = m.predict_proba(x)
    return np.stack([P[:, list(m.classes_).index(c)] if c in m.classes_ else np.zeros(n) for c in range(3)], 1)


def overlaps(C):
    """Teaching squares whose box touches a state square the heads are scored on."""
    hit = []
    for n, v in C.items():
        a = E.box_of(v[0], v[1])
        for st in F.STATES.values():
            for sq, (lon, lat) in st["test"].items():
                b = E.box_of(lon, lat)
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    hit.append((n, sq))
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    F.ELEV = "10m"
    peak = E.guard(14)
    t0 = time.time()
    C = {**M.CANDIDATES, **W.ORDINARY}
    if a.limit:
        C = {**dict(list(M.CANDIDATES.items())[:a.limit]), **dict(list(W.ORDINARY.items())[-a.limit:])}
    hit = overlaps(C)
    for n, sq in hit:
        print(f"  left out of teaching: {n} touches the held-out square {sq}", flush=True)
    C = {n: v for n, v in C.items() if n not in {h[0] for h in hit}}
    print(f"elevation: 3DEP 1/3 arc-second, about 10 m; {len(C)} squares to read", flush=True)

    G = {}
    for i, (name, (lon, lat, kind)) in enumerate(C.items()):
        g = gather(lon, lat, kind)
        if isinstance(g, str):
            print(f"  dropped {name}: {g}", flush=True)
            continue
        G[name] = g
        print(f"  teaching {i + 1} of {len(C)}, {name} ({kind}): wetland {g['share'][1]:.1%}, water {g['share'][2]:.1%}, no elevation over "
              f"{g['gap']:.0%}, {g['secs']:.0f} s; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
    names = list(G)
    K = min(FOLDS, len(names))
    fold = {n: i % K for i, n in enumerate(names)}
    print(f"\n{len(names)} squares kept of {len(C)}, {K} folds, memory {peak():.1f} GB", flush=True)

    # 1. pixels against NWI, squares held out whole
    R = {}
    for k in range(K):
        tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
        ms = {e: teach(G, tr, e) for e in (False, True)}
        for n in te:
            g = G[n]
            y = g["y"]
            land = y < 2
            scored = (y == 1).sum() >= MIN and (y == 0).sum() >= MIN
            wscored = (y == 2).sum() >= MIN
            r = {}
            for e, tag in ((False, "a"), (True, "e")):
                P = probs(ms[e], feats(g, YEAR, e), len(y))
                call = P.argmax(1)
                r["auc" + tag] = E.auc(y[land] == 1, P[land, 1]) if scored else np.nan
                r["pr" + tag] = E.prec_at_recall(y[land] == 1, P[land, 1]) if scored else np.nan
                r["w" + tag] = E.auc(y == 2, P[:, 2]) if wscored else np.nan
                r["fw" + tag] = float(((y == 0) & (call == 1)).sum() / max((y == 0).sum(), 1))
            R[n] = r
        print(f"  fold {k + 1} of {K} done, {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)

    groups = (("wetland landscapes", [n for n in names if n in M.CANDIDATES]),
              ("ordinary landscapes", [n for n in names if n in W.ORDINARY]))
    print(f"\n1. Pixels against NWI in held-out squares, {YEAR}. Median (lowest to highest), and the squares with enough pixels to score.")
    for title, ns in groups:
        if not ns:
            continue
        v = lambda key: [R[n][key] for n in ns]  # noqa: E731
        d = np.array([R[n]["auce"] - R[n]["auca"] for n in ns])
        d = d[np.isfinite(d)]
        print(f"\n  {title} ({len(ns)} squares)")
        print(f"    wetland vs upland AUC, AlphaEarth alone      {W.med(v('auca'))}")
        print(f"    wetland vs upland AUC, with 10 m elevation   {W.med(v('auce'))}")
        if len(d):
            print(f"    change square by square                      median {np.median(d):+.3f} ({d.min():+.3f} to {d.max():+.3f}), better in {(d > 0).sum()} of {len(d)}")
        print(f"    flagged that is wetland, AlphaEarth alone    {W.med(v('pra'), '.0%')}")
        print(f"    flagged that is wetland, with elevation      {W.med(v('pre'), '.0%')}")
        print(f"    open water AUC, AlphaEarth alone             {W.med(v('wa'))}")
        print(f"    open water AUC, with elevation               {W.med(v('we'))}")
        print(f"    upland called wetland, AlphaEarth alone      {W.med(v('fwa'), '.1%')}")
        print(f"    upland called wetland, with elevation        {W.med(v('fwe'), '.1%')}")
    print("\n  Each square: wetland share, wetland AUC alone and with elevation (flagged that is wetland), change")
    for _, ns in groups:
        for n in ns:
            r = R[n]
            print(f"    {n:30}{G[n]['share'][1]:>6.1%}{r['auca']:>8.3f} ({r['pra']:>4.0%}){r['auce']:>8.3f} ({r['pre']:>4.0%}){r['auce'] - r['auca']:>+8.3f}")

    # 2. small wetlands of the states' finer maps, by heads taught on every square
    full = {e: teach(G, names, e) for e in (False, True)}
    del G
    h = np.load(F.HEAD)
    saved = (h["W1"], h["b1"], h["W2"], h["b2"])
    print("\n2. Wetlands of four states' finer maps, in the held-out squares of fine_labels.py", flush=True)
    for name, st in F.STATES.items():
        tests = []
        for sq, (lon, lat) in (list(st["test"].items())[:1] if a.limit else st["test"].items()):
            g = F.gather(st, lon, lat, True)
            what = g if isinstance(g, str) else f"{len(g['idx'])} wetlands"
            print(f"  {name} held out, {sq}: {what}; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
            if not isinstance(g, str):
                tests.append(g)
        if tests:
            Rs = {"saved": F.score(saved, tests, False), "alone": F.score(full[False], tests, False),
                  "with elev": F.score(full[True], tests, True)}
            F.show(name, Rs)
            F.show(f"{name}, only wetlands NWI does not have", Rs, ~Rs["saved"]["in_nwi"])
        del tests
        if a.limit:
            break
    print("\n  saved: the head in the map (76 squares, AlphaEarth alone). alone: the same refit here on pixels that also have "
          "elevation. with elev: AlphaEarth and the terrain numbers.")

    if not a.limit:
        Wp = W.plain(full[True])
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, W1=Wp[0], b1=Wp[1], W2=Wp[2], b2=Wp[3],
                 inputs=np.array([f"AlphaEarth {i}" for i in range(64)] + [f"terrain: {t}" for t in T.TERRAIN]),
                 classes=np.array(W.CLASSES), years=np.array(YEARS), taught_on=np.array(names),
                 elevation="USGS 3DEP 1/3 arc-second, put on a 2 m UTM grid (bilinear), terrain_adds.terrain",
                 label="NWI: upland (no polygon), vegetated wetland (emergent, forested/shrub, tidal), "
                       "open water (pond, lake, riverine, tidal deepwater)")
        print(f"\n  saved {OUT.relative_to(OUT.parent.parent)} ({OUT.stat().st_size / 1e3:.0f} KB), taught on {len(names)} squares")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
