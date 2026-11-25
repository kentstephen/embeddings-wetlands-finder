# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""In one head taught across the country, does 1 m lidar elevation beat the national 10 m elevation?

national_elev.py showed the 10 m elevation helps the national three-class head. fine_labels.py compared 1 m lidar
with 10 m only for heads taught inside one state. Here the national head is taught three times on the same pixels,
in every square of the 76 where 3DEP has 1 m elevation: AlphaEarth alone, AlphaEarth with the eleven terrain
numbers from the 10 m elevation, and AlphaEarth with the same numbers from 1 m lidar. A pixel is kept only where
both elevations are real ground within 300 m (the widest window the terrain numbers read), so no number is worked
out from filled-in ground. Scored two ways, as national_elev.py:

  1. Pixels against NWI in squares held out whole (six folds).
  2. Wetlands of four states' finer maps as objects, in the held-out squares fine_labels.py uses, by heads taught
     on every kept square.

    uv run scripts/national_lidar.py              # everything, about two and a half hours
    uv run scripts/national_lidar.py --limit 2    # a pilot: a few squares, one state square
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
import national_elev as NE  # noqa: E402
import nwi_tiles as N  # noqa: E402
import terrain_adds as T  # noqa: E402
import water_head as W  # noqa: E402

YEARS = W.YEARS
YEAR = W.YEAR
KEEP = W.KEEP
TEACH = W.TEACH
FOLDS = W.FOLDS
MIN = W.MIN
REACH = 300  # meters: the widest window of terrain_adds.terrain
HEADS = (("alone", None), ("10 m", "T10"), ("1 m", "T1"))


def terrain_on(read, box, rc):
    """The eleven terrain numbers of one elevation source on the AlphaEarth pixels of a square, with a mask of the
    pixels whose ground is real everywhere within REACH, and the share of the grid with no elevation."""
    from scipy.ndimage import uniform_filter
    z, Wx, Nn, crs, _ = read(box)
    hole = np.isnan(z)
    gap = float(hole.mean())
    k = max(int(round(10 / T.STEP)), 1)
    near = uniform_filter(hole.astype(np.float32), 2 * int(round(REACH / T.STEP)) + 1)[k // 2::k, k // 2::k]
    del hole
    Tz = T.terrain(z)
    del z
    h, w = min(Tz.shape[0], near.shape[0]), min(Tz.shape[1], near.shape[1])
    both, ok = T.on_aef(np.dstack([Tz[:h, :w], near[:h, :w, None]]), Wx, Nn, crs, rc)
    Tn = both[..., :-1]
    ok &= (both[..., -1] < 1e-6) & np.isfinite(Tn).all(-1)
    return Tn.astype(np.float32), ok, gap


def gather(lon, lat, kind):
    """KEEP pixels of one square: class from NWI, AlphaEarth for each year, the terrain numbers from both elevations.
    Only pixels with all of them are kept, so the three heads see the same pixels. A string says why a square was
    dropped."""
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        try:
            T1, ok, gap1 = terrain_on(T.read_dem, box, rc)
        except Exception as e:  # noqa: BLE001
            return f"no 1 m elevation: {type(e).__name__}: {str(e)[:120]}"
        if gap1 > 0.6:
            return f"1 m elevation missing over {gap1:.0%}"
        T10, ok10, gap10 = terrain_on(F.dem10, box, rc)
        ok &= ok10
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
        used = len(rows) / ok.size
        if len(rows) < 10_000:
            return f"only {len(rows)} pixels with an embedding, a label and both elevations (1 m missing over {gap1:.0%})"
        share = [float((cls.ravel()[rows] == c).mean()) for c in range(3)]
        rows = np.sort(np.random.default_rng(0).choice(rows, min(KEEP, len(rows)), replace=False))
        pick = lambda a: a.reshape(lab.size, -1)[rows].squeeze()  # noqa: E731
        return dict(kind=kind, share=share, gap1=gap1, gap10=gap10, used=used, y=pick(cls), T1=pick(T1), T10=pick(T10),
                    X={y: pick(v) for y, v in X.items()}, secs=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}: {str(e)[:160]}"


def feats(g, year, key):
    x = g["X"][year].astype(np.float32)
    return np.hstack([x, g[key]]) if key else x


def teach(G, names, key):
    """A three-class head taught on TEACH pixels of each named square, spread evenly over the years. The same pixels
    are drawn for every head."""
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for n in names:
        g = G[n]
        r = rng.choice(len(g["y"]), min(TEACH, len(g["y"])), replace=False)
        for part, y in zip(np.array_split(r, len(YEARS)), YEARS):
            part = np.sort(part)
            Xs.append(feats(g, y, key)[part])
            ys.append(g["y"][part])
    return T.head().fit(np.concatenate(Xs), np.concatenate(ys))


def state_square(st, lon, lat):
    """A held-out state square read with each elevation in turn (fine_labels.gather), scored only on pixels both
    have. Returns the square once per elevation, or a string saying why it was dropped."""
    out = {}
    for elev in ("lidar", "10m"):
        F.ELEV = elev
        g = F.gather(st, lon, lat, True)
        if isinstance(g, str):
            return f"{elev}: {g}"
        out[elev] = g
    a, b = out["lidar"], out["10m"]
    ok = a["ok"] & b["ok"]
    back = a["back"] & b["back"]
    b = {**a, "T": b["T"]}
    for g in (a, b):
        g["ok"], g["back"] = ok, back
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(16)
    t0 = time.time()
    C = {**M.CANDIDATES, **W.ORDINARY}
    if a.limit:
        C = {**dict(list(M.CANDIDATES.items())[:a.limit]), **dict(list(W.ORDINARY.items())[-a.limit:])}
    hit = NE.overlaps(C)
    for n, sq in hit:
        print(f"  left out of teaching: {n} touches the held-out square {sq}", flush=True)
    C = {n: v for n, v in C.items() if n not in {h[0] for h in hit}}
    print(f"elevation: 3DEP 1 m lidar (read at 2 m) and 3DEP 1/3 arc-second (about 10 m), same pixels; {len(C)} squares to read", flush=True)

    G = {}
    for i, (name, (lon, lat, kind)) in enumerate(C.items()):
        g = gather(lon, lat, kind)
        if isinstance(g, str):
            print(f"  dropped {i + 1} of {len(C)}, {name}: {g}; {(time.time() - t0) / 60:.0f} min", flush=True)
            continue
        G[name] = g
        print(f"  teaching {i + 1} of {len(C)}, {name} ({kind}): wetland {g['share'][1]:.1%}, water {g['share'][2]:.1%}, no 1 m over "
              f"{g['gap1']:.0%}, pixels usable {g['used']:.0%}, {g['secs']:.0f} s; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
    names = list(G)
    K = min(FOLDS, len(names))
    fold = {n: i % K for i, n in enumerate(names)}
    print(f"\n{len(names)} squares kept of {len(C)} (the rest have no 1 m elevation), {K} folds, memory {peak():.1f} GB", flush=True)

    # 1. pixels against NWI, squares held out whole
    R = {}
    for k in range(K):
        tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
        ms = {tag: teach(G, tr, key) for tag, key in HEADS}
        for n in te:
            g = G[n]
            y = g["y"]
            land = y < 2
            scored = (y == 1).sum() >= MIN and (y == 0).sum() >= MIN
            wscored = (y == 2).sum() >= MIN
            r = {}
            for tag, key in HEADS:
                P = NE.probs(ms[tag], feats(g, YEAR, key), len(y))
                call = P.argmax(1)
                r["auc", tag] = E.auc(y[land] == 1, P[land, 1]) if scored else np.nan
                r["pr", tag] = E.prec_at_recall(y[land] == 1, P[land, 1]) if scored else np.nan
                r["w", tag] = E.auc(y == 2, P[:, 2]) if wscored else np.nan
                r["fw", tag] = float(((y == 0) & (call == 1)).sum() / max((y == 0).sum(), 1))
            R[n] = r
        print(f"  fold {k + 1} of {K} done, {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)

    groups = (("wetland landscapes", [n for n in names if n in M.CANDIDATES]),
              ("ordinary landscapes", [n for n in names if n in W.ORDINARY]))
    tags = [t for t, _ in HEADS]
    print(f"\n1. Pixels against NWI in held-out squares, {YEAR}. Median (lowest to highest), over the squares with enough pixels to score.")
    for title, ns in groups:
        if not ns:
            continue
        print(f"\n  {title} ({len(ns)} squares)")
        for what, key, f in (("wetland vs upland AUC", "auc", ".3f"), ("flagged that is wetland", "pr", ".0%"),
                             ("open water AUC", "w", ".3f"), ("upland called wetland", "fw", ".1%")):
            for t in tags:
                print(f"    {what + ', ' + t:38}{W.med([R[n][key, t] for n in ns], f)}")
        for p, q in (("10 m", "alone"), ("1 m", "alone"), ("1 m", "10 m")):
            d = np.array([R[n]["auc", p] - R[n]["auc", q] for n in ns])
            d = d[np.isfinite(d)]
            if len(d):
                print(f"    wetland AUC, {p} against {q}, square by square: median {np.median(d):+.3f} ({d.min():+.3f} to {d.max():+.3f}), "
                      f"better in {(d > 0).sum()} of {len(d)}")
    print("\n  Each square: wetland share, pixels usable, wetland AUC alone / 10 m / 1 m (flagged that is wetland), 1 m less 10 m")
    for _, ns in groups:
        for n in ns:
            r = R[n]
            print(f"    {n:30}{G[n]['share'][1]:>6.1%}{G[n]['used']:>6.0%}" + "".join(f"{r['auc', t]:>8.3f} ({r['pr', t]:>4.0%})" for t in tags)
                  + f"{r['auc', '1 m'] - r['auc', '10 m']:>+8.3f}")

    # 2. small wetlands of the states' finer maps, by heads taught on every kept square
    full = {tag: teach(G, names, key) for tag, key in HEADS}
    del G
    h = np.load(F.HEAD)
    saved = (h["W1"], h["b1"], h["W2"], h["b2"])
    print("\n2. Wetlands of four states' finer maps, in the held-out squares of fine_labels.py", flush=True)
    for name, st in F.STATES.items():
        t1, t10 = [], []
        for sq, (lon, lat) in (list(st["test"].items())[:1] if a.limit else st["test"].items()):
            g = state_square(st, lon, lat)
            what = g if isinstance(g, str) else f"{len(g[0]['idx'])} wetlands"
            print(f"  {name} held out, {sq}: {what}; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
            if not isinstance(g, str):
                t1.append(g[0])
                t10.append(g[1])
        if t1:
            Rs = {"saved": F.score(saved, t1, False), "alone": F.score(full["alone"], t1, False),
                  "with 10 m": F.score(full["10 m"], t10, True), "with 1 m": F.score(full["1 m"], t1, True)}
            F.show(name, Rs)
            F.show(f"{name}, only wetlands NWI does not have", Rs, ~Rs["saved"]["in_nwi"])
        del t1, t10
        if a.limit:
            break
    print(f"\n  saved: the head in the map (76 squares, AlphaEarth alone). The other three are taught here on the same pixels of the "
          f"{len(names)} squares with 1 m elevation: alone (AlphaEarth), with 10 m and with 1 m (AlphaEarth and the terrain numbers "
          "from that elevation).")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
