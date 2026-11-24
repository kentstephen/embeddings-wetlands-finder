# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Does a head taught in two squares find wetlands in the third, with lidar terrain, an embedding, or both?
Then the head taught on all three is saved as plain weights.

Same squares, labels, terrain numbers and small head (one hidden layer of 64) as terrain_adds.py. Here no pixel of
the scored square is seen in teaching: the head is taught on the other two squares and scored on every vegetated
wetland and upland pixel of the third. Terrain numbers are in meters and slopes, so they mean the same thing in
every square; whether that is enough to carry a head across landscapes is the question.

The saved heads (models/wetland-head-<input>.npz, about 20 KB each) are taught on all three squares, with the
standardization folded into the first layer, so scoring needs only numpy:

    h = np.maximum(x @ W1 + b1, 0);  p = 1 / (1 + np.exp(-(h @ W2 + b2)))

x is the input named in the file's `inputs`: the embedding as read by embedding_signal.read_model, then the eleven
terrain numbers of terrain_adds.terrain.

Nothing fetched is cached. Peak memory about 8 GB; the process stops itself at 12 GB.

    uv run scripts/wetland_head.py [--year 2024] [--no-save]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import terrain_adds as T  # noqa: E402

TEACH = 60_000  # pixels taken from each teaching square
MODELS = Path(__file__).resolve().parent.parent / "models"
TESTS = ("vegetated wetland vs upland", f"the same, within {E.EDGE_M} m of the NWI line",
         "forested/shrub wetland vs upland, tree cover only")


def rows_of(s, ok, rng, n=TEACH):
    """Up to n vegetated wetland or upland pixels of a square, as indexes into its valid pixels, and their labels."""
    y = s["lab"][ok]
    r = np.flatnonzero(np.isin(y, E.VEGETATED) | (y == 0))
    r = rng.choice(r, min(n, len(r)), replace=False)
    return r, np.isin(y[r], E.VEGETATED)


def taught(D, S, names, rng):
    Xs, ys = [], []
    for n in names:
        X, ok = D[n]
        r, y = rows_of(S[n], ok, rng)
        Xs.append(X[r])
        ys.append(y)
    return T.head().fit(np.concatenate(Xs), np.concatenate(ys))


def leave_one_out(D, S):
    """For every square: scores from a head taught on the others. D: name -> (X of valid pixels, valid mask)."""
    out = {}
    for name, s in S.items():
        m = taught(D, S, [n for n in S if n != name], np.random.default_rng(0))
        X, ok = D[name]
        y_all, edge, cov = s["lab"][ok], s["edge"][ok], s["cover"][ok]
        up, veg = y_all == 0, np.isin(y_all, E.VEGETATED)
        use = veg | up
        p = np.full(len(y_all), np.nan, np.float32)
        p[use] = m.predict_proba(X[use])[:, 1]
        near = use & (edge <= E.EDGE_M)
        res = {TESTS[0]: (E.auc(veg[use], p[use]), E.prec_at_recall(veg[use], p[use])),
               TESTS[1]: (E.auc(veg[near], p[near]), E.prec_at_recall(veg[near], p[near]))}
        f = (cov == 10) & ((y_all == 2) | up)
        if (y_all[f] == 2).sum() >= 500:
            res[TESTS[2]] = (E.auc(y_all[f] == 2, p[f]), E.prec_at_recall(y_all[f] == 2, p[f]))
        out[name] = res
    return out


def save(col, D, S, year, inputs):
    """The head taught on all three squares, as two plain layers."""
    m = taught(D, S, list(S), np.random.default_rng(0))
    sc, mlp = m.steps[0][1], m.steps[1][1]
    W1 = (mlp.coefs_[0] / sc.scale_[:, None]).astype(np.float32)
    b1 = (mlp.intercepts_[0] - (sc.mean_ / sc.scale_) @ mlp.coefs_[0]).astype(np.float32)
    W2, b2 = mlp.coefs_[1][:, 0].astype(np.float32), np.float32(mlp.intercepts_[1][0])
    X = next(iter(D.values()))[0][:2000]
    p = 1 / (1 + np.exp(-(np.maximum(X @ W1 + b1, 0) @ W2 + b2)))
    gap = float(np.abs(p - m.predict_proba(X)[:, 1]).max())
    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"wetland-head-{col.lower().replace(' + ', '-')}.npz"
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2, inputs=np.array(inputs), year=year,
             taught_on=np.array(list(S)), label="vegetated wetland (NWI emergent, forested/shrub, tidal) against upland")
    print(f"  saved {path.relative_to(MODELS.parent)} ({path.stat().st_size / 1e3:.0f} KB), {len(inputs)} inputs; "
          f"largest gap between the plain weights and the fitted head on 2,000 pixels: {gap:.1e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    S = E.load_squares()
    for name, s in S.items():
        t = time.time()
        z, W, N, crs, _ = T.read_dem(s["box"])
        s["T"], s["t_ok"] = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
        print(f"  terrain {name}: {time.time() - t:.0f} s, peak {peak():.1f} GB", flush=True)
        del z

    cols = ["terrain"] + [c for m in E.MODELS for c in (m, m + " + terrain")]
    R = {}
    D = {n: (s["T"][s["t_ok"]], s["t_ok"]) for n, s in S.items()}
    R["terrain"] = leave_one_out(D, S)
    if not a.no_save:
        save("terrain", D, S, a.year, T.TERRAIN)
    for model in E.MODELS:
        t0 = time.time()
        hnd = E.open_model(model)
        De, Db = {}, {}
        for name, s in S.items():
            X, ok = E.read_model(model, hnd, a.year, s["rc"])
            ok &= s["t_ok"]
            De[name] = (X[ok], ok)
            Db[name] = (np.concatenate([X[ok], s["T"][ok]], 1), ok)
            del X
        R[model], R[model + " + terrain"] = leave_one_out(De, S), leave_one_out(Db, S)
        if not a.no_save:
            nb = De[next(iter(S))][0].shape[1]
            save(model + " + terrain", Db, S, a.year, [f"{model} {i}" for i in range(nb)] + T.TERRAIN)
        del De, Db
        print(f"  {model} done, {time.time() - t0:.0f} s, peak {peak():.1f} GB", flush=True)

    print("\nAUC (and, at 80 % of wetland found, the share of flagged pixels that are wetland). Small head taught on "
          "the other two squares, scored on every pixel of this one.")
    for t in TESTS:
        print(f"\n{t}")
        for name in S:
            if t not in R["terrain"][name]:
                continue
            print(f"  {name}")
            for c in cols:
                au, pr = R[c][name][t]
                print(f"    {c:24} {au:.3f} ({pr:.0%})")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
