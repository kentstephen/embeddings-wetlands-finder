# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "torch"]
# ///
"""Does reading the elevation itself still help when the head is taught on many landscapes?

patch_head.py found that a small network reading elevation pictures did a little better than the eleven terrain
summaries, with three squares. This asks the same over the squares of many_squares.py: the same candidates, the
same six folds, and every square scored by a head that saw no pixel of it or of the other squares in its fold.

For each kept square a random 16,000 vegetated wetland and upland pixels are kept, each with its eleven
summaries, its two elevation pictures (2 m cells and 10 m cells, 64 x 64, as in patch_head.py) and its 2024
AlphaEarth and OlmoEarth embeddings. A head is taught on 4,000 pixels from each teaching square. Compared:
summaries, pictures, and each embedding with summaries and with pictures.

Nothing fetched is cached. The pictures are held in memory as half-precision numbers, about 0.26 GB a square.
Peak memory about 20 GB with one worker; the process stops itself at 30 GB.

    uv run scripts/many_pictures.py [--workers 1] [--only "Lee, NH"] [--limit 8] [--epochs 8]
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
import terrain_adds as T  # noqa: E402

KEEP, TEACH, YEAR, BATCH = 16_000, 4_000, 2024, 256
P, COARSE, FINE_M, COARSE_M = 64, 5, 0.25, 1.0  # as in patch_head.py


def gather(name, lon, lat, kind):
    """Everything kept from one square, or a string saying why it was dropped."""
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt
    warnings.filterwarnings("ignore")
    E.guard(12)
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        lab = E.burn(E.read_nwi(box), rc)
        up, veg = lab == 0, np.isin(lab, E.VEGETATED)
        share = veg.sum() / max((veg | up).sum(), 1)
        if not M.LOW <= share <= M.HIGH or (veg | up).mean() < 0.3:
            return name, f"NWI vegetated wetland is {share:.1%} of the land, or the square is mostly water"
        z, W, N, crs, _ = T.read_dem(box)
        empty = float(np.isnan(z).mean())
        if empty > 0.05:
            return name, f"1 m elevation missing over {empty:.0%}"
        Tn, ok = T.on_aef(T.terrain(z), W, N, crs, rc)
        hole = np.isnan(z)
        if hole.any():
            z = z[tuple(distance_transform_edt(hole, return_distances=False, return_indices=True))]
        y0, y1, x0, x1 = rc
        LON, LAT = np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)
        Xm, Ym = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
        r, c = np.floor((N - Ym) / T.STEP).astype(np.int64), np.floor((Xm - W) / T.STEP).astype(np.int64)
        h, w = z.shape[0] // COARSE, z.shape[1] // COARSE
        z10 = z[:h * COARSE, :w * COARSE].reshape(h, COARSE, w, COARSE).mean((1, 3))
        half = P // 2
        ok &= (r // COARSE >= half) & (r // COARSE < h - half) & (c // COARSE >= half) & (c // COARSE < w - half)
        m_lat = E.RES * 111_320
        px = (m_lat * np.cos(np.radians(lat)), m_lat)
        wet = lab > 0
        edge = np.where(wet, distance_transform_edt(wet, sampling=px[::-1]), distance_transform_edt(~wet, sampling=px[::-1]))
        cover = E.read_worldcover(rc)
        X = {}
        for model in M.MODELS:
            a, k = E.read_model(model, E.open_model(model), YEAR, rc)
            ok &= k
            X[model] = a.astype(np.float16)
            del a
        rows = np.flatnonzero((ok & (veg | up)).ravel())
        rows = np.sort(np.random.default_rng(0).choice(rows, min(KEEP, len(rows)), replace=False))
        pick = lambda a: a.reshape(len(lab.ravel()), -1)[rows].squeeze()  # noqa: E731
        r, c = r.ravel()[rows], c.ravel()[rows]
        off = np.arange(-half, half)
        A, C = np.empty((len(rows), P, P), np.float16), np.empty((len(rows), P, P), np.float16)
        for j in range(0, len(rows), 2000):
            rr, cc = r[j:j + 2000, None, None], c[j:j + 2000, None, None]
            here = z[rr, cc]
            A[j:j + 2000] = np.arcsinh((z[rr + off[None, :, None], cc + off[None, None, :]] - here) / FINE_M)
            C[j:j + 2000] = np.arcsinh((z10[rr // COARSE + off[None, :, None], cc // COARSE + off[None, None, :]] - here) / COARSE_M)
        return name, dict(kind=kind, share=float(share), y=pick(veg), lab=pick(lab), edge=pick(edge).astype(np.float32),
                          cover=pick(cover), T=pick(Tn), X={k: pick(v) for k, v in X.items()}, A=A, C=C,
                          secs=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return name, f"failed: {type(e).__name__}: {str(e)[:160]}"


def fit(A, C, y, emb, epochs, dev, seed=0):
    """patch_head's network taught on pictures already cut out (half precision, on the device)."""
    import torch
    from torch import nn
    import patch_head as PH
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    val, tr = order[:len(y) // 10], order[len(y) // 10:]
    At, Ct = torch.as_tensor(A, device=dev), torch.as_tensor(C, device=dev)
    yt = torch.as_tensor(y.astype(np.float32), device=dev)
    et = None if emb is None else torch.as_tensor(emb, device=dev)
    net = (PH.Net() if emb is None else PH.Net(emb[tr].mean(0), emb[tr].std(0) + 1e-6)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tr) // BATCH + 1))
    loss_of = nn.BCEWithLogitsLoss()
    # the eight flips and turns of a square, each as a lookup of rows and columns
    i = torch.arange(P, device=dev)
    turns = []
    for k in range(8):
        ry, cx = (P - 1 - i if k & 2 else i)[:, None].expand(P, P), (P - 1 - i if k & 1 else i)[None, :].expand(P, P)
        turns.append((ry.T, cx.T) if k & 4 else (ry, cx))
    vt = torch.as_tensor(val, device=dev)
    best, keep = np.inf, None
    for _ in range(epochs):
        net.train()
        tt = torch.as_tensor(rng.permutation(tr), device=dev)
        pickt = rng.integers(8, size=len(tt) // BATCH + 1)
        for n, j in enumerate(range(0, len(tt), BATCH)):
            b = tt[j:j + BATCH]
            ry, cx = turns[int(pickt[n])]
            loss = loss_of(net(At[b][:, ry, cx].float()[:, None], Ct[b][:, ry, cx].float()[:, None],
                               None if et is None else et[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        tot = torch.zeros((), device=dev)
        with torch.no_grad():
            for j in range(0, len(vt), 2048):
                b = vt[j:j + 2048]
                tot += loss_of(net(At[b].float()[:, None], Ct[b].float()[:, None], None if et is None else et[b]), yt[b]) * len(b)
        if float(tot) < best:
            best, keep = float(tot), {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(keep)
    return net.eval()


def predict(net, A, C, emb, dev):
    import torch
    out = []
    with torch.no_grad():
        for j in range(0, len(A), 4096):
            e = None if emb is None else torch.as_tensor(emb[j:j + 4096], device=dev)
            out.append(torch.sigmoid(net(torch.as_tensor(A[j:j + 4096], device=dev).float()[:, None],
                                         torch.as_tensor(C[j:j + 4096], device=dev).float()[:, None], e)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(30)
    C = {k: v for k, v in M.CANDIDATES.items() if not a.only or k in a.only}
    C = dict(list(C.items())[:a.limit]) if a.limit else C
    G, t0 = {}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(gather, n, *v) for n, v in C.items()]):
            name, g = f.result()
            if isinstance(g, str):
                print(f"  dropped {name}: {g}", flush=True)
                continue
            G[name] = g
            print(f"  {name} ({g['kind']}): vegetated wetland {g['share']:.1%} of land, {len(g['y']):,} pixels kept, "
                  f"{g['secs']:.0f} s; {len(G)} squares after {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
    G = {n: G[n] for n in C if n in G}
    names = list(G)
    K = min(M.FOLDS, len(names))
    fold = {n: i % K for i, n in enumerate(names)}
    print(f"\n{len(names)} squares kept of {len(C)}, {K} folds, memory {peak():.1f} GB", flush=True)
    if len(names) < 4:
        print("too few squares to hold any out")
        return

    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    INPUTS = ["summaries", "pictures"] + [c for m in M.MODELS for c in (m + " + summaries", m + " + pictures")]
    R = {}

    def emb_of(g, col, rows=slice(None)):
        model = col.split(" + ")[0]
        return g["X"][model][rows].astype(np.float32) if model in M.MODELS else None

    for col in INPUTS:
        pics = col.endswith("pictures")
        for k in range(K):
            tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
            rng = np.random.default_rng(0)
            take = {n: np.sort(rng.choice(len(G[n]["y"]), min(TEACH, len(G[n]["y"])), replace=False)) for n in tr}
            y = np.concatenate([G[n]["y"][take[n]] for n in tr])
            e = None if emb_of(G[tr[0]], col) is None else np.concatenate([emb_of(G[n], col, take[n]) for n in tr])
            if pics:
                net = fit(np.concatenate([G[n]["A"][take[n]] for n in tr]), np.concatenate([G[n]["C"][take[n]] for n in tr]),
                          y, e, a.epochs, dev)
            else:
                Ts = np.concatenate([G[n]["T"][take[n]] for n in tr])
                m = T.head().fit(Ts if e is None else np.concatenate([e, Ts], 1), y)
            for n in te:
                g = G[n]
                en = emb_of(g, col)
                p = (predict(net, g["A"], g["C"], en, dev) if pics
                     else m.predict_proba(g["T"] if en is None else np.concatenate([en, g["T"]], 1))[:, 1])
                near = g["edge"] <= E.EDGE_M
                fo = (g["cover"] == 10) & ((g["lab"] == 2) | (g["lab"] == 0))
                R[col, n] = (E.auc(g["y"], p), E.prec_at_recall(g["y"], p), E.auc(g["y"][near], p[near]),
                             E.auc(g["lab"][fo] == 2, p[fo]) if (g["lab"][fo] == 2).sum() >= 300 else np.nan)
        print(f"  {col} done, {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)

    print(f"\nHead taught on the other folds' squares ({YEAR}), scored on each held-out square. Median over squares "
          "(lowest to highest).")
    print(f"   {'':24}{'AUC':>26}{'flagged that is wetland':>28}{'AUC within 30 m':>26}{'AUC forest':>26}")
    for col in INPUTS:
        v = np.array([R[col, n] for n in names])
        print(f"   {col:24}{M.med(v[:, 0]):>26}{M.pct(v[:, 1]):>28}{M.med(v[:, 2]):>26}{M.med(v[:, 3]):>26}")
    print("\n   Squares where pictures beat summaries, of all squares (AUC, then flagged that is wetland):")
    for a_, b_ in (("summaries", "pictures"), ("AlphaEarth + summaries", "AlphaEarth + pictures"),
                   ("OlmoEarth + summaries", "OlmoEarth + pictures")):
        d = np.array([[R[b_, n][i] - R[a_, n][i] for i in (0, 1)] for n in names])
        print(f"   {b_:24} {(d[:, 0] > 0).sum()} of {len(names)}, median change {np.median(d[:, 0]):+.3f};  "
              f"{(d[:, 1] > 0).sum()} of {len(names)}, median change {np.median(d[:, 1]):+.0%}")
    print("\n   Each square: AUC (share of flagged pixels that are wetland at 80 % found)")
    print(f"   {'':26}{'wetland':>8}" + "".join(f"{c.replace('Earth', '').replace('summaries', 'sum').replace('pictures', 'pic'):>15}" for c in INPUTS))
    for n in names:
        print(f"   {n:26}{G[n]['share']:>8.0%}" + "".join(f"{R[c, n][0]:>9.3f} ({R[c, n][1]:>3.0%})" for c in INPUTS))
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
