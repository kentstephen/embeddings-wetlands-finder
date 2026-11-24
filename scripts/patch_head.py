# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "torch"]
# ///
"""Does a head that reads the elevation itself do better than a head that reads eleven terrain summaries?

Same three squares, labels and held-out blocks as terrain_adds.py, and the same two-squares-teach, third-scores
test as wetland_head.py. In place of the eleven summaries, a small convolutional network reads two pictures of the
3DEP elevation around each 10 m pixel, both 64 x 64 cells and both as height against the pixel's own ground:

  fine     2 m cells, 128 m across
  coarse   10 m cells, 640 m across

Each picture goes through four convolutions to 64 numbers; the two sets, and the embedding when there is one, go
through one hidden layer of 64, as in the summaries' head. Teaching pictures are flipped and turned at random,
since a wetland does not care which way is north.

The pictures of a scored pixel reach into ground the head was taught on, and the other way round, as the
summaries do too (they read up to 300 m around). What is never shared is a label.

Compared, on the same pixels: terrain summaries, elevation pictures, and each of AlphaEarth and OlmoEarth alone,
with summaries, and with pictures.

Nothing fetched is cached. Peak memory about 8 GB; the process stops itself at 14 GB.

    uv run scripts/patch_head.py [--year 2024] [--epochs 8]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import terrain_adds as T  # noqa: E402
import wetland_head as H  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
P, COARSE = 64, 5  # cells across a picture; fine cells in a coarse cell
FINE_M, COARSE_M = 0.25, 1.0  # heights are squashed with asinh(height / this), so 5 cm and 5 m both register
EMB = ("AlphaEarth", "OlmoEarth")
OFF = None


def prep(s):
    """Terrain summaries for a square (as in terrain_adds), and what the network needs: the elevation at 2 m and
    10 m, and for every embedding pixel the cell of each that its center falls in."""
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt
    z, W, N, crs, _ = T.read_dem(s["box"])
    s["T"], s["t_ok"] = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
    hole = np.isnan(z)
    if hole.any():
        z = z[tuple(distance_transform_edt(hole, return_distances=False, return_indices=True))]
    y0, y1, x0, x1 = s["rc"]
    LON, LAT = np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)
    X, Y = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
    r, c = np.floor((N - Y) / T.STEP).astype(np.int64), np.floor((X - W) / T.STEP).astype(np.int64)
    h, w = z.shape[0] // COARSE, z.shape[1] // COARSE
    z10 = z[:h * COARSE, :w * COARSE].reshape(h, COARSE, w, COARSE).mean((1, 3))
    half = P // 2
    ok = ((r // COARSE >= half) & (r // COARSE < h - half) & (c // COARSE >= half) & (c // COARSE < w - half))
    r, c = np.clip(r, half, z.shape[0] - half - 1), np.clip(c, half, z.shape[1] - half - 1)
    s["net"] = dict(z=z, z10=z10.astype(np.float32), r=r.ravel(), c=c.ravel(),
                    r10=np.clip(r // COARSE, half, h - half - 1).ravel(), c10=np.clip(c // COARSE, half, w - half - 1).ravel())
    s["p_ok"] = s["t_ok"] & ok


def world(nets):
    """Every square's elevation laid end to end on the device, so one batch can draw on several squares."""
    Wd = dict(Z=torch.as_tensor(np.concatenate([g["z"].ravel() for g in nets]), device=DEV),
              Z10=torch.as_tensor(np.concatenate([g["z10"].ravel() for g in nets]), device=DEV), sq=[])
    o2 = o10 = 0
    for g in nets:
        Wd["sq"].append(dict(o2=o2, w2=g["z"].shape[1], o10=o10, w10=g["z10"].shape[1],
                             **{k: g[k] for k in ("r", "c", "r10", "c10")}))
        o2, o10 = o2 + g["z"].size, o10 + g["z10"].size
        del g["z"], g["z10"]
    return Wd


def places(Wd, sid, rows):
    """For each sample (its square and its pixel there): where its center sits in the long elevation arrays and
    how wide its square's rows are, fine then coarse, on the device."""
    i2, w2, i10, w10 = (np.empty(len(rows), np.int64) for _ in range(4))
    for k in np.unique(sid):
        m, q = sid == k, Wd["sq"][k]
        i2[m], w2[m] = q["o2"] + q["r"][rows[m]] * q["w2"] + q["c"][rows[m]], q["w2"]
        i10[m], w10[m] = q["o10"] + q["r10"][rows[m]] * q["w10"] + q["c10"][rows[m]], q["w10"]
    return tuple(torch.as_tensor(v, device=DEV) for v in (i2, w2, i10, w10))


def pictures(Wd, pl, b, turn=0):
    """The fine and coarse pictures (n, 1, P, P) for samples b (a device tensor of indexes into the places).
    turn (0 to 7) picks one of the eight flips and turns of a square, done in the lookup itself."""
    i2, w2, i10, w10 = (v[b] for v in pl)
    dy, dx = OFF[None, :, None], OFF[None, None, :]
    if turn & 1:
        dx = -dx - 1
    if turn & 2:
        dy = -dy - 1
    if turn & 4:
        dy, dx = dx.transpose(1, 2), dy.transpose(1, 2)
    here = Wd["Z"][i2][:, None, None]
    a = Wd["Z"][i2[:, None, None] + dy * w2[:, None, None] + dx] - here
    c = Wd["Z10"][i10[:, None, None] + dy * w10[:, None, None] + dx] - here
    return torch.asinh(a / FINE_M)[:, None], torch.asinh(c / COARSE_M)[:, None]


def branch():
    L = []
    for i, o in ((1, 16), (16, 32), (32, 48)):
        L += [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
    # the pixel is at the middle of the picture, so the last map is kept whole, not averaged
    return nn.Sequential(*L, nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Dropout(0.2),
                         nn.Linear(32 * (P // 8) ** 2, 64), nn.ReLU())


class Net(nn.Module):
    def __init__(self, mu=None, sd=None):
        super().__init__()
        self.fine, self.coarse = branch(), branch()
        n = 0 if mu is None else len(mu)
        if n:
            self.register_buffer("mu", torch.as_tensor(mu))
            self.register_buffer("sd", torch.as_tensor(sd))
        self.head = nn.Sequential(nn.Linear(128 + n, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, a, b, e=None):
        f = [self.fine(a), self.coarse(b)]
        if e is not None:
            f.append((e - self.mu) / self.sd)
        return self.head(torch.cat(f, 1))[:, 0]


def fit(Wd, sid, rows, y, emb, epochs, seed=0):
    """A network taught on the given pixels. A tenth of them is set aside to pick the best epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    val, tr = order[:len(y) // 10], order[len(y) // 10:]
    pl = places(Wd, sid, rows)
    yt = torch.as_tensor(y.astype(np.float32), device=DEV)
    et = None if emb is None else torch.as_tensor(emb, device=DEV)
    net = (Net() if emb is None else Net(emb[tr].mean(0), emb[tr].std(0) + 1e-6)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tr) // BATCH + 1))
    loss_of = nn.BCEWithLogitsLoss()
    vt = torch.as_tensor(val, device=DEV)
    best, keep = np.inf, None
    for _ in range(epochs):
        net.train()
        tt = torch.as_tensor(rng.permutation(tr), device=DEV)
        turns = rng.integers(8, size=len(tt) // BATCH + 1)  # one of the eight flips and turns of a square per batch
        for n, j in enumerate(range(0, len(tt), BATCH)):
            b = tt[j:j + BATCH]
            a, c = pictures(Wd, pl, b, int(turns[n]))
            loss = loss_of(net(a, c, None if et is None else et[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        tot = torch.zeros((), device=DEV)
        with torch.no_grad():
            for j in range(0, len(vt), 2048):
                b = vt[j:j + 2048]
                a, c = pictures(Wd, pl, b)
                tot += loss_of(net(a, c, None if et is None else et[b]), yt[b]) * len(b)
        if float(tot) < best:
            best, keep = float(tot), {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(keep)
    return net.eval()


def predict(net, Wd, k_sq, rows, emb):
    pl = places(Wd, np.full(len(rows), k_sq), rows)
    et = None if emb is None else torch.as_tensor(emb, device=DEV)
    out = []
    with torch.no_grad():
        for j in range(0, len(rows), 4096):
            b = torch.arange(j, min(j + 4096, len(rows)), device=DEV)
            a, c = pictures(Wd, pl, b)
            out.append(torch.sigmoid(net(a, c, None if et is None else et[b])))
    return torch.cat(out).cpu().numpy()


def held_out(nets, k_sq, pix, y, fold, inside, emb, epochs, rng):
    """As terrain_adds.held_out, with the network in place of the small head. pix: each row's pixel in the grid."""
    p = np.full(len(y), np.nan, np.float32)
    for k in range(T.FOLDS):
        tr, te = np.flatnonzero(inside & (fold != k)), np.flatnonzero(inside & (fold == k))
        if len(te) and 0 < y[tr].sum() < len(tr):
            if len(tr) > T.TRAIN_MAX:
                tr = rng.choice(tr, T.TRAIN_MAX, replace=False)
            net = fit(nets, np.full(len(tr), k_sq), pix[tr], y[tr], None if emb is None else emb[tr], epochs)
            p[te] = predict(net, nets, k_sq, pix[te], None if emb is None else emb[te])
    return p


def score(nets, k_sq, s, ok, emb, epochs, rng):
    """As terrain_adds.score: the three tests inside a square, each on blocks the network was not taught on."""
    pix = np.flatnonzero(ok.ravel())
    y_all, f, edge, cov = s["lab"][ok], s["fold"][ok] % T.FOLDS, s["edge"][ok], s["cover"][ok]
    up, veg = y_all == 0, np.isin(y_all, E.VEGETATED)
    use = veg | up
    p = held_out(nets, k_sq, pix, veg, f, use, emb, epochs, rng)
    near = use & (edge <= E.EDGE_M)
    out = {T.TESTS[0]: (E.auc(veg[use], p[use]), E.prec_at_recall(veg[use], p[use])),
           T.TESTS[1]: (E.auc(veg[near], p[near]), E.prec_at_recall(veg[near], p[near]))}
    m = (cov == 10) & ((y_all == 2) | up)
    y = y_all == 2
    if y[m].sum() >= 500:
        p2 = held_out(nets, k_sq, pix, y, f, m, emb, epochs, rng)
        out[T.TESTS[2]] = (E.auc(y[m], p2[m]), E.prec_at_recall(y[m], p2[m]))
    return out


def leave_one_out(nets, S, oks, embs, epochs):
    """As wetland_head.leave_one_out: for every square, a network taught on the other two. embs: name -> the
    embedding of the square's valid pixels, or None."""
    names, out = list(S), {}
    for name in names:
        rng = np.random.default_rng(0)
        sid, rows, ys, es = [], [], [], []
        for n in names:
            if n == name:
                continue
            r, y = H.rows_of(S[n], oks[n], rng)
            sid.append(np.full(len(r), names.index(n)))
            rows.append(np.flatnonzero(oks[n].ravel())[r])
            ys.append(y)
            if embs is not None:
                es.append(embs[n][r])
        net = fit(nets, np.concatenate(sid), np.concatenate(rows), np.concatenate(ys),
                  None if embs is None else np.concatenate(es), epochs)
        s, ok = S[name], oks[name]
        y_all, edge, cov = s["lab"][ok], s["edge"][ok], s["cover"][ok]
        up, veg = y_all == 0, np.isin(y_all, E.VEGETATED)
        use = veg | up
        u = np.flatnonzero(use)
        p = np.full(len(y_all), np.nan, np.float32)
        p[u] = predict(net, nets, names.index(name), np.flatnonzero(ok.ravel())[u], None if embs is None else embs[name][u])
        near = use & (edge <= E.EDGE_M)
        res = {H.TESTS[0]: (E.auc(veg[use], p[use]), E.prec_at_recall(veg[use], p[use])),
               H.TESTS[1]: (E.auc(veg[near], p[near]), E.prec_at_recall(veg[near], p[near]))}
        f = (cov == 10) & ((y_all == 2) | up)
        if (y_all[f] == 2).sum() >= 500:
            res[H.TESTS[2]] = (E.auc(y_all[f] == 2, p[f]), E.prec_at_recall(y_all[f] == 2, p[f]))
        out[name] = res
    return out


def main():
    global OFF
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--half-km", type=float, default=5.0)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(14)
    E.HALF_KM = a.half_km
    OFF = torch.arange(-P // 2, P // 2, device=DEV)
    print(f"device {DEV}", flush=True)
    S = E.load_squares()
    for name, s in S.items():
        t = time.time()
        prep(s)
        print(f"  elevation {name}: {s['p_ok'].mean():.1%} of pixels usable, {time.time() - t:.0f} s, "
              f"peak {peak():.1f} GB", flush=True)
    names = list(S)
    nets = world([S[n]["net"] for n in names])  # passed on as `nets`

    cols = ["summaries", "pictures"] + [c for m in EMB for c in (m, m + " + summaries", m + " + pictures")]
    IN = {t: {c: {} for c in cols} for t in T.TESTS}   # inside a square
    X = {c: None for c in cols}                        # taught on the other two

    def inside(col, name, v):
        for t, r in v.items():
            IN[t][col][name] = r

    t0 = time.time()
    oks = {n: S[n]["p_ok"] for n in names}
    for k, name in enumerate(names):
        s, ok = S[name], oks[name]
        inside("summaries", name, T.score(s["T"][ok], s, ok, np.random.default_rng(0)))
        inside("pictures", name, score(nets, k, s, ok, None, a.epochs, np.random.default_rng(0)))
        print(f"  {name}: summaries and pictures inside the square done, {(time.time() - t0) / 60:.0f} min, "
              f"peak {peak():.1f} GB", flush=True)
    X["summaries"] = H.leave_one_out({n: (S[n]["T"][oks[n]], oks[n]) for n in names}, S)
    X["pictures"] = leave_one_out(nets, S, oks, None, a.epochs)
    print(f"  summaries and pictures across squares done, {(time.time() - t0) / 60:.0f} min", flush=True)

    for model in EMB:
        hnd = E.open_model(model)
        Em, oks = {}, {}
        for name in names:
            s = S[name]
            Xe, ok = E.read_model(model, hnd, a.year, s["rc"])
            oks[name] = ok & s["p_ok"]
            Em[name] = Xe[oks[name]]
            del Xe
        for k, name in enumerate(names):
            s, ok = S[name], oks[name]
            inside(model, name, T.score(Em[name], s, ok, np.random.default_rng(0)))
            inside(model + " + summaries", name,
                   T.score(np.concatenate([Em[name], s["T"][ok]], 1), s, ok, np.random.default_rng(0)))
            inside(model + " + pictures", name, score(nets, k, s, ok, Em[name], a.epochs, np.random.default_rng(0)))
            print(f"  {model} {name}: inside the square done, {(time.time() - t0) / 60:.0f} min, "
                  f"peak {peak():.1f} GB", flush=True)
        X[model] = H.leave_one_out({n: (Em[n], oks[n]) for n in names}, S)
        X[model + " + summaries"] = H.leave_one_out(
            {n: (np.concatenate([Em[n], S[n]["T"][oks[n]]], 1), oks[n]) for n in names}, S)
        X[model + " + pictures"] = leave_one_out(nets, S, oks, Em, a.epochs)
        del Em
        print(f"  {model} across squares done, {(time.time() - t0) / 60:.0f} min, peak {peak():.1f} GB", flush=True)

    note = "AUC (and, at 80 % of wetland found, the share of flagged pixels that are wetland)."
    print(f"\nINSIDE A SQUARE. {note} Scored on 1 km blocks the head was not taught on.")
    for t in T.TESTS:
        print(f"\n{t}")
        for name in names:
            if name not in IN[t]["summaries"]:
                continue
            print(f"  {name}")
            for c in cols:
                au, pr = IN[t][c][name]
                print(f"    {c:24} {au:.3f} ({pr:.0%})")
    print(f"\nACROSS SQUARES. {note} Taught on the other two squares, scored on every pixel of this one.")
    for t in H.TESTS:
        print(f"\n{t}")
        for name in names:
            if t not in X["summaries"][name]:
                continue
            print(f"  {name}")
            for c in cols:
                au, pr = X[c][name][t]
                print(f"    {c:24} {au:.3f} ({pr:.0%})")
    print(f"\npeak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
