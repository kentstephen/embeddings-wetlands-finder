# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Does a wetland head carry to landscapes it has never seen when it is taught on many of them?

With three squares, a head taught on two and scored on the third lost a third of its precision. This script asks
whether that is a ceiling or only too few landscapes to learn from. It reads 10 km squares spread over the wetland
landscapes of the lower 48 states and, for each: NWI labels, the eleven 1 m lidar terrain numbers of
terrain_adds.py, AlphaEarth and OlmoEarth for several years, and the share of hydric soil in the USDA soil survey
map unit under each pixel. A random sample of each square's vegetated wetland and upland pixels is kept.

Squares are dealt into six folds and every square is scored by a small head (one hidden layer of 64) that saw no
pixel of it or of the other squares in its fold. Four questions:

  1. many squares    AUC and the share of flagged pixels that are wetland, per held-out square
  2. how many        the same head taught on 2, 5, 10 or all of the other squares, with the number of taught
                     pixels held the same
  3. several years   a head taught on one year against a head taught on four, each scored on all four years with
                     one fixed cutoff: does the share of wetland found stay put from year to year?
  4. soil survey     the head scored against hydric soil as a second answer key, and where NWI and the soil
                     survey agree and disagree

A candidate square is dropped when its 1 m elevation is incomplete, or NWI has too little or too much vegetated
wetland in it to score. Nothing is cached. Squares are read by --workers processes of up to 9 GB each.

    uv run scripts/many_squares.py [--workers 3] [--only "Lee, NH"] [--limit 4] [--screen]

--screen only prints NWI's wetland share for every candidate, without reading anything else.
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
import terrain_adds as T  # noqa: E402

MODELS = ("AlphaEarth", "OlmoEarth")
YEARS = (2018, 2020, 2022, 2024)
YEAR = 2024            # the year questions 1, 2 and 4 are asked in
KEEP = 50_000          # pixels kept per square
TEACH = 4_000          # pixels taught from each square
CURVE = 96_000         # pixels taught in all, whatever the number of squares, in question 2
FOLDS = 6
HYDRIC = 50            # percent of the map unit: at or above is hydric soil

# name: (center lon, center lat, landscape)
CANDIDATES = {
    "Delmarva, MD": (-75.80, 39.00, "coastal plain depressions"),
    "Stutsman County, ND": (-99.10, 47.10, "prairie potholes"),
    "Lee, NH": (-71.03, 43.12, "glaciated forest"),
    "Croatan, NC": (-77.05, 34.90, "pocosin"),
    "Dismal Swamp edge, VA": (-76.62, 36.72, "coastal plain swamp"),
    "Francis Marion, SC": (-79.70, 33.15, "Carolina bays"),
    "Okefenokee edge, GA": (-82.05, 31.05, "coastal plain swamp"),
    "Green Swamp, FL": (-81.95, 28.35, "cypress domes and flatwoods"),
    "Blackwater, FL": (-86.80, 30.75, "pine flatwoods"),
    "Yazoo, MS": (-90.90, 33.10, "alluvial bottomland"),
    "Cache River, AR": (-91.25, 35.10, "alluvial bottomland"),
    "Reelfoot, TN": (-89.30, 36.40, "alluvial bottomland"),
    "Atchafalaya edge, LA": (-91.85, 30.55, "river swamp"),
    "Brazoria, TX": (-95.45, 29.20, "coastal prairie"),
    "Rainwater Basin, NE": (-98.10, 40.60, "playa basins in cropland"),
    "Sandhills, NE": (-101.00, 42.30, "wet meadows in dunes"),
    "Prairie Coteau, SD": (-97.20, 45.20, "prairie potholes"),
    "Des Moines Lobe, IA": (-93.75, 42.80, "drained potholes in cropland"),
    "Itasca County, MN": (-93.50, 47.50, "northern peat and forest"),
    "Vilas County, WI": (-89.70, 45.90, "northern lakes and bogs"),
    "Luce County, MI": (-85.60, 46.45, "northern peat and forest"),
    "Kankakee, IN": (-86.90, 41.20, "sand plain marsh"),
    "Adirondacks, NY": (-74.50, 44.20, "mountain forest"),
    "Penobscot, ME": (-68.90, 45.20, "northern peat and forest"),
    "Plymouth County, MA": (-70.80, 41.90, "coastal pine and bogs"),
    "Pine Barrens, NJ": (-74.60, 39.75, "coastal pine and cedar swamp"),
    "Poconos, PA": (-75.40, 41.20, "glaciated plateau"),
    "Canaan Valley, WV": (-79.40, 39.07, "mountain valley"),
    "Flathead Valley, MT": (-114.10, 47.45, "glacial potholes"),
    "San Luis Valley, CO": (-105.90, 37.75, "arid basin wet meadows"),
    "Willamette Valley, OR": (-123.25, 44.35, "wet prairie"),
    "Merced, CA": (-120.45, 37.40, "vernal pools"),
    "Snohomish, WA": (-122.10, 47.90, "lowland forest and floodplain"),
    "Playa lakes, TX": (-101.80, 34.60, "playas in cropland"),
    "Roscommon, MI": (-84.60, 44.35, "northern peat and forest"),
    "Sampson County, NC": (-78.35, 34.95, "Carolina bays"),
    "Bladen County, NC": (-78.55, 34.65, "Carolina bays"),
    "Wicomico, MD": (-75.55, 38.35, "coastal plain depressions"),
    "Sussex County, DE": (-75.40, 38.65, "coastal plain depressions"),
    "Suwannee, FL": (-83.00, 30.20, "pine flatwoods"),
    "Osceola, FL": (-81.20, 28.10, "cypress domes and flatwoods"),
    "Tensas, LA": (-91.40, 32.20, "alluvial bottomland"),
    "Mobile-Tensaw, AL": (-87.90, 31.00, "river swamp"),
    "Big Thicket, TX": (-94.40, 30.40, "coastal plain swamp"),
    "Kidder County, ND": (-99.75, 47.00, "prairie potholes"),
    "Day County, SD": (-97.60, 45.40, "prairie potholes"),
    "Polk County, MN": (-96.00, 47.60, "prairie potholes"),
    "Cheyenne Bottoms, KS": (-98.65, 38.45, "playa basins in cropland"),
    "Horicon, WI": (-88.60, 43.50, "glacial marsh"),
    "Aitkin, MN": (-93.40, 46.60, "northern peat and forest"),
    "Tug Hill, NY": (-75.60, 43.70, "glaciated plateau"),
    "Montezuma, NY": (-76.75, 43.00, "glacial marsh"),
    "Northeast Kingdom, VT": (-71.90, 44.75, "mountain forest"),
    "Washington County, ME": (-67.60, 44.90, "northern peat and forest"),
    "Great Swamp, NJ": (-74.47, 40.71, "glacial lake basin"),
    "Klamath Basin, OR": (-121.85, 42.50, "arid basin marsh"),
    "Teton Valley, ID": (-111.15, 43.70, "mountain valley"),
    "Sacramento Valley, CA": (-121.90, 39.35, "managed marsh in cropland"),
    "Grays Harbor, WA": (-123.80, 46.95, "lowland forest and floodplain"),
    "Centennial Valley, MT": (-111.80, 44.65, "mountain valley"),
}
LOW, HIGH = 0.01, 0.85  # a square is kept when NWI's vegetated wetland is this share of its land


def nwi_share(name, lon, lat, kind):
    """Vegetated wetland as a share of a square's land by NWI: the quick check a candidate must pass."""
    try:
        box = E.box_of(lon, lat)
        lab = E.burn(E.read_nwi(box), E.rc_of(*box))
        up, veg = lab == 0, np.isin(lab, E.VEGETATED)
        return name, veg.sum() / max((veg | up).sum(), 1), (veg | up).mean()
    except Exception as e:  # noqa: BLE001
        return name, np.nan, f"{type(e).__name__}"


def gather(name, lon, lat, kind):
    """Everything kept from one square, or a string saying why it was dropped."""
    from scipy.ndimage import distance_transform_edt
    import flagged_forest as FF
    warnings.filterwarnings("ignore")
    E.guard(12)
    t0 = time.time()
    try:
        box = E.box_of(lon, lat)
        rc = E.rc_of(*box)
        lab = E.burn(E.read_nwi(box), rc)
        up, veg = lab == 0, np.isin(lab, E.VEGETATED)
        share = veg.sum() / max((veg | up).sum(), 1)
        if not LOW <= share <= HIGH or (veg | up).mean() < 0.3:
            return name, f"NWI vegetated wetland is {share:.1%} of the land, or the square is mostly water"
        z, W, N, crs, _ = T.read_dem(box)
        empty = float(np.isnan(z).mean())
        if empty > 0.05:
            return name, f"1 m elevation missing over {empty:.0%}"
        Tn, ok = T.on_aef(T.terrain(z), W, N, crs, rc)
        del z
        m_lat = E.RES * 111_320
        px = (m_lat * np.cos(np.radians(lat)), m_lat)
        wet = lab > 0
        edge = np.where(wet, distance_transform_edt(wet, sampling=px[::-1]), distance_transform_edt(~wet, sampling=px[::-1]))
        s = dict(rc=rc, box=box)
        cover = E.read_worldcover(rc)
        try:
            hyd = E.again(lambda: FF.hydric(s)[0])
        except Exception as e:  # noqa: BLE001
            hyd = np.full(lab.shape, np.nan, np.float32)
            print(f"  soil survey failed for {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        X = {}
        for model in MODELS:
            hnd = E.open_model(model)
            for y in YEARS:
                a, k = E.read_model(model, hnd, y, rc)
                ok &= k
                X[model, y] = a.astype(np.float16)
                del a
        rows = np.flatnonzero((ok & (veg | up)).ravel())
        rows = np.sort(np.random.default_rng(0).choice(rows, min(KEEP, len(rows)), replace=False))
        pick = lambda a: a.reshape(len(lab.ravel()), -1)[rows].squeeze()  # noqa: E731
        return name, dict(kind=kind, lon=lon, lat=lat, share=float(share), y=pick(veg), lab=pick(lab), edge=pick(edge).astype(np.float32),
                          cover=pick(cover), hyd=pick(hyd), T=pick(Tn), X={k: pick(v) for k, v in X.items()},
                          secs=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return name, f"failed: {type(e).__name__}: {str(e)[:160]}"


def feat(g, cols, year, rows=slice(None)):
    """Inputs for a set of columns ("AlphaEarth", "OlmoEarth", "terrain") in one year."""
    return np.concatenate([g["T"][rows] if c == "terrain" else g["X"][c, year][rows].astype(np.float32) for c in cols], 1)


def teach(G, names, cols, years, per, rng):
    """A head taught on `per` pixels of each named square, spread evenly over the years, and the cutoff that finds
    80 % of the wetland among the taught pixels."""
    Xs, ys = [], []
    for n in names:
        g = G[n]
        r = rng.choice(len(g["y"]), min(per, len(g["y"])), replace=False)
        for part, y in zip(np.array_split(r, len(years)), years):
            Xs.append(feat(g, cols, y, np.sort(part)))
            ys.append(g["y"][np.sort(part)])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    m = T.head().fit(X, y)
    return m, float(np.quantile(m.predict_proba(X[y])[:, 1], 0.2))


def med(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return f"{np.median(v):.3f} ({v.min():.3f} to {v.max():.3f})" if len(v) else "none"


def pct(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return f"{np.median(v):.0%} ({v.min():.0%} to {v.max():.0%})" if len(v) else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(14)
    C = {k: v for k, v in CANDIDATES.items() if not a.only or k in a.only}
    C = dict(list(C.items())[:a.limit]) if a.limit else C
    if a.screen:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            for name, share, land in ex.map(lambda kv: nwi_share(kv[0], *kv[1]), C.items()):
                keep = isinstance(land, float) and LOW <= share <= HIGH and land >= 0.3
                print(f"  {'keep' if keep else 'drop'}  {name:28} vegetated wetland {share:6.1%} of land, land {land if isinstance(land, str) else format(land, '.0%')}", flush=True)
        return
    G, t0 = {}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(gather, n, *v) for n, v in C.items()]):
            name, g = f.result()
            if isinstance(g, str):
                print(f"  dropped {name}: {g}", flush=True)
                continue
            G[name] = g
            print(f"  {name} ({g['kind']}): vegetated wetland {g['share']:.1%} of land, {len(g['y']):,} pixels kept, "
                  f"hydric soil known for {np.isfinite(g['hyd']).mean():.0%}, {g['secs']:.0f} s; "
                  f"{len(G)} squares after {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
    G = {n: G[n] for n in C if n in G}  # the listing order, which runs region by region
    names = list(G)
    K = min(FOLDS, len(names))
    fold = {n: i % K for i, n in enumerate(names)}
    print(f"\n{len(names)} squares kept of {len(C)}, {K} folds, memory {peak():.1f} GB", flush=True)
    if len(names) < 4:
        print("too few squares to hold any out")
        return

    INPUTS = [("terrain",), ("AlphaEarth",), ("AlphaEarth", "terrain"), ("OlmoEarth",), ("OlmoEarth", "terrain")]
    lbl = lambda cols: " + ".join(cols)  # noqa: E731
    R = {}      # (input, square) -> AUC, precision at 80 % found, AUC near the line, AUC in forest
    Pm = {}     # (input, square) -> held-out probability in YEAR
    Y = {}      # (input, taught years, square, year) -> share of wetland found, share of flagged that is wetland, AUC
    for cols in INPUTS:
        for k in range(K):
            tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
            ways = [((YEAR,), "one year")] + ([(YEARS, "four years")] if len(cols) == 2 else [])
            for years, way in ways:
                m, cut = teach(G, tr, cols, years, TEACH, np.random.default_rng(0))
                for n in te:
                    g = G[n]
                    for y in (YEARS if len(cols) == 2 else (YEAR,)):
                        p = m.predict_proba(feat(g, cols, y))[:, 1]
                        flag = p >= cut
                        Y[lbl(cols), way, n, y] = (flag[g["y"]].mean(), g["y"][flag].mean() if flag.any() else np.nan,
                                                   E.auc(g["y"], p))
                        if y == YEAR and way == "one year":
                            near = g["edge"] <= E.EDGE_M
                            fo = (g["cover"] == 10) & ((g["lab"] == 2) | (g["lab"] == 0))
                            R[lbl(cols), n] = (E.auc(g["y"], p), E.prec_at_recall(g["y"], p), E.auc(g["y"][near], p[near]),
                                               E.auc(g["lab"][fo] == 2, p[fo]) if (g["lab"][fo] == 2).sum() >= 300 else np.nan)
                            Pm[lbl(cols), n] = p
        print(f"  {lbl(cols)} done, {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)

    print(f"\n1. Many squares. Head taught on the other folds' squares ({YEAR}), scored on each held-out square.")
    print("   Median over squares (lowest to highest).")
    print(f"   {'':24}{'AUC':>26}{'flagged that is wetland':>28}{'AUC within 30 m':>26}{'AUC forest':>26}")
    for cols in INPUTS:
        v = np.array([R[lbl(cols), n] for n in names])
        print(f"   {lbl(cols):24}{med(v[:, 0]):>26}{pct(v[:, 1]):>28}{med(v[:, 2]):>26}{med(v[:, 3]):>26}")
    print("\n   Each square: AUC (share of flagged pixels that are wetland at 80 % found)")
    print(f"   {'':26}{'wetland':>8}{'terrain':>14}{'AlphaEarth+T':>14}{'OlmoEarth+T':>14}")
    for n in names:
        row = "".join(f"{R[c, n][0]:>8.3f} ({R[c, n][1]:>3.0%})" for c in ("terrain", "AlphaEarth + terrain", "OlmoEarth + terrain"))
        print(f"   {n:26}{G[n]['share']:>8.0%}{row}")

    print(f"\n2. How many squares to teach from. {CURVE:,} taught pixels in every case. Median over held-out squares.")
    sizes = [s for s in (2, 5, 10) if s < len(names) - len(names) // K] + ["all"]
    print(f"   {'':24}" + "".join(f"{str(s) + ' squares':>26}" for s in sizes))
    for cols in (("AlphaEarth", "terrain"), ("OlmoEarth", "terrain")):
        au, pr = {s: [] for s in sizes}, {s: [] for s in sizes}
        for k in range(K):
            tr, te = [n for n in names if fold[n] != k], [n for n in names if fold[n] == k]
            for s in sizes:
                rng = np.random.default_rng(k)
                sub = tr if s == "all" else list(rng.choice(tr, s, replace=False))
                m, _ = teach(G, sub, cols, (YEAR,), CURVE // len(sub), rng)
                for n in te:
                    p = m.predict_proba(feat(G[n], cols, YEAR))[:, 1]
                    au[s].append(E.auc(G[n]["y"], p))
                    pr[s].append(E.prec_at_recall(G[n]["y"], p))
        print(f"   {lbl(cols) + ', AUC':24}" + "".join(f"{med(au[s]):>26}" for s in sizes))
        print(f"   {lbl(cols) + ', flagged':24}" + "".join(f"{pct(pr[s]):>26}" for s in sizes), flush=True)

    print("\n3. Several years. One cutoff, set where the head finds 80 % of the wetland it was taught on. For each "
          "held-out square,\n   the share of wetland found in each of the four years, and how far that share moves "
          "between years (highest minus lowest).")
    print(f"   {'':24}{'taught on':>12}{'AUC, median':>14}{'found, median':>16}{'spread over years, median':>28}{'worst square':>14}")
    for c in ("AlphaEarth + terrain", "OlmoEarth + terrain"):
        for way in ("one year", "four years"):
            f = np.array([[Y[c, way, n, y][0] for y in YEARS] for n in names])
            au = np.array([[Y[c, way, n, y][2] for y in YEARS] for n in names])
            sp = f.max(1) - f.min(1)
            print(f"   {c:24}{way:>12}{np.median(au):>14.3f}{np.median(f):>16.0%}{np.median(sp):>28.0%}{sp.max():>14.0%}")
    print("   The same, by year scored (median share of wetland found):")
    for c in ("AlphaEarth + terrain", "OlmoEarth + terrain"):
        for way in ("one year", "four years"):
            print(f"   {c:24}{way:>12}" + "".join(f"{y}: {np.median([Y[c, way, n, y][0] for n in names]):>4.0%}   " for y in YEARS))

    print(f"\n4. Soil survey as a second answer key. Hydric soil: map unit at least {HYDRIC} % hydric. Held-out scores, {YEAR}.")
    print(f"   {'':24}{'AUC vs NWI':>26}{'AUC vs hydric soil':>26}{'AUC where both agree':>26}")
    cells = (("NWI wetland, hydric soil", True, True), ("NWI wetland, not hydric", True, False),
             ("NWI upland, hydric soil", False, True), ("NWI upland, not hydric", False, False))
    for c in ("terrain", "AlphaEarth + terrain", "OlmoEarth + terrain"):
        a1, a2, a3 = [], [], []
        cnt, flg = np.zeros(4), np.zeros(4)
        for n in names:
            g, p = G[n], Pm[c, n]
            k = np.isfinite(g["hyd"])
            if k.sum() < 1000:
                continue
            h, y = g["hyd"][k] >= HYDRIC, g["y"][k]
            a1.append(E.auc(y, p[k]))
            a2.append(E.auc(h, p[k]))
            a3.append(E.auc(y[h == y], p[k][h == y]))
            flag = p[k] >= np.quantile(p[g["y"]], 0.2)
            for i, (_, yy, hh) in enumerate(cells):
                sel = (y == yy) & (h == hh)
                cnt[i] += sel.sum()
                flg[i] += flag[sel].sum()
        print(f"   {c:24}{med(a1):>26}{med(a2):>26}{med(a3):>26}")
        if c != "terrain":
            print("     " + "; ".join(f"{nm}: {cnt[i] / cnt.sum():.0%} of pixels, {flg[i] / max(cnt[i], 1):.0%} flagged"
                                      for i, (nm, _, _) in enumerate(cells)))
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
