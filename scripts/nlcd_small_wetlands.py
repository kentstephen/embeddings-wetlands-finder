# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""Does Annual NLCD find the small wetlands of the states' finer maps as well as the saved heads do?

The same eleven held-out squares, the same wetlands as objects and the same dry background as fine_labels.py and
national_elev.py. Three maps are scored on them:

  NLCD        Annual NLCD 2024 land cover (USGS Collection 1, 30 m), woody wetland (90) and emergent herbaceous
              wetland (95), read at the nearest 30 m pixel to each 10 m pixel center. Once more with open water (11).
  alone       the saved national head, AlphaEarth alone
  with elev   the saved national head with the eleven terrain numbers from 10 m elevation

NLCD is a class, not a score, so it has one price: the share of dry background it calls wetland. A wetland is found
by NLCD when any of its pixels is wetland, or when half or more are. The heads are scored by the same two rules at
NLCD's own price in each state, and by the other scripts' rule (mean score over the cutoff) at their 5 % price.

Nothing is fitted and nothing is saved.

    uv run scripts/nlcd_small_wetlands.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import fine_labels as F  # noqa: E402
import small_wetlands as S  # noqa: E402

NLCD = ("https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/"
        "c1/v1/cu/mosaic/Annual_NLCD_LndCov_2024_CU_C1V1.tif")
WETLAND, WATER = (90, 95), 11
ELEV_HEAD = F.HEAD.with_name("wetland-water-head-alphaearth-elev10m.npz")
PRICE = 0.05


def nlcd_at(rc):
    """NLCD's class at the center of every AlphaEarth pixel of the window: nearest 30 m pixel, 250 outside the map."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"), rasterio.open(NLCD) as s:
        X, Y = Transformer.from_crs(4326, s.crs.to_wkt(), always_xy=True).transform(LON.ravel(), LAT.ravel())
        pad = 3 * abs(s.transform.a)
        w = from_bounds(X.min() - pad, Y.min() - pad, X.max() + pad, Y.max() + pad, s.transform).round_offsets().round_lengths()
        a = E.again(lambda: s.read(1, window=w, boundless=True, fill_value=250))
        tr = s.window_transform(w)
    r, c = np.floor((Y - tr.f) / tr.e).astype(int), np.floor((X - tr.c) / tr.a).astype(int)
    inside = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.full(LON.size, 250, np.uint8)
    out[inside] = a[r[inside], c[inside]]
    return out.reshape(LON.shape)


def share(g, wet):
    """Per wetland of a square, the share of its pixels that are set in `wet`."""
    from scipy import ndimage
    return np.asarray(ndimage.mean(wet.astype(np.float32), g["ids"], g["idx"])) if len(g["idx"]) else np.zeros(0)


def head_at(head, tests, elev, price):
    """Per wetland, found or not by a head whose cutoff flags `price` of the squares' dry background, by three rules:
    its mean score is over the cutoff (the rule of the other scripts), any of its pixels is, half or more are."""
    from scipy import ndimage
    ws, backs = [], []
    for g in tests:
        w = np.nan_to_num(F.wetness(head, g, elev), nan=-1.0)
        backs.append(w[g["back"]])
        ws.append(w.reshape(g["ids"].shape))
    cut = float(np.quantile(np.concatenate(backs), 1 - price))
    per = lambda fn: np.concatenate([np.asarray(fn(g, w)) if len(g["idx"]) else np.zeros(0) for g, w in zip(tests, ws)])  # noqa: E731
    over = per(lambda g, w: ndimage.mean((w >= cut).astype(np.float32), g["ids"], g["idx"]))
    return dict(mean=per(lambda g, w: ndimage.mean(np.maximum(w, 0), g["ids"], g["idx"])) >= cut, any=over > 0, half=over >= 0.5)


def table(title, acres, cols, keep):
    print(f"\n  {title}")
    print(f"    {'acres':>10} {'count':>6} " + " ".join(f"{n:>12}" for n in cols))
    rows = [(f"{lo:g} to {hi:g}" if hi < 1e8 else f"{lo:g}+", (acres >= lo) & (acres < hi)) for lo, hi in S.SIZES] + [("under 2", acres < 2)]
    for lab, k in rows:
        k = k & keep
        if k.any():
            print(f"    {lab:>10} {k.sum():>6} " + " ".join(f"{v[k].mean() * 100:>11.0f}%" for v in cols.values()))


def main():
    warnings.filterwarnings("ignore")
    F.ELEV = "10m"
    peak = E.guard(12)
    t0 = time.time()
    load = lambda p: (lambda h: (h["W1"], h["b1"], h["W2"], h["b2"]))(np.load(p))  # noqa: E731
    alone, with_elev = load(F.HEAD), load(ELEV_HEAD)
    summary = []
    for name, st in F.STATES.items():
        tests, wets, waters, kinds = [], [], [], []
        for sq, (lon, lat) in st["test"].items():
            g = F.gather(st, lon, lat, True)
            if isinstance(g, str):
                print(f"  {name}, {sq}: dropped, {g}", flush=True)
                continue
            lc = nlcd_at(E.rc_of(*E.box_of(lon, lat)))
            ok = g["ok"].reshape(lc.shape)
            tests.append(g)
            wets.append(np.isin(lc, WETLAND) & ok)
            waters.append((np.isin(lc, WETLAND) | (lc == WATER)) & ok)
            kinds.append(lc)
            print(f"  {name}, {sq}: {len(g['idx'])} wetlands, NLCD wetland {wets[-1].mean():.1%} of pixels, outside NLCD "
                  f"{(lc == 250).mean():.1%}; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
        if not tests:
            continue
        acres = np.concatenate([g["acres"] for g in tests])
        in_nwi = np.concatenate([g["in_nwi"] for g in tests])
        back = np.concatenate([g["back"] for g in tests])
        price = float(np.concatenate([w.ravel() for w in wets])[back].mean())
        price_w = float(np.concatenate([w.ravel() for w in waters])[back].mean())
        sh = np.concatenate([share(g, w) for g, w in zip(tests, wets)])
        sh_w = np.concatenate([share(g, w) for g, w in zip(tests, waters)])
        ha, he = head_at(alone, tests, False, price), head_at(with_elev, tests, True, price)
        five = {"alone": head_at(alone, tests, False, PRICE)["mean"], "with elev": head_at(with_elev, tests, True, PRICE)["mean"]}
        cols = {"NLCD any": sh > 0, "alone any": ha["any"], "elev any": he["any"],
                "NLCD half": sh >= 0.5, "alone half": ha["half"], "elev half": he["half"]}
        print(f"\n{name}: {len(acres)} wetlands of the state's map in {len(tests)} held-out squares, {in_nwi.mean():.0%} of them in NWI")
        print(f"  NLCD calls {price:.2%} of the dry background wetland ({price_w:.2%} with open water; found under 2 acres with "
              f"open water: any {(sh_w > 0)[acres < 2].mean():.0%}, half {(sh_w >= 0.5)[acres < 2].mean():.0%})")
        every = np.ones(len(acres), bool)
        table(f"found, every map at NLCD's price ({price:.2%} of dry ground flagged). any: a pixel flagged; half: half or more", acres, cols, every)
        table("the same, only wetlands NWI does not have", acres, cols, ~in_nwi)
        table(f"the heads with {PRICE:.0%} of dry ground flagged, mean score over the cutoff (the rule in the other scripts)", acres, five, every)
        # what NLCD calls the pixels of the small wetlands it has no wetland pixel in
        ids = np.concatenate([np.where(np.isin(g["ids"], g["idx"][(g["acres"] < 2) & (share(g, w) == 0)]), 1, 0).ravel()
                              for g, w in zip(tests, wets)]).astype(bool)
        v, n = np.unique(np.concatenate([k.ravel() for k in kinds])[ids], return_counts=True)
        top = sorted(zip(n, v), reverse=True)[:6]
        print("\n  NLCD's class on the pixels of the wetlands under 2 acres it misses: "
              + ", ".join(f"{c} {k / n.sum():.0%}" for k, c in top))
        k = acres < 2
        summary.append((name, int(k.sum()), price, *(cols[c][k].mean() for c in cols), *(five[c][k].mean() for c in five)))
        del tests, wets, waters, kinds

    print("\nWetlands under 2 acres found")
    heads = ["NLCD any", "alone any", "elev any", "NLCD half", "alone half", "elev half", "alone 5%", "elev 5%"]
    print(f"  {'state':>14} {'count':>6} {'price':>7} " + " ".join(f"{h:>10}" for h in heads))
    for name, n, price, *v in summary:
        print(f"  {name:>14} {n:>6} {price:>7.2%} " + " ".join(f"{x * 100:>9.0f}%" for x in v))
    print("\n  alone, elev: the saved national heads (AlphaEarth alone, with 10 m elevation) at NLCD's price in that state, by NLCD's\n"
          "  rule. 5%: the same heads with 5 % of dry ground flagged, mean score over the cutoff.")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
