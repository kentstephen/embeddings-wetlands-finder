# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""Do the heads find vernal pools?

A vernal pool is a small basin, usually under trees, that holds water in spring and is dry by late summer: often
one to four 10 m pixels. Massachusetts publishes two point layers (NHESP): certified pools, confirmed in the field
by the animals breeding in them, and potential pools, picked out on spring aerial photographs. This script scores
them in the Massachusetts squares of fine_labels.py.

Each pool is the 3 by 3 block of pixels around its point (a point can be 10 m off), and its score is the block's
mean. Dry background is scored the same way: 3 by 3 blocks around pixels outside NWI, the state's wetland map and
every pool, two pixels clear. A pool is found when its score is over the cutoff that flags 5 % (or 2 %) of the
background blocks.

  saved heads   the national heads, AlphaEarth alone and with 10 m elevation, as they are (wetland plus water)
  taught heads  pool or background, taught in the state's teaching squares and scored in its three held-out
                squares, on AlphaEarth, TESSERA, OlmoEarth, the terrain numbers, and AlphaEarth with them

Nothing is saved. One square at a time, about 4 GB.

    uv run scripts/vernal_pools.py [--limit 3]
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import fine_labels as F  # noqa: E402
import nwi_tiles as N  # noqa: E402
import small_wetlands as S  # noqa: E402
import terrain_adds as T  # noqa: E402
import water_head as W  # noqa: E402

MODELS = ("AlphaEarth", "TESSERA", "OlmoEarth")
YEAR = 2024
MA = "https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/AGOL"
LAYERS = {"certified": "NHESP_Certified_Vernal_Pools/MapServer/0", "potential": "NHESP_Potential_Vernal_Pools/MapServer/0"}
BACK = 6_000        # background blocks kept per square
PRICES = (0.02, 0.05)
ELEV_HEAD = F.HEAD.with_name("wetland-water-head-alphaearth-elev10m.npz")


def pools(layer, box):
    """Every pool point of a layer inside the box, as (lon, lat)."""
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode(dict(where="1=1", geometry=",".join(f"{v:.6f}" for v in box), geometryType="esriGeometryEnvelope", inSR=4326,
                                        outSR=4326, spatialRel="esriSpatialRelIntersects", outFields="OBJECTID", resultOffset=off, f="geojson"))
        d = E.again(lambda: json.load(urllib.request.urlopen(f"{MA}/{layer}/query?{q}", timeout=180, context=S.TLS)))  # noqa: B023
        if "error" in d:
            raise RuntimeError(str(d["error"])[:160])
        fs = d.get("features", [])
        for f in fs:
            g = f.get("geometry") or {}
            c = g.get("coordinates") or []
            out += [tuple(c)] if g.get("type") == "Point" else [tuple(p) for p in c]
        if len(fs) < 1000 and not d.get("exceededTransferLimit") and not d.get("properties", {}).get("exceededTransferLimit"):
            return out
        off += len(fs)


def gather(st, lon, lat):
    """One square: the nine pixels of every pool and of BACK background blocks, with every input."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy import ndimage
    t0 = time.time()
    box = E.box_of(lon, lat)
    rc = E.rc_of(*box)
    y0, y1, x0, x1 = rc
    h, w = y1 - y0, x1 - x0
    tr = Affine(E.RES, 0, E.X0 + x0 * E.RES, 0, -E.RES, E.Y0 - y0 * E.RES)
    z, Wx, Nn, crs, _ = F.dem10(box)
    Tn, ok = T.on_aef(T.terrain(z), Wx, Nn, crs, rc)
    del z
    ok &= np.isfinite(Tn).all(-1)
    X = {}
    for m in MODELS:
        a, k = E.read_model(m, E.open_model(m), YEAR, rc)
        ok &= k
        X[m] = a.astype(np.float16)
        del a
    X["terrain"] = Tn.astype(np.float32)
    nwi = E.burn(N.read(box), rc) > 0
    polys = F.fine_polys(st, box)
    fine = rasterize([(g, 1) for g in polys], out_shape=(h, w), transform=tr, fill=0, dtype="uint8") > 0 if polys else np.zeros((h, w), bool)
    pts, kind = [], []
    for name, layer in LAYERS.items():
        p = pools(layer, box)
        pts += p
        kind += [name] * len(p)
    pts = np.array(pts, float).reshape(-1, 2)
    c = np.floor((pts[:, 0] - E.X0) / E.RES).astype(int) - x0
    r = np.floor((E.Y0 - pts[:, 1]) / E.RES).astype(int) - y0
    inside = (c >= 1) & (c < w - 1) & (r >= 1) & (r < h - 1)
    c, r, kind = c[inside], r[inside], np.array(kind)[inside]
    # a certified pool is often a potential pool too: one pixel, one pool, certified first
    _, first = np.unique(r * w + c, return_index=True)
    first = np.sort(first)
    c, r, kind = c[first], r[first], kind[first]
    pool = np.zeros((h, w), bool)
    pool[r, c] = True
    clear = ~ndimage.binary_dilation(nwi | fine | ndimage.binary_dilation(pool, iterations=1), iterations=2)
    inner = np.zeros((h, w), bool)
    inner[1:-1, 1:-1] = True
    # a block counts when all nine of its pixels have every input
    whole = ndimage.minimum_filter(ok.astype(np.uint8), 3) > 0
    b = np.flatnonzero((clear & whole & inner).ravel())
    b = np.sort(np.random.default_rng(0).choice(b, min(BACK, len(b)), replace=False))
    keep = whole[r, c]
    r, c, kind = r[keep], c[keep], kind[keep]
    centers = np.concatenate([r * w + c, b])
    dr, dc = np.meshgrid((-1, 0, 1), (-1, 0, 1), indexing="ij")
    idx = (centers[:, None] + (dr.ravel() * w + dc.ravel())[None, :]).ravel()
    out = dict(n_pool=len(r), kind=kind, in_nwi=nwi[r, c], in_fine=fine[r, c],
               X={k: v.reshape(h * w, -1)[idx] for k, v in X.items()}, secs=time.time() - t0)
    return out


def block_mean(v, n):
    return v.reshape(n, 9).mean(1)


def found(pool, back):
    back = np.sort(back)
    return [float((pool >= np.quantile(back, 1 - p)).mean()) for p in PRICES] + [E.auc(np.r_[np.ones(len(pool)), np.zeros(len(back))] > 0, np.r_[pool, back])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    F.ELEV = "10m"
    peak = E.guard(10)
    t0 = time.time()
    st = F.STATES["Massachusetts"]
    teach_at = [p for p in st["teach"] if p not in F.LIDAR_DROPPED][:a.limit]
    test_at = dict(list(st["test"].items())[:a.limit])
    G, Tst = [], {}
    for i, (lon, lat) in enumerate(teach_at):
        try:
            g = gather(st, lon, lat)
        except Exception as e:  # noqa: BLE001
            print(f"  teaching {i + 1} of {len(teach_at)} ({lon}, {lat}): failed, {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        G.append(g)
        print(f"  teaching {i + 1} of {len(teach_at)} ({lon}, {lat}): {g['n_pool']} pools ({(g['kind'] == 'certified').sum()} certified), "
              f"{g['in_nwi'].mean():.0%} in NWI, {g['in_fine'].mean():.0%} in the state's wetland map; {g['secs']:.0f} s, memory {peak():.1f} GB", flush=True)
    for sq, (lon, lat) in test_at.items():
        try:
            g = gather(st, lon, lat)
        except Exception as e:  # noqa: BLE001
            print(f"  held out, {sq}: failed, {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        Tst[sq] = g
        print(f"  held out, {sq}: {g['n_pool']} pools ({(g['kind'] == 'certified').sum()} certified), {g['in_nwi'].mean():.0%} in NWI, "
              f"{g['in_fine'].mean():.0%} in the state's wetland map; {g['secs']:.0f} s, memory {peak():.1f} GB", flush=True)
    if not G or not Tst:
        print("nothing to teach on or to score")
        return
    INPUTS = {"AlphaEarth": ["AlphaEarth"], "TESSERA": ["TESSERA"], "OlmoEarth": ["OlmoEarth"], "terrain": ["terrain"],
              "AlphaEarth + terrain": ["AlphaEarth", "terrain"]}
    feats = lambda g, parts: np.hstack([g["X"][p].astype(np.float32) for p in parts])  # noqa: E731
    lab = lambda g: np.repeat(np.r_[np.ones(g["n_pool"]), np.zeros(len(g["X"]["terrain"]) // 9 - g["n_pool"])], 9)  # noqa: E731
    scores = {}  # input -> per held-out square, the block scores
    for inp, parts in INPUTS.items():
        m = T.head().fit(np.concatenate([feats(g, parts) for g in G]), np.concatenate([lab(g) for g in G]))
        scores[f"taught, {inp}"] = {sq: block_mean(m.predict_proba(feats(g, parts))[:, 1], len(g["X"]["terrain"]) // 9) for sq, g in Tst.items()}
    for tag, path, parts in (("saved, AlphaEarth alone", F.HEAD, ["AlphaEarth"]), ("saved, with elevation", ELEV_HEAD, ["AlphaEarth", "terrain"])):
        hd = np.load(path)
        scores[tag] = {sq: block_mean(1 - W.run_plain(hd["W1"], hd["b1"], hd["W2"], hd["b2"], feats(g, parts))[:, 0], len(g["X"]["terrain"]) // 9)
                       for sq, g in Tst.items()}
    n_all = sum(g["n_pool"] for g in Tst.values())
    kind = np.concatenate([g["kind"] for g in Tst.values()])
    in_nwi = np.concatenate([g["in_nwi"] for g in Tst.values()])
    in_fine = np.concatenate([g["in_fine"] for g in Tst.values()])
    print(f"\nVernal pools in {len(Tst)} held-out Massachusetts squares, {YEAR}: {n_all} pools ({(kind == 'certified').sum()} certified), "
          f"{in_nwi.mean():.0%} inside an NWI polygon, {in_fine.mean():.0%} inside the state's wetland map. Taught on {len(G)} squares "
          f"({sum(g['n_pool'] for g in G)} pools).")
    groups = {"all pools": np.ones(n_all, bool), "certified": kind == "certified", "potential only": kind != "certified",
              "in neither wetland map": ~in_nwi & ~in_fine, "in NWI": in_nwi}
    for title, keep in groups.items():
        if keep.sum() < 20:
            continue
        print(f"\n  {title} ({keep.sum()}): found with 2 % and 5 % of dry ground flagged, and AUC against dry ground")
        for tag, by in scores.items():
            pool = np.concatenate([by[sq][:g["n_pool"]] for sq, g in Tst.items()])[keep]
            back = np.concatenate([by[sq][g["n_pool"]:] for sq, g in Tst.items()])
            f2, f5, auc = found(pool, back)
            print(f"    {tag:32}{f2:>7.0%}{f5:>7.0%}{auc:>8.3f}")
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
