# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Is there wetland signal in yearly 10 m embeddings alone, before any lidar?

Three embeddings on the same pixels: AlphaEarth Foundations (64 numbers), TESSERA v1.1 (128), OlmoEarth v1.3 (128),
all read from Source Coop.

Three 10 km squares in different landscapes. For each: read the embedding of every 10 m pixel, read the National Wetlands Inventory polygons from the USFWS map service,
burn them onto the embedding grid, and teach a logistic regression (a linear probe) to tell wetland from upland.

The probe is always scored on ground it was not taught on: the square is cut into 1 km blocks and the blocks are
dealt into five folds. A second test teaches on two squares and scores on the third.

Nothing is cached: every run reads the embeddings and the polygons again. The only file written is the figure
(--fig, about 1 MB, AlphaEarth's map). One model is held at a time. Peak memory about 6 GB; the process stops itself
at 12 GB.

    uv run scripts/embedding_signal.py [--year 2024] [--fig figures/embedding_signal.png]
"""
import argparse
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RES, X0, Y0 = 8.983111749910169e-05, -180.0, 83.68570533713473  # the mosaic's grid, degrees
NWI = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services"
SEED, BLOCK_PX, FOLDS, TRAIN_MAX, EDGE_M = 0, 100, 5, 150_000, 30

# name: (center lon, center lat, what kind of ground)
SQUARES = {
    "Delmarva, MD": (-75.80, 39.00, "coastal plain: forested depressions among crop fields"),
    "Stutsman County, ND": (-99.10, 47.10, "prairie potholes: marsh basins in grass and crops"),
    "Lee, NH": (-71.03, 43.12, "glaciated forest: swamps under tree canopy"),
}
HALF_KM = 5.0

# NWI WETLAND_TYPE -> code burned onto the grid. 0 is upland (no polygon).
TYPES = {
    "Freshwater Emergent Wetland": 1,
    "Freshwater Forested/Shrub Wetland": 2,
    "Freshwater Pond": 3,
    "Lake": 4,
    "Riverine": 5,
    "Estuarine and Marine Wetland": 6,
    "Estuarine and Marine Deepwater": 7,
    "Other": 8,
}
NAME = {0: "upland", 1: "emergent", 2: "forested/shrub", 3: "pond", 4: "lake", 5: "riverine", 6: "tidal wetland",
        7: "tidal deepwater", 8: "other"}
VEGETATED = (1, 2, 6)  # the kinds that are not open water


def guard(max_gb):
    """Stop the process on its own when its memory passes max_gb; return a function giving the peak so far."""
    import psutil
    peak = [0.0]

    def run():
        p = psutil.Process()
        while True:
            gb = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
    return lambda: peak[0]


def box_of(lon, lat):
    dy = HALF_KM / 111.32
    dx = dy / np.cos(np.radians(lat))
    return lon - dx, lat - dy, lon + dx, lat + dy


def rc_of(lon0, lat0, lon1, lat1):
    """AlphaEarth mosaic rows and columns (y0, y1, x0, x1) whose pixel centers fall in the box."""
    x0, x1 = int(np.ceil((lon0 - X0) / RES - 0.5)), int(np.floor((lon1 - X0) / RES - 0.5)) + 1
    y0, y1 = int(np.ceil((Y0 - lat1) / RES - 0.5)), int(np.floor((Y0 - lat0) / RES - 0.5)) + 1
    return y0, y1, x0, x1


def again(fn):
    for attempt in range(4):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(3 + 5 * attempt)


def open_model(model):
    import zarr
    from obstore.store import HTTPStore, S3Store
    from zarr.storage import ObjectStore
    zarr.config.set({"async.concurrency": 64})
    if model == "AlphaEarth":
        st = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                     virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
        return zarr.open_group(ObjectStore(st, read_only=True), mode="r")
    if model == "TESSERA":
        return ObjectStore(HTTPStore.from_url("https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"), read_only=True)
    return ObjectStore(S3Store.from_url("s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
                                        endpoint="https://data.source.coop", skip_signature=True,
                                        client_options={"timeout": "120s", "connect_timeout": "10s"}), read_only=True)


def read_model(model, h, year, rc):
    """float32 (rows, cols, bands) on AlphaEarth's grid and a mask of pixels with data. TESSERA and OlmoEarth are
    stored per UTM zone on their own 10 m grids: each AlphaEarth pixel takes the pixel under its center."""
    import zarr
    y0, y1, x0, x1 = rc
    if model == "AlphaEarth":
        times = [int(t) for t in h["time"][:]]
        q = again(lambda: np.asarray(h["embeddings"][times.index(year), :, y0:y1, x0:x1]))
        f = q.astype(np.float32) / 127.5
        return np.moveaxis(np.sign(f) * f * f, 0, -1), q[0] != -128
    from pyproj import Transformer
    LON, LAT = np.meshgrid(X0 + (np.arange(x0, x1) + 0.5) * RES, Y0 - (np.arange(y0, y1) + 0.5) * RES)
    z = zarr.open_group(h, mode="r", path=f"utm{int((LON.mean() + 180) // 6) + 1:02d}")
    X, Y = Transformer.from_crs(4326, z.attrs["proj:code"], always_xy=True).transform(LON.ravel(), LAT.ravel())
    a, _, tx0, _, e, ty0 = z.attrs["spatial:transform"]
    col, row = np.floor((X - tx0) / a).astype(np.int64), np.floor((Y - ty0) / e).astype(np.int64)
    r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
    at = (row - r0) * (c1 - c0) + (col - c0)
    ti = [int(t) for t in z["time"][:]].index(year) if "time" in z else year - 2017
    q = again(lambda: np.asarray(z["embeddings"][ti, :, r0:r1, c0:c1])).reshape(128, -1)[:, at].T
    if model == "TESSERA":  # int8 with one scale per pixel
        sc = again(lambda: np.asarray(z["scales"][ti, r0:r1, c0:c1])).ravel()[at].astype(np.float32)
        ok = np.isfinite(sc) & (sc != 0)
        f = q.astype(np.float32) * np.where(ok, sc, 0)[:, None]
    else:
        ok = q[:, 0] != -128
        f = q.astype(np.float32)
    return f.reshape(y1 - y0, x1 - x0, 128), ok.reshape(y1 - y0, x1 - x0)


def get_json(url, params):
    u = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(u, timeout=120) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(3 + 5 * attempt)


def read_nwi(box):
    """NWI polygons that touch the box, as (shapely geometry, type code), larger polygons first."""
    from shapely.geometry import shape
    out, seen, off, page = [], set(), 0, 500
    while True:
        try:
            d = get_json(f"{NWI}/Wetlands/MapServer/0/query", {
                "geometry": ",".join(f"{v:.6f}" for v in box), "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                "outSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "OBJECTID,ATTRIBUTE,WETLAND_TYPE",
                "orderByFields": "OBJECTID", "resultOffset": off, "resultRecordCount": page, "f": "geojson"})
        except Exception:  # noqa: BLE001
            if page <= 20:
                raise
            page //= 5  # the service answers HTTP 500 when a page of large polygons is too big; ask for fewer
            continue
        fs = d.get("features", [])
        for f in fs:
            p = f["properties"]
            if p["OBJECTID"] in seen or not f.get("geometry"):
                continue
            seen.add(p["OBJECTID"])
            out.append((shape(f["geometry"]), TYPES.get(p["WETLAND_TYPE"], 8)))
        if len(fs) < page:
            break
        off += len(fs)
    return sorted(out, key=lambda t: -t[0].area)


def image_years(lon, lat):
    """Years of the photographs NWI was drawn from at a point, from the service's Image Year layer."""
    try:
        d = get_json(f"{NWI}/Data_Source/MapServer/3/query", {
            "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326, "outFields": "*",
            "returnGeometry": "false", "f": "json"})
        ys = set()
        for f in d.get("features", []):
            for k, v in f["attributes"].items():
                if k.upper() == "IMAGE_YR" and v not in (None, "", 0):
                    ys.add(str(v))
        return ", ".join(sorted(ys)) or "not given"
    except Exception as e:  # noqa: BLE001
        return f"lookup failed ({type(e).__name__})"


def read_worldcover(rc):
    """ESA WorldCover 2021 (10 m, CC BY 4.0) at the center of every embedding pixel; 10 is tree cover, 0 no data."""
    import rasterio
    from rasterio.windows import Window
    y0, y1, x0, x1 = rc
    lon, lat = X0 + (np.arange(x0, x1) + 0.5) * RES, Y0 - (np.arange(y0, y1) + 0.5) * RES
    out = np.zeros((len(lat), len(lon)), np.uint8)
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for ty in sorted({int(np.floor(v / 3) * 3) for v in (lat[0], lat[-1])}):
            for tx in sorted({int(np.floor(v / 3) * 3) for v in (lon[0], lon[-1])}):
                nm = f"{'N' if ty >= 0 else 'S'}{abs(ty):02d}{'E' if tx >= 0 else 'W'}{abs(tx):03d}"
                url = f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{nm}_Map.tif"
                with rasterio.open(url) as r:
                    c = np.floor((lon - r.transform.c) / r.transform.a).astype(int)
                    rw = np.floor((lat - r.transform.f) / r.transform.e).astype(int)
                    mc, mr = (c >= 0) & (c < r.width), (rw >= 0) & (rw < r.height)
                    if not (mc.any() and mr.any()):
                        continue
                    c0, r0 = c[mc].min(), rw[mr].min()
                    a = r.read(1, window=Window(c0, r0, c[mc].max() - c0 + 1, rw[mr].max() - r0 + 1))
                    out[np.ix_(mr, mc)] = a[np.ix_(rw[mr] - r0, c[mc] - c0)]
    return out


def burn(polys, rc):
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    y0, y1, x0, x1 = rc
    tr = Affine(RES, 0, X0 + x0 * RES, 0, -RES, Y0 - y0 * RES)
    if not polys:
        return np.zeros((y1 - y0, x1 - x0), np.uint8)
    return rasterize(polys, out_shape=(y1 - y0, x1 - x0), transform=tr, fill=0, dtype="uint8")  # pixel center inside


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=400))


def fit(X, y, rows, rng):
    """A probe taught on at most TRAIN_MAX of the given rows."""
    k = rng.choice(rows, TRAIN_MAX, replace=False) if len(rows) > TRAIN_MAX else rows
    return probe().fit(X[k], y[k])


def held_out(X, y, fold, inside, rng):
    """Probability for every row in `inside` from a probe taught on the other folds' rows in `inside`."""
    p = np.full(len(y), np.nan, np.float32)
    for k in range(FOLDS):
        tr, te = np.flatnonzero(inside & (fold != k)), np.flatnonzero(inside & (fold == k))
        if len(te) and 0 < y[tr].sum() < len(tr):
            p[te] = fit(X, y, tr, rng).predict_proba(X[te])[:, 1]
    return p


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def prec_at_recall(y, s, recall=0.8):
    """Share of flagged pixels that are wetland when the cutoff finds `recall` of the wetland pixels."""
    if not 0 < y.sum() < len(y):
        return np.nan
    cut = np.quantile(s[y], 1 - recall)
    return y[s >= cut].mean()


def load_squares():
    """Labels and reference layers for every square, on AlphaEarth's grid."""
    S = {}
    for name, (lon, lat, kind) in SQUARES.items():
        box = box_of(lon, lat)
        rc = rc_of(*box)
        polys = read_nwi(box)
        lab = burn(polys, rc)
        h, w = lab.shape
        m_lat = RES * 111_320
        px_m = (m_lat * np.cos(np.radians(lat)), m_lat)  # pixel width, height in meters
        wet = lab > 0
        edge = np.where(wet, distance_transform_edt(wet, sampling=px_m[::-1]),
                        distance_transform_edt(~wet, sampling=px_m[::-1]))  # meters to the NWI line
        blocks = (np.arange(h)[:, None] // BLOCK_PX) * 1000 + np.arange(w)[None, :] // BLOCK_PX
        ids = np.unique(blocks)
        fold_of = dict(zip(ids.tolist(), (np.random.default_rng(SEED).permutation(len(ids)) % FOLDS).tolist()))
        S[name] = dict(lab=lab, edge=edge, fold=np.vectorize(fold_of.get)(blocks), box=box, rc=rc, px_m=px_m,
                       cover=read_worldcover(rc))
        print(f"{name} ({kind}): {h} x {w} pixels, {len(polys)} NWI polygons, photo years {image_years(lon, lat)}; "
              + ", ".join(f"{NAME[c]} {np.mean(lab == c):.1%}" for c in sorted(NAME) if (lab == c).any()), flush=True)
    return S


MODELS = ("AlphaEarth", "TESSERA", "OlmoEarth")
TESTS = ("vegetated wetland vs upland",
         f"the same, within {EDGE_M} m of the NWI line",
         "forested/shrub wetland vs upland, tree cover only",
         "emergent wetland vs upland, low cover only",
         "taught on the other two squares")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--fig", default="figures/embedding_signal.png")
    a = ap.parse_args()
    peak = guard(12)

    S = load_squares()

    R = {t: {m: {} for m in MODELS} for t in TESTS}   # AUC
    Pr = {t: {m: {} for m in MODELS} for t in TESTS}  # share of flagged pixels that are wetland at 80 % found
    for model in MODELS:
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        hnd = open_model(model)
        D = {}
        for name, s in S.items():
            E, ok = read_model(model, hnd, a.year, s["rc"])
            D[name] = (E[ok], ok)
            print(f"  {model} {name}: {E.shape[-1]} numbers a pixel, {ok.mean():.1%} with data, {time.time() - t0:.0f} s", flush=True)
            del E
        for name, s in S.items():
            X, ok = D[name]
            y_all, f, edge, cov = s["lab"][ok], s["fold"][ok], s["edge"][ok], s["cover"][ok]
            up, veg = y_all == 0, np.isin(y_all, VEGETATED)
            use = veg | up  # open water is left out of both sides
            p = held_out(X, veg, f, use, rng)
            if model == "AlphaEarth":
                s["p_veg"] = np.full(s["lab"].shape, np.nan, np.float32)
                s["p_veg"][ok] = p
            R[TESTS[0]][model][name], Pr[TESTS[0]][model][name] = auc(veg[use], p[use]), prec_at_recall(veg[use], p[use])
            near = use & (edge <= EDGE_M)
            R[TESTS[1]][model][name], Pr[TESTS[1]][model][name] = auc(veg[near], p[near]), prec_at_recall(veg[near], p[near])
            low = np.isin(cov, (20, 30, 40, 60, 90))  # shrub, grass, crop, bare, herbaceous wetland
            for t, code, inside in ((TESTS[2], 2, cov == 10), (TESTS[3], 1, low)):
                m = inside & ((y_all == code) | up)
                y = y_all == code
                if y[m].sum() < 500:
                    continue
                p2 = held_out(X, y, f, m, rng)
                R[t][model][name], Pr[t][model][name] = auc(y[m], p2[m]), prec_at_recall(y[m], p2[m])
        for name, s in S.items():
            Xs, ys = [], []
            for other in S:
                if other == name:
                    continue
                Xo, oko = D[other]
                y_o = S[other]["lab"][oko]
                rows = np.flatnonzero(np.isin(y_o, VEGETATED) | (y_o == 0))
                k = rng.choice(rows, min(TRAIN_MAX // 2, len(rows)), replace=False)
                Xs.append(Xo[k])
                ys.append(np.isin(y_o[k], VEGETATED))
            m = probe().fit(np.concatenate(Xs), np.concatenate(ys))
            X, ok = D[name]
            y_t = s["lab"][ok]
            u = np.flatnonzero(np.isin(y_t, VEGETATED) | (y_t == 0))
            pt, v = m.predict_proba(X[u])[:, 1], np.isin(y_t[u], VEGETATED)
            R[TESTS[4]][model][name], Pr[TESTS[4]][model][name] = auc(v, pt), prec_at_recall(v, pt)
        del D
        print(f"  {model} done, {time.time() - t0:.0f} s, peak memory {peak():.1f} GB", flush=True)

    print("\nAUC (and, at 80 % of wetland found, the share of flagged pixels that are wetland). Linear probe, scored on "
          "1 km blocks it was not taught on.")
    for t in TESTS:
        print(f"\n{t}")
        print(f"  {'':22}" + "".join(f"{m:>16}" for m in MODELS))
        for name in S:
            if not any(name in R[t][m] for m in MODELS):
                continue
            print(f"  {name:22}" + "".join(
                f"{R[t][m][name]:>9.3f} ({Pr[t][m][name]:>3.0%})" if name in R[t][m] else f"{'':>16}" for m in MODELS))

    figure(S, a.year, Path(a.fig))
    print(f"\npeak memory {peak():.1f} GB")


def figure(S, year, path):
    """One row per square: NWI as burned, and the held-out probability of vegetated wetland."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    # upland, emergent, forested/shrub, open water and other: checked for protan separation
    cols = ["#f2f1ee", "#E69F00", "#0072B2", "#CC79A7"]
    group = np.array([0, 1, 2, 3, 3, 3, 1, 3, 3])  # type code -> color slot (tidal wetland drawn as emergent)
    fig, ax = plt.subplots(len(S), 2, figsize=(11, 4.6 * len(S)), constrained_layout=True)
    for i, (name, s) in enumerate(S.items()):
        ext = (s["box"][0], s["box"][2], s["box"][1], s["box"][3])
        asp = s["px_m"][1] / s["px_m"][0]
        ax[i, 0].imshow(group[s["lab"]], cmap=ListedColormap(cols), vmin=0, vmax=3, extent=ext, aspect=asp,
                        interpolation="nearest")
        ax[i, 0].set_title(f"{name}: National Wetlands Inventory", fontsize=10, loc="left")
        im = ax[i, 1].imshow(s["p_veg"], cmap="cividis", vmin=0, vmax=1, extent=ext, aspect=asp, interpolation="nearest")
        ax[i, 1].set_title(f"Linear probe on {year} embeddings, held-out blocks", fontsize=10, loc="left")
        for x in ax[i]:
            x.tick_params(labelsize=7, length=2)
            for sp in x.spines.values():
                sp.set_visible(False)
    ax[0, 0].legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, [
        "upland (no polygon)", "emergent, tidal marsh", "forested/shrub", "open water, other"])],
        loc="upper right", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none")
    fig.colorbar(im, ax=ax[:, 1], shrink=0.4, label="probability of vegetated wetland")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"\nfigure: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
