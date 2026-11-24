# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pmtiles", "mapbox-vector-tile", "truststore"]
# ///
"""Do finer labels, and lidar elevation, let a head find small wetlands?

small_wetlands.py showed the head taught on NWI finds few wetlands under 2 acres in a state's finer map. Here heads
are taught inside each state that publishes a finer map and scored on squares of that state held out whole:

  national    the saved head (76 squares of the lower 48, NWI labels, AlphaEarth alone)
  NWI         taught in this state's squares on NWI labels, AlphaEarth alone      (what being local buys)
  fine        the same squares, the state's map joined with NWI as the label       (what better labels buy)
  NWI + elev  NWI labels, AlphaEarth and eleven numbers from 1 m lidar elevation
  fine + elev the state's labels, AlphaEarth and the elevation numbers             (what elevation buys on top)

Then each state's fine heads are scored in the other states, to see whether what is learned carries.

Every wetland of the state's map in a held-out square is one object. A head's cutoff is set where a fixed share of
the square set's dry background (outside both maps, two pixels clear) is flagged, so heads are compared at the same
price; a wetland is found when its mean score (one minus the upland probability) is at or over that cutoff. Rank is
the share of background that scores below the wetland.

    uv run scripts/fine_labels.py            # every state
    uv run scripts/fine_labels.py --pilot    # two teaching squares and one held out, Vermont only
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
import nwi_tiles as N  # noqa: E402
import small_wetlands as S  # noqa: E402
import terrain_adds as T  # noqa: E402

YEARS = (2022, 2024)   # taught on both
YEAR = 2024            # scored on
KEEP = 40_000          # pixels kept from a teaching square
TEACH = 8_000          # pixels taught from each square
PRICES = (0.02, 0.05)  # share of dry background flagged
WATER = (3, 4, 5, 7)   # NWI pond, lake, riverine, tidal deepwater
HEAD = S.HEAD

# A state's finer map: ArcGIS layers queried by box. `where` leaves out what is not wetland in a layer.
STATES = {
    "Vermont": dict(
        url="https://anrmaps.vermont.gov/arcgis/rest/services/Open_Data/OPENDATA_ANR_WATER_SP_NOCACHE_v2/MapServer",
        layers={193: None, 194: None},
        test={"Montpelier": (-72.635, 44.245), "Middlebury": (-73.185, 43.995), "Franklin County": (-72.885, 44.795)},
        teach=[(-73.10, 44.45), (-72.95, 44.25), (-72.75, 44.50), (-72.50, 44.55), (-72.30, 44.35), (-72.10, 44.55),
               (-72.45, 44.05), (-72.80, 43.95), (-72.60, 43.75), (-73.05, 43.65), (-72.75, 43.45), (-72.50, 43.30),
               (-73.10, 43.25), (-72.85, 43.05), (-72.65, 42.90), (-73.15, 42.95), (-73.15, 44.30), (-73.05, 44.90),
               (-72.55, 44.85), (-72.20, 44.85)],
    ),
    # MassDEP Wetlands: 1:12,000 photo interpretation updated from 0.5 m 2005 imagery. Open water and the coastal
    # classes (beach, dune, barrier, rocky shore, tidal flat) are left out.
    "Massachusetts": dict(
        url="https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/AGOL/DEP_Wetlands_FieldMaps/MapServer",
        layers={1: "IT_VALC NOT IN ('BE','D','BA','RS','TF','OW') AND IT_VALC NOT LIKE 'BB-%'"},
        test={"Northampton": (-72.60, 42.25), "Worcester west": (-71.89, 42.30), "Foxborough": (-71.30, 42.10)},
        teach=[(-72.90, 42.40), (-72.70, 42.55), (-72.45, 42.45), (-72.20, 42.55), (-72.00, 42.45), (-71.80, 42.55),
               (-71.55, 42.50), (-71.35, 42.35), (-72.30, 42.20), (-72.05, 42.15), (-71.80, 42.15), (-71.55, 42.15),
               (-73.20, 42.40), (-73.15, 42.20)],
    ),
    # New Jersey wetlands from the 2020 land use / land cover: 2020 imagery, quarter-acre changes.
    "New Jersey": dict(
        url="https://services1.arcgis.com/QWdNfRs7lkPq4g4Q/arcgis/rest/services/Wetlands_2020/FeatureServer",
        layers={14: None},
        test={"Princeton": (-74.69, 40.295), "Lakehurst": (-74.40, 39.95), "Salem and Gloucester": (-75.10, 39.60)},
        teach=[(-74.70, 41.10), (-74.90, 40.95), (-74.55, 40.95), (-74.85, 40.70), (-74.60, 40.60), (-74.80, 40.45),
               (-74.45, 40.45), (-74.35, 40.20), (-74.75, 40.05), (-74.75, 39.85), (-74.90, 39.75), (-74.65, 39.55),
               (-74.85, 39.40), (-75.15, 39.38)],
    ),
    # Delaware State Wetlands Mapping Project 2017: leaf-off imagery at 1:5,000, quarter-acre and smaller. Rivers,
    # lakes and tidal deepwater are left out.
    "Delaware": dict(
        url="https://enterprise.firstmap.delaware.gov/arcgis/rest/services/Hydrology/DE_Wetlands/MapServer",
        layers={6: "WETLAND_TY NOT IN ('Riverine','Lake','Estuarine and Marine Deepwater')"}, id="OBJECTID_1",
        test={"Smyrna": (-75.64, 39.26), "Milford west": (-75.52, 38.90)},
        teach=[(-75.68, 39.50), (-75.66, 39.38), (-75.60, 39.12), (-75.58, 39.01), (-75.62, 38.78), (-75.42, 38.78),
               (-75.62, 38.66), (-75.40, 38.66), (-75.50, 38.54), (-75.25, 38.56)],
    ),
}


ELEV = "lidar"
# Teaching squares the lidar run dropped for missing 1 m elevation: left out of the 10 m run too, so both runs teach on
# the same squares.
LIDAR_DROPPED = {(-71.8, 42.55), (-71.8, 42.15)}
TEN = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"


def dem10(box):
    """3DEP's seamless 1/3 arc-second elevation (about 10 m, one file per degree square on S3) over the box, put on
    the same 2 m UTM grid the lidar reader makes (bilinear), so terrain_adds.terrain works out the same eleven
    numbers from it. Returns what terrain_adds.read_dem returns."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    crs = f"EPSG:{32600 + int((0.5 * (box[0] + box[2]) + 180) // 6) + 1}"
    xs, ys = Transformer.from_crs(4326, crs, always_xy=True).transform([box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
    S2 = T.STEP
    W, Nn = np.floor(min(xs) / S2) * S2 - 400, np.ceil(max(ys) / S2) * S2 + 400  # 400 m margin, as the lidar reader
    w, h = int((max(xs) + 400 - W) / S2) + 1, int((Nn - (min(ys) - 400)) / S2) + 1
    z = np.full((h, w), np.nan, np.float32)
    tr = Affine(S2, 0, W, 0, -S2, Nn)
    used = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"):
        # a file is named for its north and west edges; 0.01 degrees more on each side covers the margin
        for n in range(int(np.ceil(box[1] - 0.01)), int(np.ceil(box[3] + 0.01)) + 1):
            for wd in range(int(np.ceil(-box[2] - 0.01)), int(np.ceil(-box[0] + 0.01)) + 1):
                name = f"n{n:02d}w{wd:03d}"
                part = np.full((h, w), np.nan, np.float32)
                try:
                    with rasterio.open(f"/vsicurl/{TEN}/{name}/USGS_13_{name}.tif") as r:
                        E.again(lambda: reproject(rasterio.band(r, 1), part, dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear,  # noqa: B023
                                                  src_nodata=r.nodata, dst_nodata=np.nan))
                except rasterio.errors.RasterioIOError:
                    continue
                part[part < -1000] = np.nan
                m = np.isnan(z) & ~np.isnan(part)
                if m.any():
                    z[m] = part[m]
                    used.append(name)
    return z, W, Nn, crs, used


def fine_polys(st, box):
    """Every polygon of the state's map that touches the box, in lon, lat."""
    from shapely.geometry import shape
    from shapely.validation import make_valid
    out = []
    for layer, where in st["layers"].items():
        off = 0
        while True:
            q = dict(geometry=",".join(f"{v:.6f}" for v in box), geometryType="esriGeometryEnvelope", inSR=4326, outSR=4326,
                     spatialRel="esriSpatialRelIntersects", outFields=st.get("id", "OBJECTID"), resultOffset=off, f="geojson")
            if where:
                q["where"] = where
            url = f"{st['url']}/{layer}/query?{urllib.parse.urlencode(q)}"
            d = E.again(lambda: json.load(urllib.request.urlopen(url, timeout=180, context=S.TLS)))  # noqa: B023
            if "error" in d:
                raise RuntimeError(f"{layer}: {str(d['error'])[:160]}")
            fs = d.get("features", [])
            for f in fs:
                if f.get("geometry"):
                    g = make_valid(shape(f["geometry"]))
                    if not g.is_empty:
                        out.append(g)
            if not fs or not (d.get("exceededTransferLimit") or d.get("properties", {}).get("exceededTransferLimit") or len(fs) >= st.get("page", 1000)):
                break
            off += len(fs)
    return out


def gather(st, lon, lat, whole):
    """A square: embeddings, elevation numbers, NWI and the state's map. A teaching square keeps KEEP pixels; a
    held-out square (whole) keeps its grid and its wetlands as objects. A string says why a square was dropped."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy import ndimage
    warnings.filterwarnings("ignore")
    box = E.box_of(lon, lat)
    rc = E.rc_of(*box)
    y0, y1, x0, x1 = rc
    shp = (y1 - y0, x1 - x0)
    tr = Affine(E.RES, 0, E.X0 + x0 * E.RES, 0, -E.RES, E.Y0 - y0 * E.RES)
    try:
        z, W, Nn, crs, _ = dem10(box) if ELEV == "10m" else T.read_dem(box)
    except Exception as e:  # noqa: BLE001
        return f"1 m elevation failed: {type(e).__name__}: {str(e)[:100]}"
    empty = float(np.isnan(z).mean())
    if empty > 0.05:
        return f"1 m elevation missing over {empty:.0%}"
    Tn, ok = T.on_aef(T.terrain(z), W, Nn, crs, rc)
    del z
    ok &= np.isfinite(Tn).all(-1)
    polys = fine_polys(st, box)
    obs = sorted(S.objects(polys, box), key=lambda t: -t[1]) if polys else []
    fine = rasterize([(g, 1) for g, _ in obs], out_shape=shp, transform=tr, fill=0, dtype="uint8") > 0 if obs else np.zeros(shp, bool)
    lab = E.burn(N.read(box), rc)
    water = np.isin(lab, WATER)
    y_nwi = np.where(water, 2, np.where(np.isin(lab, E.VEGETATED), 1, 0)).astype(np.uint8)
    y_nwi[(lab > 0) & (y_nwi == 0)] = 255  # NWI "other": left out
    y_fine = np.where(water, 2, np.where(fine | (y_nwi == 1), 1, 0)).astype(np.uint8)
    hnd = E.open_model("AlphaEarth")
    X = {}
    for y in YEARS:
        a, k = E.read_model("AlphaEarth", hnd, y, rc)
        ok &= k
        X[y] = a.astype(np.float16)
        del a
    if not whole:
        rows = np.flatnonzero(ok.ravel())
        rows = np.sort(np.random.default_rng(0).choice(rows, min(KEEP, len(rows)), replace=False))
        pick = lambda a: a.reshape(lab.size, -1)[rows].squeeze()  # noqa: E731
        return dict(X={y: pick(v) for y, v in X.items()}, T=pick(Tn).astype(np.float32), y_nwi=pick(y_nwi), y_fine=pick(y_fine),
                    fine=float(fine.mean()), nwi=float((lab > 0).mean()))
    # every pixel a wetland touches is its own; larger first, so a small one keeps its pixels where shapes meet
    ids = rasterize([(g, i + 1) for i, (g, _) in enumerate(obs)], out_shape=shp, transform=tr, fill=0, dtype="int32", all_touched=True) \
        if obs else np.zeros(shp, np.int32)
    idx = np.arange(1, len(obs) + 1)
    n_px = ndimage.sum(np.ones(shp), ids, idx) if obs else np.zeros(0)
    in_nwi = ndimage.mean((lab > 0).astype(np.float32), ids, idx) >= 0.5 if obs else np.zeros(0, bool)
    back = ~ndimage.binary_dilation((ids > 0) | (lab > 0), iterations=2) & ok
    return dict(X=X[YEAR].reshape(-1, X[YEAR].shape[-1]), T=Tn.reshape(-1, Tn.shape[-1]).astype(np.float32), ok=ok.ravel(), ids=ids,
                idx=idx[n_px > 0], acres=np.array([a for _, a in obs])[n_px > 0], in_nwi=np.asarray(in_nwi)[n_px > 0], back=back.ravel(),
                fine=float(fine.mean()), nwi=float((lab > 0).mean()))


def feats(g, year, elev):
    x = (g["X"][year] if isinstance(g["X"], dict) else g["X"]).astype(np.float32)
    return np.hstack([x, g["T"]]) if elev else x


def teach(G, label, elev, rng):
    Xs, ys = [], []
    for g in G:
        r = np.flatnonzero(g[label] != 255)
        r = rng.choice(r, min(TEACH, len(r)), replace=False)
        for part, y in zip(np.array_split(r, len(YEARS)), YEARS):
            part = np.sort(part)
            Xs.append(feats(g, y, elev)[part])
            ys.append(g[label][part])
    return T.head().fit(np.concatenate(Xs), np.concatenate(ys))


def wetness(head, g, elev):
    """One minus the upland probability at every pixel of a held-out square (NaN where there is no input)."""
    x = feats(g, YEAR, elev)
    if isinstance(head, tuple):
        W1, b1, W2, b2 = head
        z = np.maximum(x @ W1 + b1, 0) @ W2 + b2
        p = np.exp(z - z.max(1, keepdims=True))
        up = p[:, 0] / p.sum(1)
    else:
        x = np.where(np.isfinite(x), x, 0)
        up = head.predict_proba(x)[:, list(head.classes_).index(0)]
    return np.where(g["ok"], 1 - up, np.nan).astype(np.float32)


def score(head, tests, elev):
    """Per wetland over the held-out squares: acres, in NWI, found at each price, rank. Cutoffs come from all the
    squares' background together."""
    from scipy import ndimage
    per, backs = [], []
    for g in tests:
        w = wetness(head, g, elev)
        b = w[g["back"]]
        backs.append(b[np.isfinite(b)])
        m = ndimage.mean(np.nan_to_num(w).reshape(g["ids"].shape), g["ids"], g["idx"]) if len(g["idx"]) else np.zeros(0)
        per.append((g["acres"], g["in_nwi"], np.asarray(m)))
    back = np.sort(np.concatenate(backs))
    cuts = [float(np.quantile(back, 1 - p)) for p in PRICES]
    acres, in_nwi, mean = (np.concatenate(c) for c in zip(*per))
    return dict(acres=acres, in_nwi=in_nwi, found=[mean >= c for c in cuts], rank=np.searchsorted(back, mean) / len(back), cuts=cuts)


def show(title, R, keep=None):
    """Heads side by side, by wetland size: found at each price, and rank."""
    names = list(R)
    a = R[names[0]]["acres"]
    keep = np.ones(len(a), bool) if keep is None else keep
    for j, price in enumerate(PRICES):
        print(f"\n  {title}: wetlands found with {price:.0%} of dry ground flagged")
        print(f"    {'acres':>10} {'count':>6} " + " ".join(f"{n:>11}" for n in names))
        for lo, hi in S.SIZES:
            k = keep & (a >= lo) & (a < hi)
            if k.any():
                lab = f"{lo:g} to {hi:g}" if hi < 1e8 else f"{lo:g}+"
                print(f"    {lab:>10} {k.sum():>6} " + " ".join(f"{R[n]['found'][j][k].mean() * 100:>10.0f}%" for n in names))
        k = keep & (a < 2)
        if k.any():
            print(f"    {'under 2':>10} {k.sum():>6} " + " ".join(f"{R[n]['found'][j][k].mean() * 100:>10.0f}%" for n in names))
    print(f"\n  {title}: rank against dry ground, median")
    print(f"    {'acres':>10} {'count':>6} " + " ".join(f"{n:>11}" for n in names))
    for lo, hi in S.SIZES:
        k = keep & (a >= lo) & (a < hi)
        if k.any():
            lab = f"{lo:g} to {hi:g}" if hi < 1e8 else f"{lo:g}+"
            print(f"    {lab:>10} {k.sum():>6} " + " ".join(f"{np.median(R[n]['rank'][k]):>11.2f}" for n in names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--states", default="")
    ap.add_argument("--elev", default="lidar", choices=("lidar", "10m"), help="1 m lidar elevation, or 3DEP's 10 m")
    a = ap.parse_args()
    global ELEV
    ELEV = a.elev
    print(f"elevation: {'3DEP 1/3 arc-second, about 10 m' if ELEV == '10m' else '3DEP 1 m lidar, read at 2 m'}", flush=True)
    peak = E.guard(14)
    t0 = time.time()
    h = np.load(HEAD)
    national = (h["W1"], h["b1"], h["W2"], h["b2"])
    want = [s for s in STATES if not a.states or s in a.states.split(",")]
    if a.pilot:
        want = want[:1]
    G, Tst, heads = {}, {}, {}
    for name in want:
        st = STATES[name]
        teach_at = st["teach"][:2] if a.pilot else st["teach"]
        test_at = dict(list(st["test"].items())[:1]) if a.pilot else st["test"]
        G[name], Tst[name] = [], []
        for i, (lon, lat) in enumerate(teach_at):
            g = "left out: the lidar run had no 1 m elevation here" if ELEV == "10m" and (lon, lat) in LIDAR_DROPPED else gather(st, lon, lat, False)
            what = g if isinstance(g, str) else f"NWI {g['nwi']:.1%} of pixels, the state's map {g['fine']:.1%}"
            print(f"  {name} teaching {i + 1} of {len(teach_at)} ({lon}, {lat}): {what}; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
            if not isinstance(g, str):
                G[name].append(g)
        for sq, (lon, lat) in test_at.items():
            g = gather(st, lon, lat, True)
            what = g if isinstance(g, str) else f"{len(g['idx'])} wetlands, NWI {g['nwi']:.1%} of pixels, the state's map {g['fine']:.1%}"
            print(f"  {name} held out, {sq}: {what}; {(time.time() - t0) / 60:.0f} min, memory {peak():.1f} GB", flush=True)
            if not isinstance(g, str):
                Tst[name].append(g)
    for name in want:
        if not G[name] or not Tst[name]:
            print(f"\n{name}: no squares to teach on or to score, skipped")
            continue
        rng = np.random.default_rng(0)
        heads[name] = {"NWI": (teach(G[name], "y_nwi", False, rng), False), "fine": (teach(G[name], "y_fine", False, rng), False),
                       "NWI + elev": (teach(G[name], "y_nwi", True, rng), True), "fine + elev": (teach(G[name], "y_fine", True, rng), True)}
        R = {"national": score(national, Tst[name], False)}
        R.update({k: score(m, Tst[name], e) for k, (m, e) in heads[name].items()})
        n_w = len(R["national"]["acres"])
        print(f"\n{name}: taught on {len(G[name])} squares, scored on {len(Tst[name])} held out ({n_w} wetlands of the state's map, "
              f"{R['national']['in_nwi'].mean():.0%} of them in NWI), {YEAR}")
        show(name, R)
        show(f"{name}, only wetlands NWI does not have", R, ~R["national"]["in_nwi"])
    if len(heads) > 1:
        print("\nAcross states: a state's fine heads scored on another state's held-out squares")
        for here in heads:
            R = {"national": score(national, Tst[here], False), "own fine": score(heads[here]["fine"][0], Tst[here], False),
                 "own f+elev": score(heads[here]["fine + elev"][0], Tst[here], True)}
            for there in heads:
                if there != here:
                    R[f"{there[:6]} fine"] = score(heads[there]["fine"][0], Tst[here], False)
                    R[f"{there[:6]} f+el"] = score(heads[there]["fine + elev"][0], Tst[here], True)
            show(f"scored in {here}", R)
    print(f"\n{(time.time() - t0) / 60:.0f} min, peak memory of the main process {peak():.1f} GB")


if __name__ == "__main__":
    main()
