# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil", "pyshp"]
# ///
"""How does a head taught on NWI score against a wetland map made at a field research site?

Every other score in this repository is agreement with NWI. The USGS Cottonwood Lake Study Area (Stutsman County,
North Dakota, about 1 km2 near the center of that square) has yearly maps of the vegetation zones of 17 prairie
pothole wetlands, 1975 to 2015 (Mushet and others 2017, doi:10.5066/F7J101DQ, public domain). They were drawn by the
site's own crew from low-altitude photographs of the site tied to GPS ground control: not a line walked on the
ground, but a far closer look than NWI's, by people who work in those wetlands every year.

The head is the one from terrain_adds.py, taught on NWI in the 1 km blocks of the square it is not scored on. On
the pixels inside the study area it is scored twice: against NWI, and against the site map, where wet meadow,
shallow marsh, deep marsh and shrub zones are wetland, ground inside the study area boundary and outside every
zone is taken as upland, and open water and exposed soil are left out. NWI and the site map are also compared
with each other on the same pixels.

The site map is from 2015 and the embeddings start in 2017 (the default year here), so two years of change sit
between them. Nothing is cached. Peak memory about 6 GB; the process stops itself at 12 GB.

    uv run scripts/field_check.py [--year 2017]
"""
import argparse
import io
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import terrain_adds as T  # noqa: E402

SB = "https://www.sciencebase.gov/catalog/file/get/"
ZONES = SB + "58c84d0ee4b0849ce97961c7"  # vegetation zones, 2015
BOUND = SB + "5c5496c6e4b0708288fee15a"  # study area boundary
SQUARE = "Stutsman County, ND"
WET_ZONES = ("wetmeadow", "shallowmarsh", "deepmarsh", "tree_shrub")
LEFT_OUT = ("openwater", "expsoil")


def shapes(url):
    """(geometry in lon/lat, attributes) for every record of the first shapefile found in a zip, nested or not."""
    import shapefile
    from pyproj import CRS, Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    raw = E.again(lambda: urllib.request.urlopen(url, timeout=180).read())

    def parts(zf):
        names = zf.namelist()
        shp = next((n for n in names if n.lower().endswith(".shp")), None)
        if shp:
            base = shp[:-4]
            get = lambda ext: next((zf.read(n) for n in names if n.lower() == (base + ext).lower()), None)  # noqa: E731
            return get(".shp"), get(".dbf"), get(".shx"), get(".prj")
        for n in names:
            if n.lower().endswith(".zip"):
                got = parts(zipfile.ZipFile(io.BytesIO(zf.read(n))))
                if got:
                    return got
        return None

    shp, dbf, shx, prj = parts(zipfile.ZipFile(io.BytesIO(raw)))
    tf = Transformer.from_crs(CRS.from_wkt(prj.decode()), 4326, always_xy=True).transform
    r = shapefile.Reader(shp=io.BytesIO(shp), dbf=io.BytesIO(dbf), shx=io.BytesIO(shx))
    return [(transform(tf, shape(sr.shape.__geo_interface__)), sr.record.as_dict()) for sr in r.shapeRecords()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2017)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    E.SQUARES = {SQUARE: E.SQUARES[SQUARE]}
    s = E.load_squares()[SQUARE]
    zones = shapes(ZONES)
    kinds = sorted({str(r.get("Zone")).lower() for _, r in zones})
    code = {k: i + 1 for i, k in enumerate(kinds)}
    site = E.burn(sorted(((g, code[str(r.get("Zone")).lower()]) for g, r in zones), key=lambda t: -t[0].area), s["rc"])
    area = E.burn([(g, 1) for g, _ in shapes(BOUND)], s["rc"]) > 0
    ha = s["px_m"][0] * s["px_m"][1] / 1e4
    print(f"\nSite map: {len(zones)} zone polygons ({', '.join(kinds)}); study area {area.sum() * ha:.0f} ha, "
          f"{area.sum():,} pixels; zones cover {(area & (site > 0)).sum() * ha:.0f} ha of it")
    wet_s = np.isin(site, [code[k] for k in WET_ZONES if k in code])
    out_s = np.isin(site, [code[k] for k in LEFT_OUT if k in code])
    up_s = site == 0
    lab = s["lab"]
    inA = area & ~out_s
    print(f"  inside the study area, open water and exposed soil left out: site wetland {(inA & wet_s).sum():,} pixels, "
          f"site upland {(inA & up_s).sum():,}")
    print("  NWI on the same pixels: " + ", ".join(
        f"{E.NAME[c]} {np.mean(lab[inA] == c):.0%}" for c in sorted(E.NAME) if (lab[inA] == c).any()))
    print(f"  of the site's wetland pixels, NWI has a polygon on {np.mean(lab[inA & wet_s] > 0):.0%} "
          f"(a vegetated kind on {np.isin(lab[inA & wet_s], E.VEGETATED).mean():.0%}); "
          f"of the site's upland pixels, NWI has a polygon on {np.mean(lab[inA & up_s] > 0):.0%}")

    z, W, N, crs, _ = T.read_dem(s["box"])
    Tn, t_ok = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
    del z

    def held_out_all(X, ok):
        """Probability for every valid pixel from a head taught on NWI in the other folds."""
        y_all, fold = lab[ok], s["fold"][ok] % T.FOLDS
        veg, use = np.isin(y_all, E.VEGETATED), np.isin(y_all, E.VEGETATED) | (y_all == 0)
        rng = np.random.default_rng(0)
        p = np.full(len(y_all), np.nan, np.float32)
        for k in range(T.FOLDS):
            tr, te = np.flatnonzero(use & (fold != k)), np.flatnonzero(fold == k)
            if len(tr) > T.TRAIN_MAX:
                tr = rng.choice(tr, T.TRAIN_MAX, replace=False)
            p[te] = T.head().fit(X[tr], veg[tr]).predict_proba(X[te])[:, 1]
        P = np.full(lab.shape, np.nan, np.float32)
        P[ok] = p
        return P, np.quantile(p[use & veg], 0.2)

    def report(col, X, ok):
        P, cut = held_out_all(X, ok)
        m = inA & ok & (wet_s | up_s)
        n = inA & ok & (np.isin(lab, E.VEGETATED) | (lab == 0))
        sq = ok & (np.isin(lab, E.VEGETATED) | (lab == 0))
        print(f"    {col:24} {E.auc(wet_s[m], P[m]):>9.3f} {E.auc(np.isin(lab[n], E.VEGETATED), P[n]):>9.3f} "
              f"{E.auc(np.isin(lab[sq], E.VEGETATED), P[sq]):>9.3f} {np.mean(P[m & wet_s] >= cut):>9.0%} {np.mean(P[m & up_s] >= cut):>9.0%}", flush=True)

    print(f"\nHead taught on NWI ({a.year} embeddings), held-out blocks. AUC against the site map on the study area, "
          "against NWI on the study area,\nand against NWI on the whole square; then the share of the site's wetland and "
          "upland pixels the head flags\nat the cutoff that finds 80 % of NWI's vegetated wetland in the square.")
    print(f"    {'':24} {'site map':>9} {'NWI here':>9} {'NWI all':>9} {'wet flag':>9} {'up flag':>9}")
    report("terrain", Tn[t_ok], t_ok)
    for model in E.MODELS:
        X, ok = E.read_model(model, E.open_model(model), a.year, s["rc"])
        ok &= t_ok
        report(model, X[ok], ok)
        report(model + " + terrain", np.concatenate([X[ok], Tn[ok]], 1), ok)
        del X
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
