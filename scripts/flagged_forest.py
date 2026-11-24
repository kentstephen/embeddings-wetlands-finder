# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""What is the forest that the head flags as wetland and NWI calls upland?

NWI is drawn from aerial photographs and is known to miss wetland under tree canopy, so some of the head's
"false alarms" in forest may be wetland NWI missed. This script cannot settle that, but it describes them.

The head is the one from terrain_adds.py on an embedding with terrain, scored on held-out 1 km blocks, with the
cutoff that finds 80 % of NWI's vegetated wetland. Forest is ESA WorldCover tree cover. Three groups of forest
pixels are compared: upland by NWI and flagged, upland by NWI and not flagged, and NWI forested/shrub wetland.

For each group: how far the flagged pixels are from NWI's line (a pixel within 10 or 30 m is as likely a
disagreement about where the line is as a missed wetland), whether they form patches, their terrain, and the
share of hydric soil under them. Hydric soil comes from the USDA soil survey (gNATSGO map units at 10 m from
Planetary Computer, hydric share of each map unit from Soil Data Access): a second opinion made by soil
scientists in the field, though its map units are far coarser than 10 m.

Nothing is cached. Peak memory about 8 GB; the process stops itself at 12 GB.

The figure (--fig, about 3 MB) shows, for each square with forest: the whole square over a lidar hillshade with
NWI and the flagged upland forest, then a close view of the largest flagged patch that sits away from any NWI
polygon, over the hillshade and over the newest NAIP aerial photo.

    uv run scripts/flagged_forest.py [--model AlphaEarth] [--year 2024] [--fig figures/flagged_forest.png]
"""
import argparse
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import pc  # noqa: E402
import terrain_adds as T  # noqa: E402

SDA = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
PATCH_HA, FAR_M = 0.5, 30
ZOOM_M, FINE = 900, 5  # half-height of the close view in meters; close-view cells per 10 m pixel
C_NWI, C_FLAG, C_BOX = "#0072B2", "#FFB000", "#CC79A7"  # blue against orange: reads without red or green


def hydric(s):
    """Percent of the soil map unit under each pixel that is rated hydric (NaN where the survey has none)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    y0, y1, x0, x1 = s["rc"]
    LON, LAT = np.meshgrid(E.X0 + (np.arange(x0, x1) + 0.5) * E.RES, E.Y0 - (np.arange(y0, y1) + 0.5) * E.RES)
    mk = np.zeros(LON.shape, np.int64)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3"):
        for it in pc.search("gnatsgo-rasters", s["box"]):
            with rasterio.open(pc.signed("gnatsgo-rasters", it["assets"]["mukey"]["href"])) as r:
                X, Y = Transformer.from_crs(4326, r.crs, always_xy=True).transform(LON, LAT)
                c = np.floor((X - r.transform.c) / r.transform.a).astype(np.int64)
                rw = np.floor((Y - r.transform.f) / r.transform.e).astype(np.int64)
                m = (c >= 0) & (c < r.width) & (rw >= 0) & (rw < r.height)
                if not m.any():
                    continue
                c0, r0 = c[m].min(), rw[m].min()
                a = r.read(1, window=Window(c0, r0, c[m].max() - c0 + 1, rw[m].max() - r0 + 1))
                mk[m] = a[rw[m] - r0, c[m] - c0]
    keys = np.unique(mk[mk > 0])
    q = ("SELECT mukey, SUM(CASE WHEN hydricrating = 'Yes' THEN comppct_r ELSE 0 END) FROM component WHERE mukey IN ("
         + ",".join(f"'{k}'" for k in keys) + ") GROUP BY mukey")
    req = urllib.request.Request(SDA, json.dumps({"query": q, "format": "JSON"}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        pct = {int(k): float(v or 0) for k, v in json.load(r).get("Table", [])}
    out = np.full(mk.shape, np.nan, np.float32)
    for k in keys:
        if int(k) in pct:
            out[mk == k] = pct[int(k)]
    return out, len(keys), len(pct)


def shade(z):
    """Hillshade (0 to 1) of a 2 m elevation grid, lit from the north-west, relief stretched six times."""
    gy, gx = np.gradient(np.nan_to_num(z, nan=float(np.nanmean(z))), T.STEP)
    slope, aspect = np.pi / 2 - np.arctan(6 * np.hypot(gx, gy)), np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    return (np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)).astype(np.float32)


def at(G, W, N, crs, LON, LAT, step):
    """Values of a projected grid (north-west corner W, N; square cells of `step`) at lon/lat points."""
    from pyproj import Transformer
    X, Y = Transformer.from_crs(4326, crs, always_xy=True).transform(LON, LAT)
    c = np.clip(np.floor((X - W) / step).astype(int), 0, G.shape[-1] - 1)
    r = np.clip(np.floor((N - Y) / step).astype(int), 0, G.shape[-2] - 1)
    return G[..., r, c]


def views(name, s, A, L, k, z, W, N, crs):
    """What the figure needs for one square: the whole square and a close view around patch k."""
    import imagery_adds as I
    from scipy.ndimage import uniform_filter
    hs = shade(z)
    y0, y1, x0, x1 = s["rc"]
    over = T.on_aef(uniform_filter(hs, 5)[2::5, 2::5], W, N, crs, s["rc"])[0]
    rr, cc = np.nonzero(L == k)
    hr = int(round(ZOOM_M / s["px_m"][1]))
    hc = int(round(ZOOM_M * 1.25 / s["px_m"][0]))
    r0, c0 = int(np.clip(rr.mean() - hr, 0, A.shape[0] - 2 * hr)), int(np.clip(cc.mean() - hc, 0, A.shape[1] - 2 * hc))
    r1, c1 = r0 + 2 * hr, c0 + 2 * hc
    lon = E.X0 + (x0 + c0 + (np.arange((c1 - c0) * FINE) + 0.5) / FINE) * E.RES
    lat = E.Y0 - (y0 + r0 + (np.arange((r1 - r0) * FINE) + 0.5) / FINE) * E.RES
    LON, LAT = np.meshgrid(lon, lat)
    ext = (lon[0], lon[-1], lat[-1], lat[0])
    q, W2, N2, crs2, days = I.read_naip((ext[0], ext[2], ext[1], ext[3]))
    photo = np.moveaxis(at(np.nan_to_num(q[:3]), W2, N2, crs2, LON, LAT, I.STEP), 0, -1)
    return dict(name=name, over=over, lab=s["lab"], A=A, box=s["box"], asp=s["px_m"][1] / s["px_m"][0], ext=ext,
                win=(r0, r1, c0, c1), hs=at(hs, W, N, crs, LON, LAT, T.STEP), photo=np.clip(photo * 1.5, 0, 1),
                days=days, center=(float(lon.mean()), float(lat.mean())))


def figure(V, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    def fill(ax, mask, color, alpha, ext):
        rgba = np.zeros(mask.shape + (4,), np.float32)
        rgba[mask] = to_rgba(color, alpha)
        ax.imshow(rgba, extent=ext, aspect=ax.get_aspect(), interpolation="nearest")

    def line(ax, mask, color, ext):
        h, w = mask.shape
        x = ext[0] + (np.arange(w) + 0.5) / w * (ext[1] - ext[0])
        y = ext[3] - (np.arange(h) + 0.5) / h * (ext[3] - ext[2])
        ax.contour(x, y, mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.3)

    fig, ax = plt.subplots(len(V), 3, figsize=(17, 5.4 * len(V)), constrained_layout=True, squeeze=False)
    for i, v in enumerate(V):
        ext = (v["box"][0], v["box"][2], v["box"][1], v["box"][3])
        r0, r1, c0, c1 = v["win"]
        a0, a1, a2 = ax[i]
        a0.imshow(v["over"], cmap="gray", vmin=0.2, vmax=1.1, extent=ext, aspect=v["asp"])
        fill(a0, v["lab"] > 0, C_NWI, 0.55, ext)
        fill(a0, v["A"], C_FLAG, 0.95, ext)
        e = v["ext"]
        a0.add_patch(Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fill=False, edgecolor=C_BOX, linewidth=1.8))
        a0.set_title(f"{v['name']}: the 10 km square", fontsize=10, loc="left")
        a1.imshow(v["hs"], cmap="gray", vmin=0.2, vmax=1.1, extent=e, aspect=v["asp"])
        fill(a1, v["A"][r0:r1, c0:c1], C_FLAG, 0.6, e)
        line(a1, v["lab"][r0:r1, c0:c1] > 0, "#56B4E9", e)
        a1.set_title(f"Close view at {v['center'][1]:.4f}, {v['center'][0]:.4f}: lidar hillshade", fontsize=10, loc="left")
        a2.imshow(v["photo"], extent=e, aspect=v["asp"])
        line(a2, v["A"][r0:r1, c0:c1], C_FLAG, e)
        line(a2, v["lab"][r0:r1, c0:c1] > 0, "#56B4E9", e)
        a2.set_title(f"Same view: NAIP aerial photo, {', '.join(v['days'])}", fontsize=10, loc="left")
        for x in ax[i]:
            x.tick_params(labelsize=7, length=2)
            for sp in x.spines.values():
                sp.set_visible(False)
    ax[0, 0].legend(handles=[Patch(color=C_NWI, alpha=0.55, label="NWI wetland or water"),
                             Patch(color=C_FLAG, label="forest NWI calls upland, flagged by the head"),
                             Line2D([], [], color=C_BOX, linewidth=1.8, label="close view")],
                    loc="upper right", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none")
    ax[0, 1].legend(handles=[Patch(color=C_FLAG, alpha=0.6, label="flagged forest"),
                             Line2D([], [], color="#56B4E9", linewidth=1.3, label="NWI line")],
                    loc="upper right", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"\nfigure: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AlphaEarth", choices=E.MODELS)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--fig", default="figures/flagged_forest.png")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    S = E.load_squares()
    hnd = E.open_model(a.model)
    print(f"\nHead: {a.model} {a.year} + terrain, held-out blocks, cutoff at 80 % of NWI vegetated wetland found.")
    V = []
    for name, s in S.items():
        z, W, N, crs, _ = T.read_dem(s["box"])
        Tn, t_ok = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
        X, ok = E.read_model(a.model, hnd, a.year, s["rc"])
        ok &= t_ok
        X = np.concatenate([X[ok], Tn[ok]], 1)
        lab, fold = s["lab"][ok], s["fold"][ok] % T.FOLDS
        up, veg = lab == 0, np.isin(lab, E.VEGETATED)
        use = veg | up
        p = T.held_out(X, veg, fold, use, np.random.default_rng(0))
        del X
        cut = np.quantile(p[use & veg], 0.2)
        P = np.full(s["lab"].shape, np.nan, np.float32)
        P[ok] = p
        forest = ok & (s["cover"] == 10)
        A = forest & (s["lab"] == 0) & (P >= cut)   # upland by NWI, flagged
        B = forest & (s["lab"] == 0) & (P < cut)    # upland by NWI, not flagged
        C = forest & (s["lab"] == 2)                # NWI forested/shrub wetland
        ha = s["px_m"][0] * s["px_m"][1] / 1e4
        print(f"\n{name}")
        if A.sum() + B.sum() < 500:
            print("  almost no forest: skipped")
            continue
        print(f"  upland forest by NWI: {(A.sum() + B.sum()) * ha:,.0f} ha, of which flagged {A.sum() * ha:,.0f} ha "
              f"({A.sum() / (A.sum() + B.sum()):.1%}). NWI forested/shrub wetland under tree cover: {C.sum() * ha:,.0f} ha, "
              f"of which flagged {(C & (P >= cut)).sum() / max(C.sum(), 1):.1%}.")
        d = s["edge"][A]
        print("  flagged upland forest by distance to the NWI line: "
              + ", ".join(f"{t} {m.mean():.0%}" for t, m in (("within 10 m", d <= 10), ("10 to 30 m", (d > 10) & (d <= 30)),
                                                              ("30 to 100 m", (d > 30) & (d <= 100)), ("over 100 m", d > 100))))
        L, n = label(A, structure=np.ones((3, 3)))
        size = np.bincount(L.ravel())[1:] * ha
        big = np.flatnonzero(size >= PATCH_HA) + 1
        far = [k for k in big if np.median(s["edge"][L == k]) > FAR_M] if len(big) < 5000 else []
        print(f"  patches: {n:,} in all; {len(big):,} of {PATCH_HA} ha or more hold {size[big - 1].sum() / max(size.sum(), 1e-9):.0%} "
              f"of the flagged area; {len(far):,} of those sit mostly over {FAR_M} m from any NWI polygon "
              f"({sum(size[k - 1] for k in far):,.0f} ha, largest {max((size[k - 1] for k in far), default=0):.1f} ha)")
        if a.fig and far and C.sum() >= 500:
            V.append(views(name, s, A, L, max(far, key=lambda k: size[k - 1]), z, W, N, crs))
        del z
        try:
            H, nk, nh = hydric(s)
            soil = f"soil survey: {nk} map units, {nh} with a hydric rating"
        except Exception as e:  # noqa: BLE001
            H, soil = None, f"soil survey not read ({type(e).__name__}: {e})"
        print(f"  {soil}")
        print(f"  {'':34}{'slope %':>9}{'above low':>11}{'depression':>12}{'hydric soil':>13}{'mostly hydric':>15}")
        for nm, M in (("upland by NWI, flagged", A), ("upland by NWI, not flagged", B), ("NWI forested/shrub wetland", C)):
            if not M.any():
                continue
            t = Tn[M]
            row = (f"  {nm:34}{np.median(np.expm1(t[:, 0])):>9.1f}{np.median(np.expm1(t[:, 8])):>9.2f} m"
                   f"{np.mean(np.expm1(t[:, 10]) > 0.05):>11.0%} ")
            if H is not None:
                h = H[M]
                h = h[np.isfinite(h)]
                row += f"{h.mean():>11.0f} %{np.mean(h >= 50):>14.0%}" if len(h) else f"{'none':>12}"
            print(row)
        print("    (median slope; median height above the lowest ground within 100 m; share of pixels more than 5 cm "
              "below a depression's fill level;\n     mean hydric share of the soil map unit; share of pixels on map units "
              "at least half hydric)", flush=True)
        del Tn
    if V:
        figure(V, f"Forest the head flags as wetland where NWI has none ({a.model} {a.year} + lidar terrain, held-out blocks)",
               Path(a.fig))
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
