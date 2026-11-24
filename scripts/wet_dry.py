# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Does the wetland signal in a yearly embedding depend on whether the year was wet or dry?

For each square the wettest and driest calendar years of 2017 to 2025 are picked from daily precipitation at the
square's center (ERA5, read from Open-Meteo's archive). The same linear probe as embedding_signal.py is then
taught and scored four ways on the same held-out 1 km blocks: taught and scored in the wet year, taught and
scored in the dry year, taught in the wet year and scored in the dry one, and the reverse. The NWI labels are the
same for both years, so a change in score is a change in what the embedding shows, or in the ground since the
NWI photo year, not in the answer key.

It also reports how much of each NWI kind the wet-year probe calls wetland (probability of 0.5 or more) when it is
shown the wet year and when it is shown the dry year.

A calendar year's rain is a rough stand-in for how wet the ground was: basins also carry water over from earlier
years. Nothing is cached. Peak memory about 4 GB; the process stops itself at 12 GB.

    uv run scripts/wet_dry.py [--wet 2019 --dry 2020]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402

YEARS = range(2017, 2026)
WAYS = ("wet -> wet", "dry -> dry", "wet -> dry", "dry -> wet")  # taught on -> scored on
GROUPS = (("all vegetated", E.VEGETATED), ("emergent", (1,)), ("forested/shrub", (2,)))


def rain(lon, lat):
    """Calendar-year precipitation in mm at a point."""
    u = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={YEARS[0]}-01-01"
         f"&end_date={YEARS[-1]}-12-31&daily=precipitation_sum&timezone=UTC")
    with urllib.request.urlopen(u, timeout=120) as r:
        d = json.load(r)["daily"]
    out = dict.fromkeys(YEARS, 0.0)
    for t, v in zip(d["time"], d["precipitation_sum"]):
        out[int(t[:4])] += v or 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wet", type=int)
    ap.add_argument("--dry", type=int)
    a = ap.parse_args()
    peak = E.guard(12)
    S = E.load_squares()
    for name, s in S.items():
        mm = rain(*E.SQUARES[name][:2])
        s["wet"], s["dry"] = a.wet or max(mm, key=mm.get), a.dry or min(mm, key=mm.get)
        print(f"  {name}: wet year {s['wet']} ({mm[s['wet']]:.0f} mm), dry year {s['dry']} ({mm[s['dry']]:.0f} mm); "
              + ", ".join(f"{y} {v:.0f}" for y, v in mm.items()), flush=True)

    R, C = {}, {}
    for model in E.MODELS:
        hnd = E.open_model(model)
        for name, s in S.items():
            rng = np.random.default_rng(E.SEED)
            Xw, okw = E.read_model(model, hnd, s["wet"], s["rc"])
            Xd, okd = E.read_model(model, hnd, s["dry"], s["rc"])
            ok = okw & okd
            X = {"wet": Xw[ok], "dry": Xd[ok]}
            del Xw, Xd
            y_all, fold = s["lab"][ok], s["fold"][ok]
            up, veg = y_all == 0, np.isin(y_all, E.VEGETATED)
            use = veg | up
            P = {w: np.full(len(veg), np.nan, np.float32) for w in WAYS}
            for k in range(E.FOLDS):
                tr, te = np.flatnonzero(use & (fold != k)), np.flatnonzero(use & (fold == k))
                if not len(te) or not 0 < veg[tr].sum() < len(tr):
                    continue
                for src in ("wet", "dry"):
                    m = E.fit(X[src], veg, tr, rng)
                    for dst in ("wet", "dry"):
                        P[f"{src} -> {dst}"][te] = m.predict_proba(X[dst][te])[:, 1]
            for g, codes in GROUPS:
                yes = np.isin(y_all, codes)
                sel = (yes | up) & np.isfinite(P[WAYS[0]])
                if yes[sel].sum() >= 500:
                    R[model, name, g] = [E.auc(yes[sel], P[w][sel]) for w in WAYS]
            for g, codes in (("emergent", (1,)), ("forested/shrub", (2,)), ("upland", (0,))):
                sel = np.isin(y_all, codes) & np.isfinite(P[WAYS[0]])
                if sel.sum() >= 500:
                    C[model, name, g] = [float((P[w][sel] >= 0.5).mean()) for w in ("wet -> wet", "wet -> dry")]
            del X
            print(f"  {model} {name} done, peak memory {peak():.1f} GB", flush=True)

    print("\nAUC of a linear probe, taught on -> scored on, same held-out 1 km blocks")
    for name, s in S.items():
        print(f"\n  {name} (wet {s['wet']}, dry {s['dry']})")
        print(f"    {'':28}" + "".join(f"{w:>12}" for w in WAYS))
        for g, _ in GROUPS:
            for model in E.MODELS:
                if (model, name, g) in R:
                    print(f"    {g + ', ' + model:28}" + "".join(f"{v:>12.3f}" for v in R[model, name, g]))
    print("\nShare of each NWI kind the wet-year probe calls wetland (probability 0.5 or more): shown the wet year, "
          "shown the dry year")
    for name in S:
        print(f"\n  {name}")
        for g in ("emergent", "forested/shrub", "upland"):
            for model in E.MODELS:
                if (model, name, g) in C:
                    w, d = C[model, name, g]
                    print(f"    {g + ', ' + model:28} {w:>6.1%} {d:>6.1%}")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
