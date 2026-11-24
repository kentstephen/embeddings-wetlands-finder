# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3.1", "obstore", "pyproj", "scikit-learn", "scipy", "scikit-image", "rasterio", "shapely", "matplotlib", "psutil"]
# ///
"""Do autumn aerial photos find wetland under trees where summer photos do not?

NAIP is flown in summer almost everywhere, with leaves on. Over the Lee, New Hampshire square two flights were
late: 16 and 26 October 2018 and 24 October 2021, when much of the hardwood canopy has turned or dropped. This
compares them with the 23 August 2023 flight on the same pixels, using the eleven numbers per pixel of
imagery_adds.py, the same held-out blocks and the same small head.

Each flight is scored alone, with terrain, and on top of an embedding with terrain. Late October in southern New
Hampshire is past peak color but oaks and beech still hold leaves, so this is partly leaf-off, not bare canopy.

Nothing fetched is cached. Peak memory about 8 GB; the process stops itself at 12 GB.

    uv run scripts/naip_leafoff.py [--year 2024]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_signal as E  # noqa: E402
import imagery_adds as I  # noqa: E402
import terrain_adds as T  # noqa: E402

SQUARE = "Lee, NH"
FLIGHTS = {"August 2023": ("2023-08",), "October 2018": ("2018-10",), "October 2021": ("2021-10",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    peak = E.guard(12)
    E.SQUARES = {SQUARE: E.SQUARES[SQUARE]}
    s = E.load_squares()[SQUARE]
    z, W, N, crs, _ = T.read_dem(s["box"])
    Tn, ok = T.on_aef(T.terrain(z), W, N, crs, s["rc"])
    del z
    F = {}
    for name, days in FLIGHTS.items():
        t = time.time()
        z, W, N, crs, used = I.read_naip(s["box"], days)
        f, have = I.imagery(z)
        F[name], i_ok = T.on_aef(f, W, N, crs, s["rc"])
        i_ok &= T.on_aef(have, W, N, crs, s["rc"])[0]
        ok &= i_ok
        print(f"  {name}: flown {', '.join(used)}, {i_ok.mean():.1%} of pixels have a photo, {time.time() - t:.0f} s, "
              f"peak {peak():.1f} GB", flush=True)
        del z
    print(f"  pixels with terrain and all three flights: {ok.mean():.1%}", flush=True)

    R = {}

    def run(col, X):
        R[col] = T.score(X, s, ok, np.random.default_rng(0))
        print(f"  {col} done, peak {peak():.1f} GB", flush=True)

    both = np.concatenate([F["August 2023"][ok], F["October 2021"][ok]], 1)
    run("terrain", Tn[ok])
    for name in FLIGHTS:
        run(name, F[name][ok])
    run("August 2023 + October 2021", both)
    for name in FLIGHTS:
        run(f"terrain + {name}", np.concatenate([Tn[ok], F[name][ok]], 1))
    for model in ("AlphaEarth", "OlmoEarth"):
        X, k = E.read_model(model, E.open_model(model), a.year, s["rc"])
        if not k[ok].all():
            print(f"  {model}: {(~k[ok]).sum()} scored pixels have no embedding and are read as zeros", flush=True)
        X = np.concatenate([X[ok], Tn[ok]], 1)
        run(f"{model} + terrain", X)
        for name in FLIGHTS:
            run(f"{model} + terrain + {name}", np.concatenate([X, F[name][ok]], 1))
        del X

    print(f"\n{SQUARE}. AUC (and, at 80 % of wetland found, the share of flagged pixels that are wetland). Small head, "
          "scored on 1 km blocks it was not taught on.")
    for t in T.TESTS:
        print(f"\n{t}")
        for c, v in R.items():
            if t in v:
                print(f"    {c:40} {v[t][0]:.3f} ({v[t][1]:.0%})")
    print(f"\npeak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
