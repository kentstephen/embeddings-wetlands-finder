# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""The saved wetland and water head as JSON for the web map (web/public/models/wetland_water_aef.json).

The weights are written row-major as the page reads them ([hidden][inputs], then [classes][hidden]), with 64 probe
embeddings (int8, as the store holds them) and the head's logits for them, so the page can check its own arithmetic
against this file's on the CPU and the GPU.

A head with elevation has eleven terrain numbers after the 64 bands. Its probe pixels carry made-up terrain numbers
too (probe_t), read as they are.

    uv run scripts/export_head.py [--head models/wetland-water-head-alphaearth.npz]
    uv run scripts/export_head.py --head models/wetland-water-head-alphaearth-elev10m.npz \
        --out web/public/models/wetland_water_aef_elev10m.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(ROOT / "models" / "wetland-water-head-alphaearth.npz"))
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "models" / "wetland_water_aef.json"))
    a = ap.parse_args()
    z = np.load(a.head, allow_pickle=False)
    W1, b1, W2, b2 = (z[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 128, size=(64, 64)).astype(np.int8)
    f = q.astype(np.float32) / 127.5
    x = np.sign(f) * f * f
    extra = {}
    if W1.shape[0] > 64:  # terrain numbers: the first (slope) is never under 0
        t = rng.normal(0, 1, size=(64, W1.shape[0] - 64)).astype(np.float32)
        t[:, 0] = np.abs(t[:, 0])
        x = np.hstack([x, t])
        extra = dict(probe_t=[round(float(v), 6) for v in t.ravel()], inputs=[str(v) for v in z["inputs"][64:]])
    out = np.maximum(x @ W1 + b1, 0) @ W2 + b2
    r = lambda v: [round(float(x), 6) for x in np.asarray(v).ravel()]  # noqa: E731
    d = dict(name=Path(a.head).stem, n_in=int(W1.shape[0]), hidden=int(W1.shape[1]), n_out=int(W2.shape[1]),
             classes=[str(c) for c in z["classes"]] if "classes" in z else ["upland", "vegetated wetland", "open water"],
             years=list(range(2017, 2026)), taught_on=int(len(z["taught_on"])),
             w1=r(W1.T), b1=r(b1), w2=r(W2.T), b2=r(b2), probe_q=[int(v) for v in q.ravel()], probe_out=r(out), **extra)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, separators=(",", ":")))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} KB): {d['n_in']} -> {d['hidden']} -> {d['n_out']}, taught on {d['taught_on']} squares")


if __name__ == "__main__":
    main()
