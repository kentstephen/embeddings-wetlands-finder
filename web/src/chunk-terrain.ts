import { RES, X0, Y0 } from "./aef";
import { type Dem, PER_DEG } from "./dem";
import { CELL, filled, fillHoles, MARGIN, metersPerDegree, N_TERRAIN, STEP, terrain } from "./terrain";

/**
 * The eleven terrain numbers under every pixel of one chunk of the mosaic, band-major [11][h * w].
 *
 * The ground is put on a 2 m grid in meters around the chunk's center (north up, MARGIN past each side), as the
 * Python reader puts it on a 2 m UTM grid; the numbers are worked out on it and averaged over 10 m cells, and each
 * pixel takes the cell its center falls in.
 *
 * One number is worked out another way than the Python: the depth below the level a depression would fill to. The
 * fill is done on the elevation model's own pixels (about 10 m) and carried to the 2 m grid, where the Python fills
 * the 2 m grid itself. The 2 m ground is those pixels blended, so it has the same depressions; the fill is the one
 * step that cannot run on the GPU, and this keeps it small.
 *
 * prepare() is the part done on the CPU for either path: the elevation window with its holes closed, its fill, and
 * where the 2 m grid and the chunk's pixels sit in it. terrain-gpu.ts does the rest on the GPU; cpuTerrain here is
 * the same arithmetic for a browser without WebGPU, and what the GPU is checked against.
 */
export type Job = {
  /** the 2 m grid */
  gw: number; gh: number;
  /** the elevation window */
  dw: number; dh: number;
  /** the window's pixel under the center of the 2 m cell at column c, row r: (u0 + c * du, v0 + r * dv) */
  u0: number; du: number; v0: number; dv: number;
  /** the 10 m grid */
  cols: number; rows: number;
  /** the chunk */
  pw: number; ph: number;
  /** [dw * dh] elevation less its mean, [dw * dh] fill level less the same, [pw] each pixel column's 10 m column, [ph] each row's */
  data: Float32Array;
};

/** Where the ground has none at all: null. */
export function prepare(px0: number, py0: number, pw: number, ph: number, d: Dem): Job | null {
  const lon0 = X0 + (px0 + pw / 2) * RES, lat0 = Y0 - (py0 + ph / 2) * RES;
  const [mLon, mLat] = metersPerDegree(lat0);
  const size = CELL * STEP;
  const cols = 2 * Math.ceil(((pw / 2) * RES * mLon + MARGIN) / size), rows = 2 * Math.ceil(((ph / 2) * RES * mLat + MARGIN) / size);
  const west = (-cols * size) / 2, north = (rows * size) / 2;
  const n = d.w * d.h;
  const data = new Float32Array(2 * n + pw + ph);
  const z = data.subarray(0, n);
  z.set(d.z);
  if (!fillHoles(z, d.w, d.h)) return null;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += z[i]!;
  mean /= n;
  for (let i = 0; i < n; i++) z[i] = z[i]! - mean;
  data.set(filled(z, d.w, d.h), n);
  for (let c = 0; c < pw; c++) data[2 * n + c] = Math.min(cols - 1, Math.max(0, Math.floor(((X0 + (px0 + c + 0.5) * RES - lon0) * mLon - west) / size)));
  for (let r = 0; r < ph; r++) data[2 * n + pw + r] = Math.min(rows - 1, Math.max(0, Math.floor((north - (Y0 - (py0 + r + 0.5) * RES - lat0) * mLat) / size)));
  return {
    gw: cols * CELL, gh: rows * CELL, dw: d.w, dh: d.h, cols, rows, pw, ph, data,
    u0: (lon0 + 180) * PER_DEG - 0.5 - d.gc0 + ((west + STEP / 2) / mLon) * PER_DEG, du: (STEP / mLon) * PER_DEG,
    v0: (90 - lat0) * PER_DEG - 0.5 - d.gr0 - ((north - STEP / 2) / mLat) * PER_DEG, dv: (STEP / mLat) * PER_DEG,
  };
}

/** The box of longitude and latitude a chunk's 2 m grid covers. */
export function boxOf(px0: number, py0: number, pw: number, ph: number): [number, number, number, number] {
  const lon0 = X0 + (px0 + pw / 2) * RES, lat0 = Y0 - (py0 + ph / 2) * RES;
  const [mLon, mLat] = metersPerDegree(lat0);
  const x = (Math.ceil(((pw / 2) * RES * mLon + MARGIN) / (CELL * STEP)) * CELL * STEP) / mLon;
  const y = (Math.ceil(((ph / 2) * RES * mLat + MARGIN) / (CELL * STEP)) * CELL * STEP) / mLat;
  return [lon0 - x, lat0 - y, lon0 + x, lat0 + y];
}

/** The numbers on the CPU: the same arithmetic as the GPU's, with the 2 m grid held in memory. */
export function cpuTerrain(j: Job): Float32Array {
  const { gw, gh, dw, dh, data } = j;
  const nd = dw * dh;
  const z = new Float32Array(gw * gh), depth = new Float32Array(gw * gh);
  for (let r = 0; r < gh; r++) {
    const v = Math.fround(Math.fround(j.v0) + r * Math.fround(j.dv));
    const r0 = Math.min(dh - 2, Math.max(0, Math.floor(v))), fy = Math.min(1, Math.max(0, v - r0));
    for (let c = 0; c < gw; c++) {
      const u = Math.fround(Math.fround(j.u0) + c * Math.fround(j.du));
      const c0 = Math.min(dw - 2, Math.max(0, Math.floor(u))), fx = Math.min(1, Math.max(0, u - c0));
      const i = r0 * dw + c0;
      const at = (o: number) => (data[o + i]! * (1 - fx) + data[o + i + 1]! * fx) * (1 - fy) + (data[o + i + dw]! * (1 - fx) + data[o + i + dw + 1]! * fx) * fy;
      const zv = at(0);
      z[r * gw + c] = zv;
      depth[r * gw + c] = Math.log(1 + Math.max(at(nd) - zv, 0));
    }
  }
  const t = terrain(z, gw, gh, depth)!;
  const n = j.pw * j.ph;
  const out = new Float32Array(N_TERRAIN * n);
  for (let r = 0; r < j.ph; r++) {
    const R = data[2 * nd + j.pw + r]!;
    for (let b = 0; b < N_TERRAIN; b++) {
      const src = t.f[b]!, o = b * n + r * j.pw;
      for (let c = 0; c < j.pw; c++) out[o + c] = src[R * t.cols + data[2 * nd + c]!]!;
    }
  }
  return out;
}

/** Made-up rolling ground with a level pond, as a job, for checking the GPU against the CPU. */
export function probeJob(): Job {
  const w = 220, h = 200;
  const z = new Float32Array(w * h);
  let s = 12345;
  const rnd = () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 4294967296);
  const k = Array.from({ length: 8 }, () => [rnd() * 0.25, rnd() * 0.25, rnd() * 6.28, 0.3 + rnd() * 3]);
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      let v = 40;
      for (const [a, b, p, m] of k) v += m! * Math.sin(a! * c + b! * r + p!);
      z[r * w + c] = r > 60 && r < 90 && c > 80 && c < 130 ? 36 : v + rnd() * 0.05;
    }
  // a chunk at 35 N, 90 W whose window this ground is taken to be
  const px0 = Math.round((-90 - X0) / RES), py0 = Math.round((Y0 - 35) / RES);
  const [west, , , north] = boxOf(px0, py0, 64, 64);
  return prepare(px0, py0, 64, 64, { z, gc0: Math.floor((west + 180) * PER_DEG) - 2, gr0: Math.floor((90 - north) * PER_DEG) - 2, w, h })!;
}
