/**
 * The wetland and water head (scripts/water_head.py, exported by scripts/export_head.py): one 64 -> 64 -> 3 network
 * on a single year's AlphaEarth embedding, dequantized sign(q) * (q / 127.5)^2. The standardization is folded into
 * the first layer. Outputs: the logits of upland, vegetated wetland, open water; a softmax makes them probabilities.
 *
 * Per pixel the scorer writes two bytes: wetland, 0 = no embedding, else 1 + round(p * 254); water, round(p * 255).
 */
export type WetHead = {
  name: string;
  n_in: number;
  hidden: number;
  n_out: number;
  classes: string[];
  years: number[];
  taught_on: number;
  /** [hidden][n_in] row-major, then [n_out][hidden] */
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
  /** 64 embeddings as the store holds them (int8, pixel-major) and the head's logits for them */
  probe_q: number[];
  probe_out: number[];
};

export const NODATA = -128;
export const WET_STEPS = 254;

/** A wetland byte back to a probability; NaN for 0 (no embedding). */
export const wetOf = (c: number) => (c > 0 ? (c - 1) / WET_STEPS : NaN);
export const waterOf = (c: number) => c / 255;

/** The weights in one array, as the GPU reads them: w1, b1, w2, b2. */
export function packHead(h: WetHead): Float32Array {
  const out = new Float32Array(h.w1.length + h.b1.length + h.w2.length + h.b2.length);
  let o = 0;
  for (const a of [h.w1, h.b1, h.w2, h.b2]) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}

export const deq = (q: number) => {
  const f = q / 127.5;
  return f * Math.abs(f);
};

/** The logits for one pixel of a band-major chunk; false where it has no embedding (band 0 at -128). */
function forward(h: WetHead, w: Float32Array, e: Int8Array, n: number, p: number, x: Float32Array, hid: Float32Array, z: Float32Array): boolean {
  const NI = h.n_in;
  if (e[p] === NODATA) return false;
  for (let b = 0; b < NI; b++) x[b] = deq(e[b * n + p]!);
  const ob1 = h.hidden * NI, o2 = ob1 + h.hidden, ob2 = o2 + h.n_out * h.hidden;
  for (let j = 0; j < h.hidden; j++) {
    let s = w[ob1 + j]!;
    const r = j * NI;
    for (let i = 0; i < NI; i++) s += w[r + i]! * x[i]!;
    hid[j] = s > 0 ? s : 0;
  }
  for (let c = 0; c < h.n_out; c++) {
    let s = w[ob2 + c]!;
    const r = o2 + c * h.hidden;
    for (let j = 0; j < h.hidden; j++) s += w[r + j]! * hid[j]!;
    z[c] = s;
  }
  return true;
}

/** The two bytes of a pixel from its logits (upland, wetland, water). */
export function bytesOf(z: ArrayLike<number>): [number, number] {
  const m = Math.max(z[0]!, z[1]!, z[2]!);
  const a = Math.exp(z[0]! - m), b = Math.exp(z[1]! - m), c = Math.exp(z[2]! - m);
  const s = a + b + c;
  return [1 + Math.round((b / s) * WET_STEPS), Math.round((c / s) * 255)];
}

/** The head on the CPU, for when WebGPU is missing: e is one year, band-major int8 [64, n]; out [wetland n, water n]. */
export function scoreCpu(h: WetHead, w: Float32Array, e: Int8Array, n: number): Uint8Array {
  const out = new Uint8Array(2 * n);
  const x = new Float32Array(h.n_in), hid = new Float32Array(h.hidden), z = new Float32Array(h.n_out);
  for (let p = 0; p < n; p++) {
    if (!forward(h, w, e, n, p, x, hid, z)) continue;
    const [a, b] = bytesOf(z);
    out[p] = a;
    out[n + p] = b;
  }
  return out;
}

/** The probe embeddings as a band-major chunk. */
export function probeChunk(h: WetHead): { e: Int8Array; n: number } {
  const n = h.probe_q.length / h.n_in;
  const e = new Int8Array(h.probe_q.length);
  for (let p = 0; p < n; p++) for (let b = 0; b < h.n_in; b++) e[b * n + p] = h.probe_q[p * h.n_in + b]!;
  return { e, n };
}

/** The largest gap between this code's logits and the export's own on its probe embeddings. */
export function probeGap(h: WetHead, w: Float32Array): number {
  const { e, n } = probeChunk(h);
  const x = new Float32Array(h.n_in), hid = new Float32Array(h.hidden), z = new Float32Array(h.n_out);
  let worst = 0;
  for (let p = 0; p < n; p++) {
    forward(h, w, e, n, p, x, hid, z);
    for (let c = 0; c < h.n_out; c++) worst = Math.max(worst, Math.abs(z[c]! - h.probe_out[p * h.n_out + c]!));
  }
  return worst;
}
