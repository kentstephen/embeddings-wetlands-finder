/**
 * The eleven terrain numbers the head with elevation reads, worked out as scripts/terrain_adds.py works them out
 * (terrain()): elevation on a 2 m grid, each number computed there and averaged over 10 m cells. In order: slope,
 * curvature, roughness inside the cell, height against the mean of the surroundings at 10, 30, 100 and 300 m, height
 * above the lowest ground within 30, 100 and 300 m, and depth below the level a depression would fill to.
 *
 * Edges are mirrored, as scipy's default ("reflect") does. A margin wider than the largest window keeps the edges
 * out of the part that is used.
 */
export const STEP = 2;
export const CELL = 5; // 2 m steps in a 10 m cell
export const N_TERRAIN = 11;
/** Ground read past each side of what is scored, meters: the Python reader's own margin. */
export const MARGIN = 400;

const steps = (m: number) => Math.max(Math.round(m / STEP), 1);

/** A running mean over `size` (odd) along rows then columns, mirrored at the edges. */
function boxMean(src: Float32Array, w: number, h: number, size: number, out: Float32Array = new Float32Array(w * h)): Float32Array {
  const r = (size - 1) >> 1;
  const tmp = new Float32Array(w * h);
  const line = new Float64Array(Math.max(w, h) + 2 * r + 1);
  const pass = (from: Float32Array, to: Float32Array, n: number, stride: number, lines: number, lineStride: number) => {
    for (let l = 0; l < lines; l++) {
      const base = l * lineStride;
      // prefix sums of the mirrored line: index i of the line sits at p = i + r
      line[0] = 0;
      for (let p = 0; p < n + 2 * r; p++) {
        let i = p - r;
        if (i < 0) i = -i - 1;
        else if (i >= n) i = 2 * n - i - 1;
        if (i < 0) i = 0;
        else if (i >= n) i = n - 1;
        line[p + 1] = line[p]! + from[base + i * stride]!;
      }
      for (let i = 0; i < n; i++) to[base + i * stride] = (line[i + size]! - line[i]!) / size;
    }
  };
  pass(src, tmp, w, 1, h, w);
  pass(tmp, out, h, w, w, 1);
  return out;
}

/** The lowest value within `r` cells along rows then columns (a square window; mirroring adds nothing new to it). */
function boxMin(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const dq = new Int32Array(Math.max(w, h));
  const pass = (from: Float32Array, to: Float32Array, n: number, stride: number, lines: number, lineStride: number) => {
    for (let l = 0; l < lines; l++) {
      const base = l * lineStride;
      let head = 0, tail = 0, next = 0;
      for (let i = 0; i < n; i++) {
        const hi = Math.min(n - 1, i + r);
        for (; next <= hi; next++) {
          const v = from[base + next * stride]!;
          while (tail > head && from[base + dq[tail - 1]! * stride]! >= v) tail--;
          dq[tail++] = next;
        }
        while (dq[head]! < i - r) head++;
        to[base + i * stride] = from[base + dq[head]! * stride]!;
      }
    }
  };
  pass(src, tmp, w, 1, h, w);
  pass(tmp, out, h, w, w, 1);
  return out;
}

/** A Gaussian blur of sigma cells, cut off at four sigma, mirrored at the edges. */
function gauss(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const r = Math.floor(4 * sigma + 0.5);
  const k = new Float64Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) sum += k[i + r] = Math.exp((-0.5 * i * i) / (sigma * sigma));
  for (let i = 0; i < k.length; i++) k[i] = k[i]! / sum;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const pass = (from: Float32Array, to: Float32Array, n: number, stride: number, lines: number, lineStride: number) => {
    for (let l = 0; l < lines; l++) {
      const base = l * lineStride;
      for (let i = 0; i < n; i++) {
        let s = 0;
        if (i >= r && i < n - r) for (let j = -r; j <= r; j++) s += k[j + r]! * from[base + (i + j) * stride]!;
        else
          for (let j = -r; j <= r; j++) {
            let q = i + j;
            if (q < 0) q = -q - 1;
            else if (q >= n) q = 2 * n - q - 1;
            s += k[j + r]! * from[base + Math.min(n - 1, Math.max(0, q)) * stride]!;
          }
        to[base + i * stride] = s;
      }
    }
  };
  pass(src, tmp, w, 1, h, w);
  pass(tmp, out, h, w, w, 1);
  return out;
}

/** The level every cell would fill to if water could only leave at the grid's edge (eight neighbors). */
export function filled(z: Float32Array, w: number, h: number): Float32Array {
  const N = w * h;
  const fill = new Float32Array(N);
  const seen = new Uint8Array(N);
  const hk = new Float32Array(N), hi = new Int32Array(N);
  let hn = 0;
  const push = (k: number, i: number) => {
    let p = hn++;
    while (p > 0) {
      const q = (p - 1) >> 1;
      if (hk[q]! <= k) break;
      hk[p] = hk[q]!;
      hi[p] = hi[q]!;
      p = q;
    }
    hk[p] = k;
    hi[p] = i;
  };
  const pop = () => {
    const top = hi[0]!;
    const k = hk[--hn]!, i = hi[hn]!;
    let p = 0;
    for (;;) {
      let c = 2 * p + 1;
      if (c >= hn) break;
      if (c + 1 < hn && hk[c + 1]! < hk[c]!) c++;
      if (hk[c]! >= k) break;
      hk[p] = hk[c]!;
      hi[p] = hi[c]!;
      p = c;
    }
    hk[p] = k;
    hi[p] = i;
    return top;
  };
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++)
      if (r === 0 || c === 0 || r === h - 1 || c === w - 1) {
        const i = r * w + c;
        fill[i] = z[i]!;
        seen[i] = 1;
        push(fill[i]!, i);
      }
  while (hn > 0) {
    const i = pop();
    const v = fill[i]!;
    const r = (i / w) | 0, c = i - r * w;
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
        const j = rr * w + cc;
        if (seen[j]) continue;
        seen[j] = 1;
        fill[j] = z[j]! > v ? z[j]! : v;
        push(fill[j]!, j);
      }
  }
  return fill;
}

/** Ground from the nearest cell that has some, into the cells that have none (NaN). False when no cell has any. */
export function fillHoles(z: Float32Array, w: number, h: number): boolean {
  const N = w * h;
  const queue = new Int32Array(N);
  let n = 0, holes = 0;
  for (let i = 0; i < N; i++) {
    if (z[i]! === z[i]!) queue[n++] = i;
    else holes++;
  }
  if (!n) return false;
  if (!holes) return true;
  for (let q = 0; q < n; q++) {
    const i = queue[q]!;
    const r = (i / w) | 0, c = i - r * w;
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
        const j = rr * w + cc;
        if (z[j]! === z[j]!) continue;
        z[j] = z[i]!;
        queue[n++] = j;
      }
  }
  return true;
}

/**
 * The eleven numbers on the 10 m grid: z is elevation on a 2 m grid, w by h, NaN where there is none (filled here
 * from the nearest ground). Returns [11][rows * cols] with rows = floor(h / 5), cols = floor(w / 5), or null when the
 * grid has no ground at all. depth: the last number (log of one plus the depth below the fill level) already worked
 * out on the 2 m grid, in place of filling the 2 m ground here.
 */
export function terrain(z: Float32Array, w: number, h: number, depth?: Float32Array): { f: Float32Array[]; rows: number; cols: number } | null {
  if (!fillHoles(z, w, h)) return null;
  const N = w * h;
  const rows = Math.floor(h / CELL), cols = Math.floor(w / CELL);
  const f: Float32Array[] = [];
  const tmp = new Float32Array(N);
  /** A 2 m layer averaged over the 10 m cells. */
  const keep = (a: Float32Array) => {
    const out = new Float32Array(rows * cols);
    for (let R = 0; R < rows; R++)
      for (let C = 0; C < cols; C++) {
        let s = 0;
        for (let dr = 0; dr < CELL; dr++) {
          const o = (R * CELL + dr) * w + C * CELL;
          for (let dc = 0; dc < CELL; dc++) s += a[o + dc]!;
        }
        const v = s / (CELL * CELL);
        out[R * cols + C] = v === v && Math.abs(v) !== Infinity ? v : 0;
      }
    f.push(out);
  };

  // slope and curvature of the smoothed ground
  const zs = gauss(z, w, h, 2);
  for (let r = 0; r < h; r++) {
    const ra = r > 0 ? r - 1 : r, rb = r < h - 1 ? r + 1 : r;
    for (let c = 0; c < w; c++) {
      const ca = c > 0 ? c - 1 : c, cb = c < w - 1 ? c + 1 : c;
      const gy = (zs[rb * w + c]! - zs[ra * w + c]!) / ((rb - ra) * STEP);
      const gx = (zs[r * w + cb]! - zs[r * w + ca]!) / ((cb - ca) * STEP);
      tmp[r * w + c] = Math.log1p(100 * Math.hypot(gx, gy));
    }
  }
  keep(tmp);
  for (let r = 0; r < h; r++) {
    const ra = r > 0 ? r - 1 : r, rb = r < h - 1 ? r + 1 : r;
    for (let c = 0; c < w; c++) {
      const ca = c > 0 ? c - 1 : c, cb = c < w - 1 ? c + 1 : c;
      const i = r * w + c;
      tmp[i] = (zs[ra * w + c]! + zs[rb * w + c]! + zs[r * w + ca]! + zs[r * w + cb]! - 4 * zs[i]!) / (STEP * STEP);
    }
  }
  keep(tmp);

  // roughness: how far the ground strays from its 10 m mean
  const mean = boxMean(z, w, h, steps(10));
  const d2 = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const d = z[i]! - mean[i]!;
    d2[i] = d * d;
  }
  boxMean(d2, w, h, steps(10), mean);
  for (let i = 0; i < N; i++) tmp[i] = Math.log1p(100 * Math.sqrt(Math.max(mean[i]!, 0)));
  keep(tmp);

  // height against the mean of the surroundings
  for (const m of [10, 30, 100, 300]) {
    boxMean(z, w, h, 2 * steps(m) + 1, mean);
    for (let i = 0; i < N; i++) tmp[i] = z[i]! - mean[i]!;
    keep(tmp);
  }
  // height above the lowest ground nearby
  for (const m of [30, 100, 300]) {
    const lo = boxMin(z, w, h, steps(m));
    for (let i = 0; i < N; i++) tmp[i] = Math.log1p(Math.max(z[i]! - lo[i]!, 0));
    keep(tmp);
  }
  // depth below the level a depression would fill to
  if (depth) keep(depth);
  else {
    const top = filled(z, w, h);
    for (let i = 0; i < N; i++) tmp[i] = Math.log1p(Math.max(top[i]! - z[i]!, 0));
    keep(tmp);
  }
  return { f, rows, cols };
}

/** Meters in a degree of longitude and of latitude at a latitude (WGS 84). */
export function metersPerDegree(lat: number): [number, number] {
  const p = (lat * Math.PI) / 180;
  return [111412.84 * Math.cos(p) - 93.5 * Math.cos(3 * p) + 0.118 * Math.cos(5 * p), 111132.92 - 559.82 * Math.cos(2 * p) + 1.175 * Math.cos(4 * p) - 0.0023 * Math.cos(6 * p)];
}
