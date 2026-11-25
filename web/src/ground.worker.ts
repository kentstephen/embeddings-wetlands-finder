/**
 * The ground under a view: elevation read from terrain tiles, and how much ground drains through every pixel.
 *
 * Elevation: the Terrarium tiles of the AWS Open Data terrain set (Mapzen's mosaic; in the United States from USGS
 * 3DEP at about 10 m), meters = R * 256 + G + B / 256 - 32768.
 *
 * Flow: depressions are filled with a tiny slope so every pixel drains (priority flood from the edge of the scene),
 * each pixel sends its water to its steepest lower neighbor, and the pixels upstream of each one are counted. A level
 * water surface takes one count all over. The scene is the tiles asked for, so a stream that enters at the edge starts
 * from nothing there.
 */
export type GroundAsk = { id: number; z: number; x0: number; x1: number; y0: number; y1: number; flow: boolean };
export type GroundAnswer = { id: number; z: number; x0: number; y0: number; w: number; h: number; elev: Float32Array; flow: Float32Array; ms: number; missing: number };

const TILE = 256;
const URL_OF = (z: number, x: number, y: number) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
// decoded tiles by "z/x/y", the last used last
const tiles = new Map<string, Float32Array | null>();
const CAP = 160;

async function tile(z: number, x: number, y: number): Promise<Float32Array | null> {
  const k = `${z}/${x}/${y}`;
  const have = tiles.get(k);
  if (have !== undefined) {
    tiles.delete(k);
    tiles.set(k, have);
    return have;
  }
  let out: Float32Array | null = null;
  try {
    const n = 2 ** z;
    const r = await fetch(URL_OF(z, ((x % n) + n) % n, y));
    if (r.ok) {
      const bmp = await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      const cv = new OffscreenCanvas(TILE, TILE);
      const cx = cv.getContext("2d", { willReadFrequently: true })!;
      cx.drawImage(bmp, 0, 0, TILE, TILE);
      bmp.close();
      const px = cx.getImageData(0, 0, TILE, TILE).data;
      out = new Float32Array(TILE * TILE);
      for (let i = 0; i < out.length; i++) out[i] = px[4 * i]! * 256 + px[4 * i + 1]! + px[4 * i + 2]! / 256 - 32768;
    }
  } catch {
    out = null;
  }
  tiles.set(k, out);
  for (const old of tiles.keys()) {
    if (tiles.size <= CAP) break;
    tiles.delete(old);
  }
  return out;
}

const DR = [-1, -1, -1, 0, 0, 1, 1, 1];
const DC = [-1, 0, 1, -1, 1, -1, 0, 1];
const DD = [Math.SQRT2, 1, Math.SQRT2, 1, 1, Math.SQRT2, 1, Math.SQRT2];

/** log10 of the pixels draining through each pixel (itself counted), for elevation z on a w by h grid. */
function flowOf(z: Float32Array, w: number, h: number): Float32Array {
  const N = w * h;
  const fill = new Float64Array(N);
  const seen = new Uint8Array(N);
  // a binary heap of pixels by filled height
  const hk = new Float64Array(N), hi = new Int32Array(N);
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
  const EPS = 1e-4;
  const order = new Int32Array(N);
  let no = 0;
  while (hn > 0) {
    const i = pop();
    order[no++] = i;
    const v = fill[i]! + EPS;
    const r = (i / w) | 0, c = i - r * w;
    for (let d = 0; d < 8; d++) {
      const rr = r + DR[d]!, cc = c + DC[d]!;
      if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
      const j = rr * w + cc;
      if (seen[j]) continue;
      seen[j] = 1;
      fill[j] = z[j]! > v ? z[j]! : v;
      push(fill[j]!, j);
    }
  }
  // every pixel counts itself, then hands its count to its steepest lower neighbor; highest pixels first
  const acc = new Float32Array(N).fill(1);
  for (let k = N - 1; k >= 0; k--) {
    const i = order[k]!;
    const r = (i / w) | 0, c = i - r * w;
    let best = 0, to = -1;
    for (let d = 0; d < 8; d++) {
      const rr = r + DR[d]!, cc = c + DC[d]!;
      if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
      const j = rr * w + cc;
      const s = (fill[i]! - fill[j]!) / DD[d]!;
      if (s > best) {
        best = s;
        to = j;
      }
    }
    if (to >= 0) acc[to] = acc[to]! + acc[i]!;
  }
  // Level water (a lake, a wide river) is flat in the elevation, so the path across it is made up by the fill and
  // shows as straight lines. Every connected level surface takes one count instead: the most that leaves it.
  const TOL = 0.006, SMALLEST = 60;
  const done = new Uint8Array(N);
  const queue = new Int32Array(N);
  for (let r = 1; r < h - 1; r++)
    for (let c = 1; c < w - 1; c++) {
      const i = r * w + c;
      if (done[i]) continue;
      const z0 = z[i]!;
      let level = true;
      for (let d = 0; d < 8 && level; d++) if (Math.abs(z[i + DR[d]! * w + DC[d]!]! - z0) > TOL) level = false;
      if (!level) continue;
      let n = 0, top = 0;
      queue[n++] = i;
      done[i] = 1;
      for (let q = 0; q < n; q++) {
        const k = queue[q]!;
        if (acc[k]! > top) top = acc[k]!;
        const kr = (k / w) | 0, kc = k - kr * w;
        for (let d = 0; d < 8; d++) {
          const rr = kr + DR[d]!, cc = kc + DC[d]!;
          if (rr < 0 || cc < 0 || rr >= h || cc >= w) continue;
          const j = rr * w + cc;
          if (done[j] || Math.abs(z[j]! - z0) > TOL) continue;
          done[j] = 1;
          queue[n++] = j;
        }
      }
      if (n >= SMALLEST) for (let q = 0; q < n; q++) acc[queue[q]!] = top;
    }
  for (let i = 0; i < N; i++) acc[i] = Math.log10(acc[i]!);
  return acc;
}

self.onmessage = async (e: MessageEvent<GroundAsk>) => {
  const { id, z, x0, x1, y0, y1 } = e.data;
  const wantFlow = e.data.flow;
  const t0 = performance.now();
  const nx = x1 - x0 + 1, ny = y1 - y0 + 1;
  const w = nx * TILE, h = ny * TILE;
  const elev = new Float32Array(w * h).fill(NaN);
  const jobs: Promise<void>[] = [];
  let missing = 0;
  for (let ty = 0; ty < ny; ty++)
    for (let tx = 0; tx < nx; tx++)
      jobs.push(
        tile(z, x0 + tx, y0 + ty).then((t) => {
          if (!t) {
            missing++;
            return;
          }
          for (let r = 0; r < TILE; r++) elev.set(t.subarray(r * TILE, (r + 1) * TILE), (ty * TILE + r) * w + tx * TILE);
        }),
      );
  await Promise.all(jobs);
  // a tile that did not come is treated as high ground, so nothing drains into it
  let top = -Infinity;
  for (let i = 0; i < elev.length; i++) if (elev[i]! > top) top = elev[i]!;
  if (!(top > -Infinity)) top = 0;
  for (let i = 0; i < elev.length; i++) if (!(elev[i]! === elev[i]!)) elev[i] = top;
  // the flow is worked out only when the page mixes it in (ground.ts, FLOW)
  const flow = wantFlow ? flowOf(elev, w, h) : new Float32Array(0);
  const out: GroundAnswer = { id, z, x0, y0, w, h, elev, flow, ms: performance.now() - t0, missing };
  (self as unknown as Worker).postMessage(out, [elev.buffer, flow.buffer]);
};
