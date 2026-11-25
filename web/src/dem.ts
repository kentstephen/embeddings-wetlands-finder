import { DecoderPool, GeoTIFF, PerOriginSemaphore } from "@developmentseed/geotiff";
import GeoTiffWorker from "@developmentseed/geotiff/pool/worker?worker";

/**
 * Ground elevation from USGS 3DEP's seamless 1/3 arc-second model (about 10 m, the lower 48 states and more): one
 * cloud-optimized GeoTIFF per degree square on the USGS bucket, named for its north and west edges, read here by
 * window with Development Seed's GeoTIFF reader. Every file sits on one grid of 10,800 pixels a degree and carries
 * six pixels past each edge; a window is put together from the squares it crosses.
 *
 * Read on the page, not in a worker (the reader parses the file's metadata with the page's XML parser); its tiles
 * are unpacked in the reader's own workers.
 */
const ROOT = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current";
/** Pixels in a degree. */
export const PER_DEG = 10800;

/** A window of the grid: column gc0 starts at longitude -180 + gc0 / PER_DEG, row gr0 at latitude 90 - gr0 / PER_DEG. NaN where there is no ground. */
export type Dem = { z: Float32Array; gc0: number; gr0: number; w: number; h: number };

let pool: DecoderPool | null = null;
const limiter = new PerOriginSemaphore({ maxRequests: 6 });
/** Why the last square that would not open did not, other than not being there. */
export const demState = { error: "" };
const files = new Map<string, Promise<{ g: GeoTIFF; c0: number; r0: number } | null>>();
/** The square whose north edge is latitude n and west edge longitude -w; null where the model has none (sea, abroad). */
function open(n: number, w: number) {
  const name = `${n >= 0 ? "n" : "s"}${String(Math.abs(n)).padStart(2, "0")}${w >= 0 ? "w" : "e"}${String(Math.abs(w)).padStart(3, "0")}`;
  let f = files.get(name);
  if (!f) {
    const url = `${ROOT}/${name}/USGS_13_${name}.tif`;
    f = GeoTIFF.fromUrl(url, { concurrencyLimiter: limiter }).then(
      (g) => {
        const [x, y] = g.xy(0, 0, "ul");
        return { g, c0: Math.round((x + 180) * PER_DEG), r0: Math.round((90 - y) * PER_DEG) };
      },
      async (e) => {
        // no such square (sea, abroad) is an answer; anything else is said
        const there = await fetch(url, { method: "HEAD" }).then((r) => r.ok, () => true);
        if (there) {
          demState.error = `${name}: ${(e as Error)?.message || e}`;
          files.delete(name);
        }
        return null;
      },
    );
    files.set(name, f);
  }
  return f;
}

/** The grid pixels that cover a box of longitude and latitude, with two more on every side. */
export async function readDem(west: number, south: number, east: number, north: number, signal?: AbortSignal): Promise<Dem> {
  const gc0 = Math.floor((west + 180) * PER_DEG) - 2, gc1 = Math.ceil((east + 180) * PER_DEG) + 2;
  const gr0 = Math.floor((90 - north) * PER_DEG) - 2, gr1 = Math.ceil((90 - south) * PER_DEG) + 2;
  const w = gc1 - gc0, h = gr1 - gr0;
  const z = new Float32Array(w * h).fill(NaN);
  const jobs: Promise<void>[] = [];
  pool ??= new DecoderPool({ size: Math.min(4, navigator.hardwareConcurrency || 2), createWorker: () => new GeoTiffWorker() });
  const decoders = pool;
  for (let dj = Math.floor(gr0 / PER_DEG); dj <= Math.floor((gr1 - 1) / PER_DEG); dj++)
    for (let di = Math.floor(gc0 / PER_DEG); di <= Math.floor((gc1 - 1) / PER_DEG); di++) {
      // the part of the window inside this degree square
      const a0 = Math.max(gc0, di * PER_DEG), a1 = Math.min(gc1, (di + 1) * PER_DEG);
      const b0 = Math.max(gr0, dj * PER_DEG), b1 = Math.min(gr1, (dj + 1) * PER_DEG);
      jobs.push(
        open(90 - dj, 180 - di).then(async (f) => {
          if (!f) return;
          const colOff = a0 - f.c0, rowOff = b0 - f.r0;
          if (colOff < 0 || rowOff < 0 || colOff + (a1 - a0) > f.g.width || rowOff + (b1 - b0) > f.g.height) return;
          const part = await f.g.read({ window: { colOff, rowOff, width: a1 - a0, height: b1 - b0 }, pool: decoders, signal });
          const src = (part.layout === "band-separate" ? part.bands[0]! : part.data) as Float32Array;
          const pw = a1 - a0;
          for (let r = 0; r < b1 - b0; r++)
            for (let c = 0; c < pw; c++) {
              const v = src[r * pw + c]!;
              if (v > -1000) z[(b0 - gr0 + r) * w + (a0 - gc0 + c)] = v;
            }
        }),
      );
    }
  await Promise.all(jobs);
  return { z, gc0, gr0, w, h };
}

/**
 * The elevation between the four nearest pixel centers at each crossing of the given longitudes and latitudes;
 * pixels with no ground are left out of the mix, and NaN where all four have none.
 */
export function sampleDem(d: Dem, lons: Float64Array, lats: Float64Array): Float32Array {
  const W = lons.length, H = lats.length;
  const out = new Float32Array(W * H);
  const c0 = new Int32Array(W), fx = new Float32Array(W);
  for (let c = 0; c < W; c++) {
    const u = (lons[c]! + 180) * PER_DEG - 0.5 - d.gc0;
    c0[c] = Math.floor(u);
    fx[c] = u - c0[c]!;
  }
  for (let r = 0; r < H; r++) {
    const v = (90 - lats[r]!) * PER_DEG - 0.5 - d.gr0;
    const r0 = Math.floor(v), fy = v - r0;
    if (r0 < 0 || r0 + 1 >= d.h) {
      out.fill(NaN, r * W, (r + 1) * W);
      continue;
    }
    const o0 = r0 * d.w, o1 = o0 + d.w;
    for (let c = 0; c < W; c++) {
      const k = c0[c]!;
      if (k < 0 || k + 1 >= d.w) {
        out[r * W + c] = NaN;
        continue;
      }
      const a = d.z[o0 + k]!, b = d.z[o0 + k + 1]!, e = d.z[o1 + k]!, g = d.z[o1 + k + 1]!;
      const f = fx[c]!;
      const s = a + b + e + g;
      if (s === s) out[r * W + c] = (a * (1 - f) + b * f) * (1 - fy) + (e * (1 - f) + g * f) * fy;
      else {
        let sum = 0, wt = 0;
        const add = (x: number, k2: number) => {
          if (x === x) {
            sum += x * k2;
            wt += k2;
          }
        };
        add(a, (1 - f) * (1 - fy));
        add(b, f * (1 - fy));
        add(e, (1 - f) * fy);
        add(g, f * fy);
        out[r * W + c] = wt > 0 ? sum / wt : NaN;
      }
    }
  }
  return out;
}
