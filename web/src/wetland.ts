import type { MinimalTileData, RasterModule } from "@developmentseed/deck.gl-raster";
import { ZarrLayer, ZarrSource } from "@developmentseed/deck.gl-zarr";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import type * as zarr from "zarrita";

import { CHUNK, MIN_ZOOM, openMosaic, RES, X0, Y0, YEAR_ORIGIN } from "./aef";
import { DemSource } from "./dem-source";
import type { FillNet } from "./fill";
import { N_EMB, waterOf, type WetHead, wetOf, WET_STEPS } from "./head";
import { Pool } from "./pool";
import { BLUGRN, glslRamp } from "./ramp";

/** One chunk on the GPU: a year's wetland and water bytes side by side (rg8uint). */
type TileData = MinimalTileData & { key: string; tex: Texture };

/**
 * A ZarrLayer that says when deck has taken it off the map. deck applies a new layer list at its next frame, not at
 * setProps: a key taken off and put back before that frame keeps its old layer and tiles and is never finalized. So a
 * layer's textures are freed here, never on a timer.
 */
class WetLayer extends ZarrLayer<TileData, "int8"> {
  static override layerName = "WetLayer";
  override finalizeState(context: Parameters<ZarrLayer<TileData, "int8">["finalizeState"]>[0]) {
    super.finalizeState(context);
    (this.props as unknown as { onFinalize?: () => void }).onFinalize?.();
  }
}

/** One chunk's planes on the page: [wetland n, water n] (head.ts). */
type Planes = { planes: Uint8Array; w: number; h: number; rough?: boolean };

/** What the shader draws with: the ramp's ends in wetland probability, the picture's strength, water on or off. */
export type Look = { lo: number; hi: number; alpha: number; water: boolean };

/**
 * A year per pixel, the score along the whole ramp. Vegetated wetland runs over the ramp's pale-to-green half by how
 * sure it is (lo at the pale end, hi at the green middle); open water runs over the teal-to-blue part the same way;
 * a pixel that is some of each sits between. Under lo nothing is drawn, so land has no color.
 * The probabilities are mixed between the four nearest pixels, so the picture stays smooth zoomed past 10 m.
 */
const WetColor = {
  name: "wet",
  fs: `uniform wetUniforms {
  float lo;
  float hi;
  float alpha;
  float water;
} wet;`,
  inject: {
    "fs:#decl": `precision highp usampler2D;
uniform highp usampler2D wetTex;
${glslRamp("blugrn", BLUGRN)}
// wetland and water probability at a pixel; x < 0 where there is no embedding
vec2 probAt(ivec2 p, ivec2 size) {
  uvec2 c = texelFetch(wetTex, clamp(p, ivec2(0), size - 1), 0).rg;
  if (c.r == 0u) return vec2(-1.0, 0.0);
  return vec2(float(c.r - 1u) / ${WET_STEPS.toFixed(1)}, float(c.g) / 255.0);
}`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(wetTex, 0);
      vec2 at = geometry.uv * vec2(size) - 0.5;
      ivec2 p0 = ivec2(floor(at));
      vec2 f = at - vec2(p0);
      vec2 near = probAt(ivec2(geometry.uv * vec2(size)), size);
      if (near.x < 0.0) discard;
      vec2 a = probAt(p0, size), b = probAt(p0 + ivec2(1, 0), size);
      vec2 c = probAt(p0 + ivec2(0, 1), size), d = probAt(p0 + ivec2(1, 1), size);
      vec2 pr = near;
      if (a.x >= 0.0 && b.x >= 0.0 && c.x >= 0.0 && d.x >= 0.0) pr = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      float pw = wet.water > 0.5 ? pr.y : 0.0;
      if (wet.water < 0.5 && pr.y >= pr.x && pr.y >= max(1.0 - pr.x - pr.y, 0.0)) discard;
      float total = pw + pr.x;
      if (total <= wet.lo) discard;
      // The score runs along the whole ramp. Wetland takes the pale-to-green half, pale where the score is low and
      // green where it is sure; open water takes the teal-to-blue part the same way. A bank between water and dry land
      // stays on the water side as it fades, so it never passes through wetland's green.
      float sure = clamp((total - wet.lo) / max(wet.hi - wet.lo, 1e-4), 0.0, 1.0);
      float t = mix(0.5 * sure, 0.65 + 0.35 * sure, pw / max(total, 1e-4));
      color = vec4(blugrn(t), wet.alpha * smoothstep(wet.lo, wet.lo + 0.08, total));
    `,
  },
  uniformTypes: { lo: "f32", hi: "f32", alpha: "f32", water: "f32" },
  getUniforms: (p: Record<string, unknown>) => p,
} as unknown as ShaderModule<{ wetTex: Texture; lo: number; hi: number; alpha: number; water: number }>;

/**
 * fill: the network that rebuilds the bands after the first 32, so each chunk draws first from the front of its bytes
 * and sharpens to the exact picture once the rest is in (zread.ts); onSharpen is called when chunks on the map have
 * sharpened (the page repaints).
 */
export async function createWetland(head: WetHead, beforeId: () => string | undefined, fill: FillNet | null = null, onSharpen: () => void = () => {}) {
  const { arr, attrs } = await openMosaic();
  const [, , H, W] = arr.shape as number[];
  const opened = await ZarrSource.open(arr, { geoZarrMetadata: attrs });
  if (!opened.is("int8")) throw new Error("Expected int8 embeddings");
  const source: ZarrSource<"int8"> = opened;
  // a head that reads elevation is given the ground under each chunk
  const dems = head.n_in > N_EMB ? new DemSource() : null;
  const pool = new Pool(head, fill, dems);

  // the planes by "year:row-col", the last used last; 128 KB each, at most CAP_BYTES
  const planes = new Map<string, Planes>();
  const CAP_BYTES = 384 * 1024 ** 2;
  let planeBytes = 0;
  const keyOf = (y: number, r: number, c: number) => `${y}:${r}-${c}`;
  const keep = (k: string, p: Planes) => {
    const old = planes.get(k);
    if (old) {
      planes.delete(k);
      planeBytes -= old.planes.byteLength;
    }
    planes.set(k, p);
    planeBytes += p.planes.byteLength;
    for (const [ok, v] of planes) {
      if (planeBytes <= CAP_BYTES) break;
      planes.delete(ok);
      planeBytes -= v.planes.byteLength;
    }
  };
  // reads in flight, shared by everyone who wants the same chunk-year; stopped once nobody does
  const inflight = new Map<string, { p: Promise<Planes>; ac: AbortController; wants: number }>();
  const loaded = new Set<string>();
  // each year's textures: deck.gl does not unload the tiles of a layer taken off the map, so its finalize frees them
  const texOf = new Map<number, Set<Texture>>();
  // the textures of each chunk, to write the exact planes into when they come
  const texAt = new Map<string, Set<Texture>>();
  // chunks drawn rough whose exact planes are still to come: stopped once no tile shows them
  const sharpening = new Map<string, AbortController>();
  const untex = (k: string, t: Texture) => {
    const s = texAt.get(k);
    if (!s) return;
    s.delete(t);
    if (!s.size) {
      texAt.delete(k);
      sharpening.get(k)?.abort();
    }
  };
  /** A chunk's planes as its texture holds them: wetland and water side by side, one texel each. */
  const texData = (p: Planes) => {
    const n = p.w * p.h;
    const data = new Uint8Array(2 * n);
    for (let i = 0; i < n; i++) {
      data[2 * i] = p.planes[i]!;
      data[2 * i + 1] = p.planes[n + i]!;
    }
    return data;
  };
  let sharpTimer = 0;
  const sharpened = () => {
    if (sharpTimer) return;
    sharpTimer = window.setTimeout(() => {
      sharpTimer = 0;
      onSharpen();
    }, 100);
  };
  function freeTextures(year: number) {
    const set = texOf.get(year);
    texOf.delete(year);
    for (const [k, s] of [...texAt]) if (k.startsWith(`${year}:`)) for (const t of [...s]) if (set?.has(t)) untex(k, t);
    set?.forEach((t) => t.destroy());
    for (const k of [...loaded]) if (k.startsWith(`${year}:`)) loaded.delete(k);
  }

  /** A chunk's planes for a year, read and scored: the rough ones answer first where the chunk reads in two parts. */
  async function planesFor(year: number, py0: number, px0: number, h: number, w: number, signal?: AbortSignal): Promise<Planes> {
    const k = keyOf(year, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
    const have = planes.get(k);
    if (have) {
      keep(k, have);
      return have;
    }
    let f = inflight.get(k);
    if (!f || f.ac.signal.aborted) {
      const ac = new AbortController();
      let answer!: (p: Planes) => void;
      let fail!: (e: unknown) => void;
      const p = new Promise<Planes>((ok, no) => {
        answer = ok;
        fail = no;
      });
      let answered = false;
      const rough = (pl: Uint8Array) => {
        if (answered) return;
        answered = true;
        const out = { planes: pl, w, h, rough: true };
        keep(k, out);
        sharpening.set(k, ac);
        answer(out);
      };
      pool.read(year - YEAR_ORIGIN, py0, px0, h, w, ac.signal, rough).then(
        (pl) => {
          const out = { planes: pl, w, h };
          keep(k, out);
          if (sharpening.get(k) === ac) sharpening.delete(k);
          if (!answered) {
            answered = true;
            answer(out);
            return;
          }
          const data = texData(out);
          for (const t of texAt.get(k) ?? []) t.writeData(data);
          sharpened();
        },
        (e) => {
          if (sharpening.get(k) === ac) sharpening.delete(k);
          // a rough chunk stopped before it sharpened: read afresh when next asked
          const had = planes.get(k);
          if (answered && had?.rough) {
            planes.delete(k);
            planeBytes -= had.planes.byteLength;
          } else if (!answered) fail(e);
        },
      );
      const g = { p, ac, wants: 0 };
      inflight.set(k, g);
      const done = () => inflight.get(k) === g && inflight.delete(k);
      p.then(done, done);
      f = g;
    }
    const g = f;
    g.wants++;
    const drop = () => {
      if (--g.wants === 0) g.ac.abort();
    };
    signal?.addEventListener("abort", drop, { once: true });
    try {
      return await g.p;
    } finally {
      signal?.removeEventListener("abort", drop);
      if (!signal?.aborted) g.wants--;
    }
  }

  const chunkRange = (bbox: [number, number, number, number]) => {
    const [w, s, e, n] = bbox;
    return {
      c0: Math.max(0, Math.floor((w - X0) / RES / CHUNK)),
      c1: Math.min(Math.ceil(W / CHUNK) - 1, Math.floor((e - X0) / RES / CHUNK)),
      r0: Math.max(0, Math.floor((Y0 - n) / RES / CHUNK)),
      r1: Math.min(Math.ceil(H / CHUNK) - 1, Math.floor((Y0 - s) / RES / CHUNK)),
    };
  };

  /** The year's layer. Only the year on the map is read; another year is read when the map steps to it. */
  function layer(opts: { year: number; visible: boolean; look: Look; onTile?: () => void; onViewportLoad?: () => void }) {
    const { year, look } = opts;
    return new WetLayer({
      id: `wet-${year}`,
      source,
      selection: { time: year - YEAR_ORIGIN, band: null },
      getTileData: async (_a, o) => {
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
        const { width, height } = o;
        const p = await planesFor(year, py0, px0, height, width, o.signal);
        const key = keyOf(year, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
        loaded.add(key);
        const data = texData(p);
        const tex = o.device.createTexture({
          format: "rg8uint",
          width,
          height,
          mipLevels: 1,
          data,
          sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
        });
        let set = texOf.get(year);
        if (!set) texOf.set(year, (set = new Set()));
        set.add(tex);
        let at = texAt.get(key);
        if (!at) texAt.set(key, (at = new Set()));
        at.add(tex);
        // exact planes may have come between the rough answer and this texture
        const now = planes.get(key);
        if (p.rough && now && !now.rough) tex.writeData(texData(now));
        return { key, tex, width, height, byteLength: data.byteLength };
      },
      renderTile: (data: TileData) => ({
        renderPipeline: [
          { module: WetColor, props: { wetTex: data.tex, lo: look.lo, hi: look.hi, alpha: look.alpha, water: look.water ? 1 : 0 } },
        ] as RasterModule[],
      }),
      // a new look redraws the tiles in hand, no read
      updateTriggers: { renderTile: `${look.lo}|${look.hi}|${look.alpha}|${look.water}` },
      visible: opts.visible,
      // the layer's own floor is a whole zoom (a fractional one stops its tiles up to the next whole zoom); the page
      // hides the layer between that and MIN_ZOOM
      minZoom: Math.floor(MIN_ZOOM),
      // the view's chunks all asked at once: the workers read them a few at a time, every front before any rest
      maxRequests: 64,
      maxCacheSize: 1200,
      onTileLoad: () => opts.onTile && opts.onTile(),
      onTileUnload: (tile: { content?: TileData | null }) => {
        const t = tile.content;
        if (!t) return;
        loaded.delete(t.key);
        texOf.get(year)?.delete(t.tex);
        untex(t.key, t.tex);
        t.tex?.destroy();
      },
      onViewportLoad: () => opts.onViewportLoad && opts.onViewportLoad(),
      // read by WetLayer.finalizeState (not a ZarrLayer prop, so spread past the type check)
      ...({ onFinalize: () => freeTextures(year) } as object),
      // @ts-expect-error beforeId is read by the MapLibre overlay (the layer draws under the labels), not typed on the layer
      beforeId: beforeId(),
    });
  }

  /** The head's probabilities at a point for a year, from the planes in hand; null where none. */
  function valueAtPoint(lon: number, lat: number, year: number): { wetland: number; water: number; upland: number; rough: boolean } | null {
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    const r = Math.floor(py / CHUNK), c = Math.floor(px / CHUNK);
    const p = planes.get(keyOf(year, r, c));
    if (!p) return null;
    const ix = px - c * CHUNK, iy = py - r * CHUNK;
    if (ix < 0 || iy < 0 || ix >= p.w || iy >= p.h) return null;
    const i = iy * p.w + ix, n = p.w * p.h;
    const wet = wetOf(p.planes[i]!);
    if (!(wet >= 0)) return null;
    const water = waterOf(p.planes[n + i]!);
    return { wetland: wet, water, upland: Math.max(0, 1 - wet - water), rough: !!p.rough };
  }

  /** How many of the chunks under a box are on the GPU for a year, and how many of those are still rough. */
  function loadedIn(bbox: [number, number, number, number], year: number) {
    const { c0, c1, r0, r1 } = chunkRange(bbox);
    let n = 0, rough = 0;
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        const k = keyOf(year, r, c);
        if (!loaded.has(k)) continue;
        n++;
        if (planes.get(k)?.rough) rough++;
      }
    return { loaded: n, rough, wanted: Math.max(0, (r1 - r0 + 1) * (c1 - c0 + 1)) };
  }

  /**
   * The wet score (wetland and open water probabilities added) under the centers of a grid of longitudes by
   * latitudes, from the planes in hand; 0 where a chunk is not scored yet or has no embedding.
   */
  function sampleWet(lons: Float64Array, lats: Float64Array, year: number, out: Float32Array) {
    const w = lons.length, h = lats.length;
    const pxs = new Int32Array(w);
    for (let c = 0; c < w; c++) pxs[c] = Math.floor((lons[c]! - X0) / RES);
    let lastR = -1, lastC = -1;
    let p: Planes | undefined;
    for (let r = 0; r < h; r++) {
      const py = Math.floor((Y0 - lats[r]!) / RES);
      const cr = Math.floor(py / CHUNK), iy = py - cr * CHUNK;
      for (let c = 0; c < w; c++) {
        const px = pxs[c]!;
        const cc = Math.floor(px / CHUNK);
        if (cr !== lastR || cc !== lastC) {
          p = planes.get(keyOf(year, cr, cc));
          lastR = cr;
          lastC = cc;
        }
        let v = 0;
        if (p) {
          const ix = px - cc * CHUNK;
          if (ix < p.w && iy < p.h) {
            const i = iy * p.w + ix;
            const a = p.planes[i]!;
            if (a > 0) v = (a - 1) / WET_STEPS + p.planes[p.w * p.h + i]! / 255;
          }
        }
        out[r * w + c] = v;
      }
    }
  }

  return { layer, valueAtPoint, loadedIn, pool, sampleWet, dems };
}
