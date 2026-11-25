import type * as maplibregl from "maplibre-gl";

import type { GroundAnswer, GroundAsk } from "./ground.worker";

/**
 * One value for every pixel of the view, shaded straight through wetland and water:
 *
 *   value = (how low the ground sits in the view + the head's wet score) / 2
 *
 * With FLOW on, how much ground drains through the pixel is a third part and the sum is over 3. It is off: the flow
 * knows only the ground in the view, draws one-pixel threads, and on flat ground follows paths the fill makes up.
 *
 * Each part runs from 0 to 1. Low: the pixel's elevation between the view's 2nd and 98th percentile, turned over. Flow:
 * log10 of the pixels draining through it over the view's 99.9th percentile. Wet: the head's wetland and open water
 * probabilities added. The first two are relative to what is on screen, so the picture rescales as the map moves.
 */
export const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
/** The ramp as drawn: yellow where the value is 0 (high, sheds water, dry), purple where it is 1. */
export const GROUND_RAMP = [...VIRIDIS].reverse();

/** Mix how much ground drains through each pixel into the value, and work it out in the worker. */
export const FLOW: boolean = false;

const TILE = 256;
// Terrarium zoom 14 is about 7 to 8 m a pixel in the lower 48 states, as fine as the 10 m elevation under it
const MAX_Z = 14;
const SRC = "ground";

/** Wet score (wetland + open water, 0 where nothing is scored yet) at the centers of a grid of longitudes by latitudes. */
export type SampleWet = (lons: Float64Array, lats: Float64Array, year: number, out: Float32Array) => void;
export type GroundParts = { low: number; flow: number; wet: number; value: number; elev: number };

const lut = (() => {
  const rgb = GROUND_RAMP.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (rgb.length - 1);
    const a = Math.min(Math.floor(x), rgb.length - 2), f = x - a;
    for (let c = 0; c < 3; c++) out[3 * i + c] = Math.round(rgb[a]![c]! * (1 - f) + rgb[a + 1]![c]! * f);
  }
  return out;
})();

const lonOf = (x: number, z: number) => (x / 2 ** z) * 360 - 180;
const latOf = (y: number, z: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 2 ** z))) * 180) / Math.PI;
const xOf = (lon: number, z: number) => ((lon + 180) / 360) * 2 ** z;
const yOf = (lat: number, z: number) => ((1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2) * 2 ** z;

export function createGround(map: maplibregl.Map, sampleWet: SampleWet, before: () => string | undefined, onState: () => void = () => {}) {
  const worker = new Worker(new URL("./ground.worker.ts", import.meta.url), { type: "module" });
  let asked = 0;
  let askedKey = "";
  let scene: GroundAnswer | null = null;
  let year = 0;
  let on = true;
  let opacity = 1;
  // pixels whose wet score is under the cutoff are drawn at the strength `dry` (0 clear, 1 full); null draws every pixel in full
  let mask: number | null = null;
  let dry = 0;
  // the parts as last drawn, for the readout
  let parts: { low: Float32Array; flo: Float32Array; wet: Float32Array } | null = null;
  let wetBuf: Float32Array | null = null;
  const canvas = document.createElement("canvas");
  let lastUrl = "";
  const state = { busy: false, ms: 0, missing: 0, lo: 0, hi: 0, cells: 0 };

  worker.onmessage = (e: MessageEvent<GroundAnswer>) => {
    if (e.data.id !== asked) return;
    scene = e.data;
    state.busy = false;
    state.ms = e.data.ms;
    state.missing = e.data.missing;
    state.cells = e.data.w * e.data.h;
    compose();
    onState();
  };

  /** The tiles under the view; a new scene is asked for when they change. */
  function ask() {
    if (!on) return;
    const b = map.getBounds();
    // one tile level past the map's whole zoom: a grid pixel is 1 to 2 screen pixels until the 10 m data runs out
    const z = Math.max(0, Math.min(MAX_Z, Math.floor(map.getZoom()) + 1));
    const n = 2 ** z;
    const x0 = Math.floor(xOf(b.getWest(), z)), x1 = Math.floor(xOf(b.getEast(), z));
    const y0 = Math.max(0, Math.floor(yOf(b.getNorth(), z))), y1 = Math.min(n - 1, Math.floor(yOf(b.getSouth(), z)));
    const key = `${z}/${x0}-${x1}/${y0}-${y1}`;
    if (key === askedKey) return compose();
    askedKey = key;
    state.busy = true;
    const msg: GroundAsk = { id: ++asked, z, x0, x1, y0, y1, flow: FLOW };
    worker.postMessage(msg);
    onState();
  }

  /** The picture from the scene in hand, the view's own range of ground, and the scores read so far. */
  function compose() {
    const s = scene;
    if (!s || !on) return;
    const { w, h, z, x0, y0, elev, flow } = s;
    const N = w * h;
    const b = map.getBounds();
    // the part of the scene on screen: the ranges come from it alone
    const c0 = Math.max(0, Math.floor((xOf(b.getWest(), z) - x0) * TILE)), c1 = Math.min(w, Math.ceil((xOf(b.getEast(), z) - x0) * TILE));
    const r0 = Math.max(0, Math.floor((yOf(b.getNorth(), z) - y0) * TILE)), r1 = Math.min(h, Math.ceil((yOf(b.getSouth(), z) - y0) * TILE));
    if (c1 <= c0 || r1 <= r0) return;
    const stepR = Math.max(1, Math.floor((r1 - r0) / 300)), stepC = Math.max(1, Math.floor((c1 - c0) / 300));
    const es: number[] = [], fs: number[] = [];
    for (let r = r0; r < r1; r += stepR)
      for (let c = c0; c < c1; c += stepC) {
        es.push(elev[r * w + c]!);
        if (FLOW) fs.push(flow[r * w + c]!);
      }
    es.sort((a, b2) => a - b2);
    fs.sort((a, b2) => a - b2);
    const q = (a: number[], p: number) => a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))]!;
    const lo = q(es, 0.02), hi = q(es, 0.98);
    const span = Math.max(hi - lo, 1e-3);
    const ftop = FLOW ? Math.max(q(fs, 0.999), 1e-3) : 1;
    state.lo = lo;
    state.hi = hi;

    const lons = new Float64Array(w), lats = new Float64Array(h);
    for (let c = 0; c < w; c++) lons[c] = lonOf(x0 + (c + 0.5) / TILE, z);
    for (let r = 0; r < h; r++) lats[r] = latOf(y0 + (r + 0.5) / TILE, z);
    if (!wetBuf || wetBuf.length !== N) wetBuf = new Float32Array(N);
    sampleWet(lons, lats, year, wetBuf);
    if (!parts || parts.low.length !== N) parts = { low: new Float32Array(N), flo: new Float32Array(N), wet: new Float32Array(N) };
    parts.wet.set(wetBuf);

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const cx = canvas.getContext("2d")!;
    const img = cx.createImageData(w, h);
    const faint = Math.round(dry * 255);
    const px = img.data;
    for (let i = 0; i < N; i++) {
      const low = 1 - Math.min(1, Math.max(0, (elev[i]! - lo) / span));
      const flo = FLOW ? Math.min(1, Math.max(0, flow[i]! / ftop)) : 0;
      const wt = Math.min(1, wetBuf[i]!);
      parts.low[i] = low;
      parts.flo[i] = flo;
      const k = Math.round((FLOW ? (low + flo + wt) / 3 : (low + wt) / 2) * 255) * 3;
      px[4 * i] = lut[k]!;
      px[4 * i + 1] = lut[k + 1]!;
      px[4 * i + 2] = lut[k + 2]!;
      px[4 * i + 3] = mask === null || wt >= mask ? 255 : faint;
    }
    cx.putImageData(img, 0, 0);
    const west = lonOf(x0, z), east = lonOf(x0 + w / TILE, z), north = latOf(y0, z), south = latOf(y0 + h / TILE, z);
    const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [[west, north], [east, north], [east, south], [west, south]];
    canvas.toBlob((blob) => {
      if (!blob || scene !== s) return;
      const url = URL.createObjectURL(blob);
      const src = map.getSource(SRC) as maplibregl.ImageSource | undefined;
      if (src) src.updateImage({ url, coordinates });
      else {
        map.addSource(SRC, { type: "image", url, coordinates });
        map.addLayer(
          { id: SRC, type: "raster", source: SRC, layout: { visibility: on ? "visible" : "none" },
            paint: { "raster-fade-duration": 0, "raster-resampling": "linear", "raster-opacity": opacity } },
          before(),
        );
      }
      if (lastUrl) {
        const old = lastUrl;
        setTimeout(() => URL.revokeObjectURL(old), 4000);
      }
      lastUrl = url;
    }, "image/png");
  }

  let timer = 0;
  /** Scores have come in: draw again soon, once for many. */
  function refresh() {
    if (timer || !on) return;
    timer = window.setTimeout(() => {
      timer = 0;
      compose();
    }, 250);
  }

  function valueAt(lon: number, lat: number): GroundParts | null {
    const s = scene;
    if (!s || !parts) return null;
    const c = Math.floor((xOf(lon, s.z) - s.x0) * TILE), r = Math.floor((yOf(lat, s.z) - s.y0) * TILE);
    if (c < 0 || r < 0 || c >= s.w || r >= s.h) return null;
    const i = r * s.w + c;
    const low = parts.low[i]!, flow = parts.flo[i]!, wet = Math.min(1, parts.wet[i]!);
    return { low, flow, wet, value: FLOW ? (low + flow + wet) / 3 : (low + wet) / 2, elev: s.elev[i]! };
  }

  function set(o: { on?: boolean; year?: number; opacity?: number; mask?: number | null; dry?: number }) {
    const wasOn = on, wasYear = year, wasMask = mask, wasDry = dry;
    if (o.dry !== undefined) dry = o.dry;
    if (o.mask !== undefined) mask = o.mask;
    if (o.on !== undefined) on = o.on;
    if (o.year !== undefined) year = o.year;
    if (o.opacity !== undefined) opacity = o.opacity;
    if (map.getLayer(SRC)) {
      map.setLayoutProperty(SRC, "visibility", on ? "visible" : "none");
      map.setPaintProperty(SRC, "raster-opacity", opacity);
    }
    if (on && (!wasOn || year !== wasYear)) ask();
    else if (on && (mask !== wasMask || dry !== wasDry)) compose();
  }

  return { ask, refresh, set, valueAt, state };
}
