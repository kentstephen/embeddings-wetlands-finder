import { cpuTerrain, type Job, prepare, probeJob } from "./chunk-terrain";
import type { Dem } from "./dem";
import { fillCpu, type FillNet, fillProbeGap, packFill, probeInput } from "./fill";
import { type Filler, makeFiller } from "./fill-gpu";
import { N_EMB, packHead, probeChunk, probeGap, scoreCpu, type WetHead } from "./head";
import { makeScorer, type Scorer } from "./head-gpu";
import { makeTerrainGpu } from "./terrain-gpu";
import { front, FRONT_BANDS, type Part } from "./zread";

/**
 * One chunk-year per ask: AlphaEarth's 64 bands read from the mosaic in two parts (zread.ts) and the head run after
 * each (WebGPU, else the CPU): a rough picture from the first 32 bands with the rest rebuilt by the fill network,
 * then the exact one. Only the two byte planes go back to the page (wetland, water: 128 KB per 256 x 256 chunk),
 * never the embedding.
 *
 * A head that reads elevation also needs the chunk's terrain numbers. The page sends the elevation window under the
 * chunk (dem.ts); the numbers are worked out here on the GPU (terrain-gpu.ts), or on the CPU without one, and go
 * straight into the scorer. A chunk with no ground at all gets no score.
 */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; timeIdx: number; ahead?: boolean };
type Msg = { type: "init"; head: WetHead; fill: FillNet | null } | Read | { type: "abort"; id: number }
  | { type: "dem"; id: number; dem?: Dem; error?: string };

let ready: Promise<Scorer> | null = null;
let fillNet: FillNet | null = null;
let filling: Promise<Filler> | null = null;
const running = new Map<number, AbortController>();
let needsGround = false;
/** The terrain numbers of each read in hand, once the page has sent the ground under it: null where there is none. */
const ground = new Map<number, { p: Promise<Float32Array | null>; ok: (t: Promise<Float32Array | null>) => void; no: (e: Error) => void }>();
/** Where each read in hand is in the mosaic. */
const readAt = new Map<number, [number, number]>();
let terrainRun: Promise<(j: Job) => Promise<Float32Array>> | null = null;

/** The terrain numbers on the GPU if it gives the CPU's own on made-up ground (to a thousandth), else on the CPU. */
async function terrainer(): Promise<(j: Job) => Promise<Float32Array>> {
  const cpu = async (j: Job) => cpuTerrain(j);
  let run = cpu, mode = "cpu", gap = NaN;
  try {
    const g = await makeTerrainGpu();
    if (g) {
      const j = probeJob();
      const a = await g(j), b = cpuTerrain(j);
      gap = 0;
      for (let i = 0; i < a.length; i++) gap = Math.max(gap, Math.abs(a[i]! - b[i]!));
      if (gap <= 1e-3) {
        run = g;
        mode = "gpu";
      }
    }
  } catch (e) {
    postMessage({ type: "terrainError", message: (e as Error)?.message || String(e) });
  }
  postMessage({ type: "terrainMode", mode, probeGap: gap });
  return run;
}

/** The fill for the front's bands: on the GPU if it gives the export's own output on the probe (to 1 in int8), else the CPU. */
async function filler(f: FillNet): Promise<Filler> {
  const w = packFill(f);
  const cpu: Filler = async (e, n) => fillCpu(f, w, e, n);
  const pr = probeInput(f);
  let run = cpu, mode = "cpu";
  try {
    const g = await makeFiller(f, w, 256 * 256);
    if (g && fillProbeGap(f, await g(pr.e, pr.n), pr.n).worst <= 1) {
      run = g;
      mode = "gpu";
    }
  } catch {
    // no WebGPU here: the CPU
  }
  postMessage({ type: "fill", mode });
  return run;
}

/** The GPU scorer if it gives the CPU's own bytes on the probe (to one step), else the CPU. */
async function scorer(h: WetHead, w: Float32Array): Promise<Scorer> {
  const cpu: Scorer = async (e, n, t) => scoreCpu(h, w, e, n, t);
  const gap = probeGap(h, w);
  let mode = "cpu", gpuGap = NaN;
  let use = cpu;
  try {
    const g = await makeScorer(h, w, 256 * 256);
    if (g) {
      const { e, n, t } = probeChunk(h);
      const a = await g(e, n, t), b = scoreCpu(h, w, e, n, t);
      gpuGap = 0;
      for (let i = 0; i < 2 * n; i++) gpuGap = Math.max(gpuGap, Math.abs(a[i]! - b[i]!));
      if (gpuGap <= 1) {
        use = g;
        mode = "gpu";
      }
    }
  } catch {
    // no WebGPU in this worker: the CPU path
  }
  postMessage({ type: "mode", mode, probeGap: gap, gpuGap });
  return use;
}

async function read(m: Read) {
  const ac = new AbortController();
  running.set(m.id, ac);
  const stop = () => {
    if (ac.signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  try {
    if (m.w !== 256 || m.h !== 256) throw new Error(`a ${m.w} x ${m.h} chunk`);
    const score = await ready!;
    const n = m.w * m.h;
    const stopped = new Promise<never>((_, no) => ac.signal.addEventListener("abort", () => no(Object.assign(new Error("aborted"), { name: "AbortError" })), { once: true }));
    stopped.catch(() => {});
    // the bytes and the ground are read side by side
    const [part, t] = await Promise.all([front(m.timeIdx, m.py0, m.px0, ac.signal, !!m.ahead) as Promise<Part>, needsGround ? Promise.race([ground.get(m.id)!.p, stopped]) : undefined]);
    stop();
    if (needsGround && !t) {
      // no ground here (open sea, outside the elevation model): no score
      const none = new Uint8Array(2 * n);
      postMessage({ id: m.id, planes: none }, { transfer: [none.buffer as ArrayBuffer] });
      return;
    }
    if (!part.full && fillNet) {
      if (!filling) filling = filler(fillNet);
      const fill = await filling;
      const planes = await score(await fill(part.front!, n), n, t);
      stop();
      postMessage({ id: m.id, planes, rough: true }, { transfer: [planes.buffer as ArrayBuffer] });
    }
    const full = part.full ?? (await part.rest(ac.signal));
    stop();
    const planes = await score(full, n, t);
    postMessage({ id: m.id, planes }, { transfer: [planes.buffer as ArrayBuffer] });
  } catch (e) {
    const err = e as { name?: string; message?: string };
    postMessage({ id: m.id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
  } finally {
    running.delete(m.id);
    ground.delete(m.id);
    readAt.delete(m.id);
  }
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  const m = ev.data;
  if (m.type === "init") {
    fillNet = m.fill && m.fill.k === FRONT_BANDS ? m.fill : null;
    needsGround = m.head.n_in > N_EMB;
    ready = scorer(m.head, packHead(m.head));
    if (needsGround) terrainRun = terrainer();
  } else if (m.type === "abort") running.get(m.id)?.abort();
  else if (m.type === "dem") {
    const g = ground.get(m.id);
    if (!g) return;
    if (m.error || !m.dem) g.no(new Error(`elevation: ${m.error || "nothing came"}`));
    else {
      const job = prepare(readAt.get(m.id)![0], readAt.get(m.id)![1], 256, 256, m.dem);
      g.ok(job ? terrainRun!.then((run) => run(job)) : Promise.resolve(null));
    }
  } else {
    if (needsGround) {
      let ok!: (t: Promise<Float32Array | null>) => void, no!: (e: Error) => void;
      const p = new Promise<Float32Array | null>((a, b) => {
        ok = a;
        no = b;
      });
      p.catch(() => {});
      ground.set(m.id, { p, ok, no });
      readAt.set(m.id, [m.px0, m.py0]);
    }
    void read(m);
  }
};
