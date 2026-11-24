import { fillCpu, type FillNet, fillProbeGap, packFill, probeInput } from "./fill";
import { type Filler, makeFiller } from "./fill-gpu";
import { packHead, probeChunk, probeGap, scoreCpu, type WetHead } from "./head";
import { makeScorer, type Scorer } from "./head-gpu";
import { front, FRONT_BANDS, type Part } from "./zread";

/**
 * One chunk-year per ask: AlphaEarth's 64 bands read from the mosaic in two parts (zread.ts) and the head run after
 * each (WebGPU, else the CPU): a rough picture from the first 32 bands with the rest rebuilt by the fill network,
 * then the exact one. Only the two byte planes go back to the page (wetland, water: 128 KB per 256 x 256 chunk),
 * never the embedding.
 */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; timeIdx: number; ahead?: boolean };
type Msg = { type: "init"; head: WetHead; fill: FillNet | null } | Read | { type: "abort"; id: number };

let ready: Promise<Scorer> | null = null;
let fillNet: FillNet | null = null;
let filling: Promise<Filler> | null = null;
const running = new Map<number, AbortController>();

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
  const cpu: Scorer = async (e, n) => scoreCpu(h, w, e, n);
  const gap = probeGap(h, w);
  let mode = "cpu", gpuGap = NaN;
  let use = cpu;
  try {
    const g = await makeScorer(h, w, 256 * 256);
    if (g) {
      const { e, n } = probeChunk(h);
      const a = await g(e, n), b = scoreCpu(h, w, e, n);
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
    const part: Part = await front(m.timeIdx, m.py0, m.px0, ac.signal, !!m.ahead);
    stop();
    if (!part.full && fillNet) {
      if (!filling) filling = filler(fillNet);
      const fill = await filling;
      const planes = await score(await fill(part.front!, n), n);
      stop();
      postMessage({ id: m.id, planes, rough: true }, { transfer: [planes.buffer as ArrayBuffer] });
    }
    const full = part.full ?? (await part.rest(ac.signal));
    stop();
    const planes = await score(full, n);
    postMessage({ id: m.id, planes }, { transfer: [planes.buffer as ArrayBuffer] });
  } catch (e) {
    const err = e as { name?: string; message?: string };
    postMessage({ id: m.id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
  } finally {
    running.delete(m.id);
  }
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  const m = ev.data;
  if (m.type === "init") {
    fillNet = m.fill && m.fill.k === FRONT_BANDS ? m.fill : null;
    ready = scorer(m.head, packHead(m.head));
  } else if (m.type === "abort") running.get(m.id)?.abort();
  else void read(m);
};
