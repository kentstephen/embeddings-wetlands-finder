import type { FillNet } from "./fill";
import type { WetHead } from "./head";
import WetWorker from "./wet.worker.ts?worker";

/** A few read-and-score workers; a chunk always goes to the same one. */
type W = { w: Worker; waiting: Map<number, { ok: (p: Uint8Array) => void; no: (e: Error) => void; rough?: (p: Uint8Array) => void }> };
export type Mode = { mode: string; probeGap: number; gpuGap: number };

const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });

export class Pool {
  private ws: W[];
  private nextId = 0;
  modes: Mode[] = [];
  fillModes: string[] = [];
  onMode: (m: Mode) => void = () => {};

  constructor(head: WetHead, fill: FillNet | null, size = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2))) {
    this.ws = Array.from({ length: size }, () => {
      const w = new WetWorker();
      const ent: W = { w, waiting: new Map() };
      w.onmessage = (ev: MessageEvent<{ type?: string; id: number; planes?: Uint8Array; rough?: boolean; error?: { name: string; message: string } } & Partial<Mode>>) => {
        const d = ev.data;
        if (d.type === "fill") {
          this.fillModes.push(d.mode!);
          return;
        }
        if (d.type === "mode") {
          const m = { mode: d.mode!, probeGap: d.probeGap!, gpuGap: d.gpuGap! };
          this.modes.push(m);
          this.onMode(m);
          return;
        }
        if (d.rough) {
          ent.waiting.get(d.id)?.rough?.(d.planes!);
          return;
        }
        const p = ent.waiting.get(d.id);
        if (!p) return;
        ent.waiting.delete(d.id);
        if (d.error) p.no(Object.assign(new Error(d.error.message), { name: d.error.name }));
        else p.ok(d.planes!);
      };
      w.postMessage({ type: "init", head, fill });
      return ent;
    });
  }

  /**
   * One chunk-year of the mosaic, read and scored: [wetland n, water n]. rough: given the planes from the chunk's
   * first 32 bands first (zread.ts), before the exact ones resolve.
   */
  read(timeIdx: number, py0: number, px0: number, h: number, w: number, signal?: AbortSignal, rough?: (p: Uint8Array) => void, ahead = false): Promise<Uint8Array> {
    if (signal?.aborted) return Promise.reject(aborted());
    // a small hash of the chunk's place, so neighbors spread over the workers
    const slot = Math.abs(Math.imul(py0 >> 8, 73856093) ^ Math.imul(px0 >> 8, 19349663));
    const ent = this.ws[slot % this.ws.length]!;
    const id = ++this.nextId;
    return new Promise<Uint8Array>((ok, no) => {
      ent.waiting.set(id, { ok, no, rough });
      ent.w.postMessage({ type: "read", id, px0, py0, w, h, timeIdx, ahead });
      signal?.addEventListener(
        "abort",
        () => {
          if (!ent.waiting.has(id)) return;
          ent.waiting.delete(id);
          ent.w.postMessage({ type: "abort", id });
          no(aborted());
        },
        { once: true },
      );
    });
  }
}
