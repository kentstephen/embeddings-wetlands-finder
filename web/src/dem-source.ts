import { CHUNK } from "./aef";
import { boxOf } from "./chunk-terrain";
import { type Dem, demState, readDem } from "./dem";

/**
 * The elevation window under each chunk of the mosaic, read once and kept while it is likely to be asked for again
 * (another year of the same place): about half a megabyte a chunk, the last used last, at most CAP chunks.
 * A read that fails is tried again; one that keeps failing is an error for whoever asked, never an empty answer.
 */
const CAP = 96, TRIES = 3;

export class DemSource {
  private have = new Map<string, Dem>();
  private flying = new Map<string, Promise<Dem>>();
  /** Why the last read that failed did. */
  error = "";
  onChange: () => void = () => {};

  /** Windows being read now. */
  get waiting() {
    return this.flying.size;
  }

  get(px0: number, py0: number, w: number, h: number): Promise<Dem> {
    const key = `${Math.floor(py0 / CHUNK)}-${Math.floor(px0 / CHUNK)}`;
    const had = this.have.get(key);
    if (had) {
      this.have.delete(key);
      this.have.set(key, had);
      return Promise.resolve(had);
    }
    let p = this.flying.get(key);
    if (!p) {
      const box = boxOf(px0, py0, w, h);
      p = (async () => {
        for (let t = 1; ; t++) {
          try {
            const d = await readDem(...box);
            if (demState.error) throw new Error(demState.error);
            return d;
          } catch (e) {
            if (t === TRIES) throw e;
            demState.error = "";
            await new Promise((ok) => setTimeout(ok, 400 * t));
          }
        }
      })();
      this.flying.set(key, p);
      p.then(
        (d) => {
          this.have.set(key, d);
          for (const k of this.have.keys()) {
            if (this.have.size <= CAP) break;
            this.have.delete(k);
          }
          this.error = "";
        },
        (e) => {
          this.error = (e as Error)?.message || String(e);
          demState.error = "";
        },
      ).finally(() => {
        this.flying.delete(key);
        this.onChange();
      });
    }
    return p;
  }
}
