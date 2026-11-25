/// <reference types="@webgpu/types" />
import { N_EMB, type WetHead, WET_STEPS } from "./head";

/**
 * The head as a WebGPU compute pass, one thread per pixel, run once per chunk-year. The year goes up as raw int8
 * packed in u32 words; each thread dequantizes, runs the network and its softmax, and writes the two bytes of head.ts
 * in one word (wetland in the low 8 bits, water in the next 8). A head with elevation takes its terrain numbers as
 * they are from a second buffer, band-major, and gives no score where the first of them is under 0.
 */
export type Scorer = (e: Int8Array, n: number, t?: Float32Array | null) => Promise<Uint8Array>;

function wgsl(h: WetHead): string {
  const NI = h.n_in, NH = h.hidden, NO = h.n_out;
  const NT = NI - N_EMB;
  const ob1 = NH * NI, o2 = ob1 + NH, ob2 = o2 + NO * NH;
  return /* wgsl */ `
struct Params { n: u32 }
@group(0) @binding(0) var<storage, read> aef: array<u32>;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<uniform> prm: Params;
${NT > 0 ? "@group(0) @binding(4) var<storage, read> ter: array<f32>;" : ""}

fn q8(k: u32) -> i32 {
  return extractBits(bitcast<i32>(aef[k >> 2u]), (k & 3u) * 8u, 8u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let n = prm.n;
  if (p >= n) { return; }
  if (q8(p) == -128) { outp[p] = 0u; return; }
  ${NT > 0 ? "if (!(ter[p] >= 0.0)) { outp[p] = 0u; return; }" : ""}
  var x: array<f32, ${NI}>;
  for (var b = 0u; b < ${N_EMB}u; b++) {
    let f = f32(q8(b * n + p)) / 127.5;
    x[b] = f * abs(f);
  }
  ${NT > 0 ? `for (var b = 0u; b < ${NT}u; b++) { x[${N_EMB}u + b] = ter[b * n + p]; }` : ""}
  var hid: array<f32, ${NH}>;
  for (var j = 0u; j < ${NH}u; j++) {
    var s = wts[${ob1}u + j];
    let r = j * ${NI}u;
    for (var i = 0u; i < ${NI}u; i++) { s += wts[r + i] * x[i]; }
    hid[j] = max(s, 0.0);
  }
  var z: array<f32, ${NO}>;
  for (var c = 0u; c < ${NO}u; c++) {
    var s = wts[${ob2}u + c];
    let r = ${o2}u + c * ${NH}u;
    for (var j = 0u; j < ${NH}u; j++) { s += wts[r + j] * hid[j]; }
    z[c] = s;
  }
  let m = max(z[0], max(z[1], z[2]));
  let a = exp(z[0] - m);
  let b = exp(z[1] - m);
  let c = exp(z[2] - m);
  let s = a + b + c;
  outp[p] = (1u + u32(round(b / s * ${WET_STEPS.toFixed(1)}))) | (u32(round(c / s * 255.0)) << 8u);
}
`;
}

export async function makeScorer(h: WetHead, w: Float32Array, maxN: number): Promise<Scorer | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: wgsl(h) });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(`head shader: ${errs.map((m) => m.message).join("; ")}`);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const aefBuf = device.createBuffer({ size: 64 * maxN, usage: S });
  const NT = h.n_in - N_EMB;
  const terBuf = NT > 0 ? device.createBuffer({ size: 4 * NT * maxN, usage: S }) : null;
  const wBuf = device.createBuffer({ size: w.byteLength, usage: S });
  device.queue.writeBuffer(wBuf, 0, w as Float32Array<ArrayBuffer>);
  const outBuf = device.createBuffer({ size: 4 * maxN, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 4 * maxN, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const prmBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aefBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: prmBuf } },
      ...(terBuf ? [{ binding: 4, resource: { buffer: terBuf } }] : []),
    ],
  });
  // one chunk at a time per worker: the buffers are shared
  let queue: Promise<unknown> = Promise.resolve();
  return (e, n, t) => {
    const run = async () => {
      if (n > maxN) throw new Error(`chunk of ${n} px is over the scorer's ${maxN}`);
      if (terBuf) {
        if (!t) return new Uint8Array(2 * n);
        device.queue.writeBuffer(terBuf, 0, t.buffer as ArrayBuffer, t.byteOffset, 4 * NT * n);
      }
      // writeBuffer wants a multiple of 4 bytes; a chunk is 64 x n bytes, always one
      device.queue.writeBuffer(aefBuf, 0, e.buffer as ArrayBuffer, e.byteOffset, e.byteLength);
      device.queue.writeBuffer(prmBuf, 0, new Uint32Array([n, 0, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4 * n);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ, 0, 4 * n);
      const packed = new Uint32Array(readBuf.getMappedRange(0, 4 * n).slice(0));
      readBuf.unmap();
      const out = new Uint8Array(2 * n);
      for (let p = 0; p < n; p++) {
        const u = packed[p]!;
        out[p] = u & 0xff;
        out[n + p] = (u >>> 8) & 0xff;
      }
      return out;
    };
    const r = queue.then(run, run);
    queue = r.catch(() => {});
    return r;
  };
}
