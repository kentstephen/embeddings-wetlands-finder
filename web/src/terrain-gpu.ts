/// <reference types="@webgpu/types" />
import type { Job } from "./chunk-terrain";
import { CELL, N_TERRAIN, STEP } from "./terrain";

/**
 * The terrain numbers of a chunk on the GPU (terrain.ts and chunk-terrain.ts say what they are). The elevation
 * window goes up as it was read, about half a megabyte; the 2 m grid and every layer worked out on it stay in GPU
 * buffers; only the chunk's eleven numbers per pixel come back.
 *
 * Passes, each over the 2 m grid: the window blended onto the grid; small windows (the blur, the 10 and 30 m means,
 * the 30 m minimum) by reading every neighbor; the 100 and 300 m means and minimums by one thread per row or column
 * that walks along it with a running sum or a queue of candidates; then each layer averaged over 10 m cells.
 */
export type TerrainGpu = (j: Job) => Promise<Float32Array>;

/** The largest 2 m grid of a 256 pixel chunk with its margin, on a side, and the largest elevation window in floats. */
const MAX_SIDE = 1700;
const MAX_DEM = 2 * 640 * 640 + 1024;
const Z = 0, A = 1, B = 2, C = 3;

const WGSL = /* wgsl */ `
struct P {
  gw: u32, gh: u32, dw: u32, dh: u32,
  u0: f32, du: f32, v0: f32, dv: f32,
  src: u32, dst: u32, axis: u32, mode: u32,
  radius: u32, band: u32, cols: u32, rows: u32,
  pw: u32, ph: u32, sigma: f32, pad: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dem: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;
@group(0) @binding(6) var<storage, read_write> out10: array<f32>;
@group(0) @binding(7) var<storage, read_write> ter: array<f32>;

fn ld(w: u32, i: u32) -> f32 {
  switch w {
    case 0u { return z[i]; }
    case 1u { return a[i]; }
    case 2u { return b[i]; }
    default { return c[i]; }
  }
}
fn sv(w: u32, i: u32, v: f32) {
  switch w {
    case 0u { z[i] = v; }
    case 1u { a[i] = v; }
    case 2u { b[i] = v; }
    default { c[i] = v; }
  }
}
// an index past either end, mirrored back in
fn mir(i: i32, n: i32) -> u32 {
  var q = i;
  if (q < 0) { q = -q - 1; } else if (q >= n) { q = 2 * n - q - 1; }
  return u32(clamp(q, 0, n - 1));
}
fn blend(o: u32, i: u32, fx: f32, fy: f32) -> f32 {
  return mix(mix(dem[o + i], dem[o + i + 1u], fx), mix(dem[o + i + p.dw], dem[o + i + p.dw + 1u], fx), fy);
}

// the window onto the 2 m grid: the ground into z, the log of one plus its depth below the fill level into c
@compute @workgroup_size(8, 8)
fn resample(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.gw || g.y >= p.gh) { return; }
  let u = p.u0 + f32(g.x) * p.du;
  let v = p.v0 + f32(g.y) * p.dv;
  let c0 = clamp(i32(floor(u)), 0, i32(p.dw) - 2);
  let r0 = clamp(i32(floor(v)), 0, i32(p.dh) - 2);
  let fx = clamp(u - f32(c0), 0.0, 1.0);
  let fy = clamp(v - f32(r0), 0.0, 1.0);
  let i = u32(r0) * p.dw + u32(c0);
  let zv = blend(0u, i, fx, fy);
  let k = g.y * p.gw + g.x;
  z[k] = zv;
  c[k] = log(1.0 + max(blend(p.dw * p.dh, i, fx, fy) - zv, 0.0));
}

// one pass of a window along a row (axis 0) or a column (axis 1): mode 0 a Gaussian, 1 the mean, 2 the minimum
@compute @workgroup_size(8, 8)
fn taps(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.gw || g.y >= p.gh) { return; }
  let r = i32(p.radius);
  let along = p.axis == 0u;
  let n = i32(select(p.gh, p.gw, along));
  let at = i32(select(g.y, g.x, along));
  let stride = select(p.gw, 1u, along);
  let base = select(g.x, g.y * p.gw, along);
  var s = 0.0;
  var wsum = 0.0;
  var lo = 3.0e38;
  for (var j = -r; j <= r; j++) {
    let q = at + j;
    if (p.mode == 2u) {
      if (q >= 0 && q < n) { lo = min(lo, ld(p.src, base + u32(q) * stride)); }
    } else {
      var w = 1.0;
      if (p.mode == 0u) { w = exp(-0.5 * f32(j * j) / (p.sigma * p.sigma)); }
      s += w * ld(p.src, base + mir(q, n) * stride);
      wsum += w;
    }
  }
  sv(p.dst, g.y * p.gw + g.x, select(s / wsum, lo, p.mode == 2u));
}

fn lineAt(line: u32, i: u32) -> u32 {
  if (p.axis == 0u) { return line * p.gw + i; }
  return i * p.gw + line;
}

// the mean over a wide window, one thread per row or column with a running sum
@compute @workgroup_size(64)
fn runmean(@builtin(global_invocation_id) g: vec3<u32>) {
  let along = p.axis == 0u;
  let n = i32(select(p.gh, p.gw, along));
  if (g.x >= select(p.gw, p.gh, along)) { return; }
  let r = i32(p.radius);
  var s = 0.0;
  for (var j = -r; j <= r; j++) { s += ld(p.src, lineAt(g.x, mir(j, n))); }
  let inv = 1.0 / f32(2 * r + 1);
  for (var i = 0; i < n; i++) {
    sv(p.dst, lineAt(g.x, u32(i)), s * inv);
    s += ld(p.src, lineAt(g.x, mir(i + r + 1, n))) - ld(p.src, lineAt(g.x, mir(i - r, n)));
  }
}

// the minimum over a wide window, one thread per row or column with a queue of the places that can still be lowest
@compute @workgroup_size(64)
fn runmin(@builtin(global_invocation_id) g: vec3<u32>) {
  let along = p.axis == 0u;
  let n = i32(select(p.gh, p.gw, along));
  if (g.x >= select(p.gw, p.gh, along)) { return; }
  let r = i32(p.radius);
  var dq: array<i32, 512>;
  var head = 0u;
  var tail = 0u;
  var next = 0;
  for (var i = 0; i < n; i++) {
    let hi = min(n - 1, i + r);
    while (next <= hi) {
      let v = ld(p.src, lineAt(g.x, u32(next)));
      while (tail > head && ld(p.src, lineAt(g.x, u32(dq[(tail - 1u) & 511u]))) >= v) { tail--; }
      dq[tail & 511u] = next;
      tail++;
      next++;
    }
    while (dq[head & 511u] < i - r) { head++; }
    sv(p.dst, lineAt(g.x, u32(i)), ld(p.src, lineAt(g.x, u32(dq[head & 511u]))));
  }
}

// a layer from the ones in hand: 0 slope of src, 1 its curvature, 2 (z - src) squared, 3 roughness from src,
// 4 z less src, 5 the log of one plus z above src
@compute @workgroup_size(8, 8)
fn derive(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.gw || g.y >= p.gh) { return; }
  let k = g.y * p.gw + g.x;
  var v = 0.0;
  if (p.mode <= 1u) {
    let xa = select(g.x, g.x - 1u, g.x > 0u);
    let xb = select(g.x, g.x + 1u, g.x + 1u < p.gw);
    let ya = select(g.y, g.y - 1u, g.y > 0u);
    let yb = select(g.y, g.y + 1u, g.y + 1u < p.gh);
    let l = ld(p.src, g.y * p.gw + xa);
    let rr = ld(p.src, g.y * p.gw + xb);
    let up = ld(p.src, ya * p.gw + g.x);
    let dn = ld(p.src, yb * p.gw + g.x);
    if (p.mode == 0u) {
      let gx = (rr - l) / (f32(xb - xa) * ${STEP.toFixed(1)});
      let gy = (dn - up) / (f32(yb - ya) * ${STEP.toFixed(1)});
      v = log(1.0 + 100.0 * sqrt(gx * gx + gy * gy));
    } else {
      v = (l + rr + up + dn - 4.0 * ld(p.src, k)) / ${(STEP * STEP).toFixed(1)};
    }
  } else if (p.mode == 2u) {
    let d = z[k] - ld(p.src, k);
    v = d * d;
  } else if (p.mode == 3u) {
    v = log(1.0 + 100.0 * sqrt(max(ld(p.src, k), 0.0)));
  } else if (p.mode == 4u) {
    v = z[k] - ld(p.src, k);
  } else {
    v = log(1.0 + max(z[k] - ld(p.src, k), 0.0));
  }
  sv(p.dst, k, v);
}

// a 2 m layer averaged over the 10 m cells, into its band
@compute @workgroup_size(8, 8)
fn block(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.cols || g.y >= p.rows) { return; }
  var s = 0.0;
  for (var dr = 0u; dr < ${CELL}u; dr++) {
    for (var dc = 0u; dc < ${CELL}u; dc++) { s += ld(p.src, (g.y * ${CELL}u + dr) * p.gw + g.x * ${CELL}u + dc); }
  }
  out10[p.band * p.cols * p.rows + g.y * p.cols + g.x] = s / ${(CELL * CELL).toFixed(1)};
}

// every pixel of the chunk takes the 10 m cell its center falls in
@compute @workgroup_size(8, 8)
fn pixels(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.pw || g.y >= p.ph) { return; }
  let o = 2u * p.dw * p.dh;
  let cell = u32(dem[o + p.pw + g.y]) * p.cols + u32(dem[o + g.x]);
  let n = p.pw * p.ph;
  for (var k = 0u; k < ${N_TERRAIN}u; k++) { ter[k * n + g.y * p.pw + g.x] = out10[k * p.cols * p.rows + cell]; }
}
`;

export async function makeTerrainGpu(): Promise<TerrainGpu | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(`terrain shader: ${errs.map((m) => m.message).join("; ")}`);
  const ro = { type: "read-only-storage" as const }, rw = { type: "storage" as const };
  const layout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[ro, rw, rw, rw, rw, rw, rw].map((buffer, i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer }))],
  });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipe = (entryPoint: string) => device.createComputePipeline({ layout: pl, compute: { module, entryPoint } });
  const P = { resample: pipe("resample"), taps: pipe("taps"), runmean: pipe("runmean"), runmin: pipe("runmin"), derive: pipe("derive"), block: pipe("block"), pixels: pipe("pixels") };
  const S = GPUBufferUsage.STORAGE;
  const grid = 4 * MAX_SIDE * MAX_SIDE;
  const tenSize = 4 * N_TERRAIN * (MAX_SIDE / CELL) ** 2, terSize = 4 * N_TERRAIN * 256 * 256;
  const demBuf = device.createBuffer({ size: 4 * MAX_DEM, usage: S | GPUBufferUsage.COPY_DST });
  const bufs = [0, 1, 2, 3].map(() => device.createBuffer({ size: grid, usage: S }));
  const tenBuf = device.createBuffer({ size: tenSize, usage: S });
  const terBuf = device.createBuffer({ size: terSize, usage: S | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: terSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  // every pass has its own settings, so its own small buffer and bind group, made once
  const SLOTS = 64;
  const slots = Array.from({ length: SLOTS }, () => {
    const prm = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bind = device.createBindGroup({ layout, entries: [prm, demBuf, ...bufs, tenBuf, terBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return { prm, bind };
  });

  let queue: Promise<unknown> = Promise.resolve();
  return (j) => {
    const run = async () => {
      if (j.gw > MAX_SIDE || j.gh > MAX_SIDE || j.data.length > MAX_DEM || j.pw * j.ph > 256 * 256) throw new Error("a chunk larger than the terrain buffers");
      device.queue.writeBuffer(demBuf, 0, j.data.buffer as ArrayBuffer, j.data.byteOffset, j.data.byteLength);
      const enc = device.createCommandEncoder();
      let used = 0;
      const go = (pipeline: GPUComputePipeline, o: { src?: number; dst?: number; axis?: number; mode?: number; radius?: number; band?: number; sigma?: number }, x: number, y = 1) => {
        const slot = slots[used++]!;
        const raw = new ArrayBuffer(80);
        const u = new Uint32Array(raw), f = new Float32Array(raw);
        u.set([j.gw, j.gh, j.dw, j.dh], 0);
        f.set([j.u0, j.du, j.v0, j.dv], 4);
        u.set([o.src ?? 0, o.dst ?? 0, o.axis ?? 0, o.mode ?? 0, o.radius ?? 0, o.band ?? 0, j.cols, j.rows, j.pw, j.ph], 8);
        f[18] = o.sigma ?? 1;
        device.queue.writeBuffer(slot.prm, 0, raw);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, slot.bind);
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      const gx = Math.ceil(j.gw / 8), gy = Math.ceil(j.gh / 8);
      /** A window over rows then columns, from src into B (A is the step between). */
      const both = (pipeline: GPUComputePipeline, src: number, o: { mode?: number; radius: number; sigma?: number }) => {
        const lines = pipeline === P.taps;
        go(pipeline, { ...o, src, dst: A, axis: 0 }, lines ? gx : Math.ceil(j.gh / 64), lines ? gy : 1);
        go(pipeline, { ...o, src: A, dst: B, axis: 1 }, lines ? gx : Math.ceil(j.gw / 64), lines ? gy : 1);
      };
      const keep = (band: number) => go(P.block, { src: C, band }, Math.ceil(j.cols / 8), Math.ceil(j.rows / 8));
      const derive = (mode: number, band: number) => {
        go(P.derive, { mode, src: B, dst: C }, gx, gy);
        keep(band);
      };
      const steps = (m: number) => Math.max(Math.round(m / STEP), 1);

      go(P.resample, {}, gx, gy);
      keep(10);
      both(P.taps, Z, { mode: 0, radius: 8, sigma: 2 });
      derive(0, 0);
      derive(1, 1);
      both(P.taps, Z, { mode: 1, radius: (steps(10) - 1) >> 1 });
      go(P.derive, { mode: 2, src: B, dst: C }, gx, gy);
      both(P.taps, C, { mode: 1, radius: (steps(10) - 1) >> 1 });
      derive(3, 2);
      [10, 30, 100, 300].forEach((m, i) => {
        if (m <= 30) both(P.taps, Z, { mode: 1, radius: steps(m) });
        else both(P.runmean, Z, { radius: steps(m) });
        derive(4, 3 + i);
      });
      [30, 100, 300].forEach((m, i) => {
        if (m <= 30) both(P.taps, Z, { mode: 2, radius: steps(m) });
        else both(P.runmin, Z, { radius: steps(m) });
        derive(5, 7 + i);
      });
      go(P.pixels, {}, Math.ceil(j.pw / 8), Math.ceil(j.ph / 8));
      const bytes = 4 * N_TERRAIN * j.pw * j.ph;
      enc.copyBufferToBuffer(terBuf, 0, readBuf, 0, bytes);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ, 0, bytes);
      const out = new Float32Array(readBuf.getMappedRange(0, bytes).slice(0));
      readBuf.unmap();
      return out;
    };
    const r = queue.then(run, run);
    queue = r.catch(() => {});
    return r;
  };
}
