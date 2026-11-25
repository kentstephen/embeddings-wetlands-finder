/// <reference types="@webgpu/types" />

/**
 * One WebGPU device for everything a worker computes (the fill network, the terrain numbers, the head). A device of
 * its own for each cost about 100 MB apiece across the browser's processes with four workers; null without WebGPU.
 */
let device: Promise<GPUDevice | null> | null = null;

export function sharedDevice(): Promise<GPUDevice | null> {
  device ??= (async () => {
    const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
    const adapter = gpu ? await gpu.requestAdapter() : null;
    return adapter ? adapter.requestDevice() : null;
  })();
  return device;
}
