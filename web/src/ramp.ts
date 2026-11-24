/**
 * CARTO BluGrn, its seven stops; the shader mixes between them, so the steps are continuous on screen.
 * One line from land to open water, read as wetness: pale green where a pixel is barely wet, green where it is
 * vegetated wetland, blue-teal where it is open water. Land itself has no color: the picture fades out before the
 * pale end.
 */
export const BLUGRN = ["#c4e6c3", "#96d2a4", "#6dbc90", "#4da284", "#36877a", "#266b6e", "#1d4f60"];

export const hexToRgb = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16) / 255,
  parseInt(h.slice(3, 5), 16) / 255,
  parseInt(h.slice(5, 7), 16) / 255,
];

/** The ramp as a GLSL const array and its lookup, `name(t)` for t in 0..1. */
export function glslRamp(name: string, stops: string[]): string {
  const rgb = stops.map(hexToRgb);
  const N = rgb.length;
  return `const vec3 ${name.toUpperCase()}[${N}] = vec3[${N}](${rgb.map((c) => `vec3(${c.map((v) => v.toFixed(4)).join(", ")})`).join(", ")});
vec3 ${name}(float t) {
  float x = clamp(t, 0.0, 1.0) * ${N - 1}.0;
  int i = min(int(floor(x)), ${N - 2});
  return mix(${name.toUpperCase()}[i], ${name.toUpperCase()}[i + 1], x - float(i));
}`;
}

export const cssGradient = (stops: string[]) => `linear-gradient(90deg, ${stops.join(", ")})`;
