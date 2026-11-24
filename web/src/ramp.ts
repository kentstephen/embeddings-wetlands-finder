/**
 * CARTO Emrld remade as fine stops (five-min-away-redux): the same hue path, extended to pale sage at the light end.
 * Here its green half, teal to pale green: wetland probability, low to high. The navy end is left off so the low end
 * stays apart from open water's blue and does not sink into a dark basemap.
 */
export const EMRLD = [
  "#217a79", "#287f7b", "#2f857d", "#378a7f", "#3e9080", "#469581", "#4c9b82", "#51a183", "#56a785", "#5aad86",
  "#5fb388", "#64b989", "#6abf8b", "#72c48c", "#79ca8e", "#81cf8f", "#88d591", "#8fdb93", "#96e096", "#9fe598",
  "#abea9a", "#b9ed9d", "#c9f0a0", "#d9f3a7", "#e9f3bf",
];

/** Open water, less sure to sure: one blue hue, apart from the greens in hue and in lightness. */
export const WATER = ["#1d4f9e", "#2a6fdb", "#5a97ee"];

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
