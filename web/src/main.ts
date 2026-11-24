import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { MapLibreOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { Protocol } from "pmtiles";

import { MIN_ZOOM, RES, X0, Y0 } from "./aef";
import type { FillNet } from "./fill";
import { createGround, GROUND_RAMP } from "./ground";
import type { WetHead } from "./head";
import { BLUGRN } from "./ramp";
import { makeSearch } from "./search";
import { createWetland, type Look } from "./wetland";

maplibregl.setWorkerUrl(maplibreWorkerUrl);
maplibregl.addProtocol("pmtiles", new Protocol().tile);

// The basemap: the USGS topographic map from The National Map (contours, shaded relief, streams and names in one picture)
const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: { topo: { type: "raster", tiles: ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 16 } },
  layers: [{ id: "topo", type: "raster", source: "topo" }],
};
const NWI_TILES = "https://data.source.coop/cboettig/wetlands/nwi-v2.pmtiles";
// Sentinel-2, one cloudless mosaic a year: EOxCloudless by EOX (free for non-commercial use with the credit on the map,
// CC BY-NC-SA 4.0). 2018 is left out (its mosaic is hazy and pale); the map's 2018 shows 2019's imagery.
const S2_YEARS = [2017, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
const S2_URL = (y: number) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${y}_3857/default/g/{z}/{y}/{x}.jpg`;
const s2YearFor = (y: number) => S2_YEARS.reduce((a, b) => (Math.abs(b - y) <= Math.abs(a - y) ? b : a));
// the Yazoo basin, Mississippi: bottomland forest, oxbows and fields
const HOME = { center: [-90.9, 33.1] as [number, number], zoom: 12.6 };
// the map never tilts; the grid is drawn flat
// the map never tilts; the grid is drawn flat
const FLAT = { maxPitch: 0, pitchWithRotate: false, touchPitch: false, dragRotate: false };
/** The ramp's top: wetland probability at its light end. The low end is the reader's (the "from" slider). */
const HI = 0.95;

const [head, fill] = (await Promise.all([
  fetch(`${import.meta.env.BASE_URL}models/wetland_water_aef.json`).then((r) => r.json()),
  // the map runs without the fill (each chunk then draws once, from all its bytes)
  fetch(`${import.meta.env.BASE_URL}models/fill_aef_32p.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
])) as [WetHead, FillNet | null];

// view: "ground" is one value per pixel (low in the view, draining ground, wet score); "wet" is wetland and water alone
const st = { year: head.years[head.years.length - 1]!, lo: 0.08, alpha: 0.9, water: true, nwi: false, labels: true, holding: false, pair: false,
  view: "ground" as "ground" | "wet", galpha: 1, hidden: false, mask: false, picked: null as [number, number] | null };
/** The strength slider's value for the view on the map: the ground picture and the wetland picture each keep their own. */
const strength = () => (st.view === "ground" ? st.galpha : st.alpha);

const mapEl = document.getElementById("map")!;
// the pair's left side (P): Sentinel-2 alone, one camera with the map
const mapEl2 = document.createElement("div");
mapEl2.id = "map2";
document.body.prepend(mapEl2);
const CHEV = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"/></svg>';
// the card, as in the biomass and Earthwork maps: a header that folds, the year, the key, then keys in place of buttons
document.body.insertAdjacentHTML(
  "beforeend",
  `<div class="at-top">
  <div class="card at-glass" id="card">
    <div class="at-hd" id="hd"><span class="t" id="title"></span><i class="at-ramp ramp"></i><span class="hy" id="hy"></span><button class="at-cb" id="cb">${CHEV}</button></div>
    <div class="body">
    <div class="sub" id="sub"></div>
    <div class="yearrow">
      <button class="step" id="prev" title="Earlier year ([)">&lsaquo;</button>
      <div class="year" id="year"></div>
      <button class="step" id="next" title="Later year (])">&rsaquo;</button>
    </div>
    <div class="key"><div class="bar ramp"></div><div class="ticks" id="ticks"></div></div>
    <label class="at-row" id="loRow"><span class="at-lab">Wet from</span><input id="lo" type="range" min="5" max="90" step="1"><span class="at-wtxt" id="loVal"></span></label>
    <label class="at-row"><span class="at-lab">Strength</span><input id="alpha" type="range" min="10" max="100" step="1"><span class="at-wtxt" id="alphaVal"></span></label>
    <div class="readout" id="read">&nbsp;</div>
    <div class="status" id="status"></div>
    <div class="keys" id="keys"></div>
    <div class="credits">Elevation: Terrarium terrain tiles (AWS Open Data; USGS 3DEP in the United States) &middot; head taught on ${head.taught_on} squares of the lower 48 states from NWI polygons; NWI is not an input &middot; AlphaEarth &copy; Google, Google DeepMind, via Source Cooperative &middot; NWI: USFWS via cboettig/wetlands &middot; EOxCloudless <a href="https://cloudless.eox.at" target="_blank" rel="noopener">https://cloudless.eox.at</a> by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2017 to 2025) &middot; basemap: USGS The National Map &middot; search by Photon and Nominatim over &copy; OpenStreetMap</div>
    </div>
  </div>
</div>
<div class="s2card at-glass" id="s2card"></div>
<div class="hint at-glass" id="hint"></div>`,
);
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const say = (t: string) => {
  $("status").textContent = t;
};

const map = new maplibregl.Map({ container: mapEl, style: STYLE, center: HOME.center, zoom: HOME.zoom, attributionControl: false, hash: true, ...FLAT });
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
const overlay = new MapLibreOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
new ResizeObserver(() => map.resize()).observe(mapEl);
// place search over the card, one width with it
const search = makeSearch(map, MIN_ZOOM);
document.querySelector(".at-top")!.prepend(search.el);
await new Promise<void>((ok) => (map.loaded() ? ok() : map.once("load", () => ok())));

const firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
/** The outline's color: white over the ground picture (it reads against the whole yellow-to-purple ramp), orange over the pale topo map. */
const nwiColor = () => (st.view === "ground" && !st.hidden && !st.mask ? "#ffffff" : "#d9730d");
/** The NWI outlines on a map, under its labels: one stroke in its own color, no casing. */
function addNwi(m: maplibregl.Map, before?: string) {
  m.addSource("nwi", { type: "vector", url: `pmtiles://${NWI_TILES}` });
  m.addLayer(
    { id: "nwi-line", type: "line", source: "nwi", "source-layer": "nwi-v2", minzoom: 10, layout: { visibility: st.nwi ? "visible" : "none" },
      paint: { "line-color": nwiColor(), "line-opacity": 0.9, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.3, 13, 0.8, 16, 1.6] } },
    before,
  );
}
addNwi(map, firstLabel);
/** Sentinel-2 on a map: the year's mosaic over everything, nothing drawn on it (made the first time it is asked for), or none. */
function showS2(m: maplibregl.Map, year: number | null) {
  if (year !== null && !m.getLayer(`s2-${year}`)) {
    m.addSource(`s2-${year}`, { type: "raster", tiles: [S2_URL(year)], tileSize: 256, maxzoom: 15 });
    m.addLayer({ id: `s2-${year}`, type: "raster", source: `s2-${year}`, paint: { "raster-fade-duration": 0 } });
  }
  for (const y of S2_YEARS) if (m.getLayer(`s2-${y}`)) m.setLayoutProperty(`s2-${y}`, "visibility", y === year ? "visible" : "none");
  // the picked pixel stays over the imagery
  if (m.getLayer("picked")) m.moveLayer("picked");
}
/** The 10 m pixel's square on the mosaic's grid, as a closed ring. */
function pixelRing(lon: number, lat: number): [number, number][] {
  const x0 = X0 + Math.floor((lon - X0) / RES) * RES, y0 = Y0 - Math.floor((Y0 - lat) / RES) * RES;
  return [[x0, y0], [x0 + RES, y0], [x0 + RES, y0 - RES], [x0, y0 - RES], [x0, y0]];
}
/**
 * The picked pixel's boundary on a map, one stroke in its own color: gold over imagery and the pale basemap, white
 * over the ground picture (gold is lost in the ramp's yellow end).
 */
function showPicked(m: maplibregl.Map, overPicture: boolean) {
  const pt = st.picked;
  const data = { type: "FeatureCollection" as const,
    features: pt ? [{ type: "Feature" as const, properties: {}, geometry: { type: "LineString" as const, coordinates: pixelRing(pt[0], pt[1]) } }] : [] };
  const src = m.getSource("picked") as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(data);
  else {
    m.addSource("picked", { type: "geojson", data });
    m.addLayer({ id: "picked", type: "line", source: "picked", paint: { "line-color": "#ffc828", "line-width": 2.5 } });
  }
  m.setPaintProperty("picked", "line-color", overPicture ? "#ffffff" : "#ffc828");
  m.moveLayer("picked");
}
function drawPicked() {
  showPicked(map, st.view === "ground" && !st.hidden && !st.mask && !st.holding && map.getZoom() >= MIN_ZOOM);
  // its style is in (loaded() is also false while its tiles are still coming, which would skip the release)
  if (map2 && map2Ready) showPicked(map2, false);
}
const setLabels = (m: maplibregl.Map) => {
  for (const l of m.getStyle().layers) if (l.type === "symbol") m.setLayoutProperty(l.id, "visibility", st.labels ? "visible" : "none");
};

const wet = await createWetland(head, () => firstLabel, fill, () => {
  map.triggerRepaint();
  ground.refresh();
  status();
});
// the ground picture sits under the NWI outlines; it is drawn again as the scores under it come in
const ground = createGround(map, wet.sampleWet, () => (map.getLayer("nwi-line") ? "nwi-line" : firstLabel), () => status());
/** Close enough to read and score. */
const closeEnough = () => map.getZoom() >= MIN_ZOOM;
let wasClose = closeEnough();
const groundOn = () => st.view === "ground" && !st.holding && !st.hidden && map.getZoom() >= MIN_ZOOM;

const viewBox = (): [number, number, number, number] => {
  const b = map.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
};
function status() {
  if (map.getZoom() < MIN_ZOOM) return say("");
  const { loaded, rough, wanted } = wet.loadedIn(viewBox(), st.year);
  const how = wet.pool.modes[0] ? ` Scoring on the ${wet.pool.modes[0].mode === "gpu" ? "GPU" : "CPU"}.` : "";
  const g = ground.state;
  const gs = st.view !== "ground" ? "" : g.busy ? " Working out the flow." : g.cells ? ` Ground ${Math.round(g.lo)} to ${Math.round(g.hi)} m in view; flow in ${(g.ms / 1000).toFixed(1)} s.` : "";
  if (gs) return say(loaded < wanted ? `Reading ${st.year}: ${loaded} of ${wanted} pieces in.${gs}` : `${st.year}: all ${wanted} pieces in.${gs}${how}`);
  if (loaded < wanted) say(`Reading ${st.year}: ${loaded} of ${wanted} pieces in.${how}`);
  else if (rough) say(`${st.year}: drawn from half the bands, ${rough} of ${wanted} pieces still sharpening.${how}`);
  else say(`${st.year}: all ${wanted} pieces in.${how}`);
}
function draw() {
  // in the ground view the scores are still read and kept, drawn with no strength: the ground picture takes them
  const look: Look = { lo: st.lo, hi: HI, alpha: st.view === "ground" ? 0 : st.alpha, water: st.water };
  ground.set({ on: groundOn(), year: st.year, opacity: st.galpha, mask: st.mask ? st.lo : null });
  if (map.getLayer("nwi-line")) map.setPaintProperty("nwi-line", "line-color", nwiColor());
  // while Space is held the imagery covers the map and the picture steps aside
  showS2(map, st.holding ? s2YearFor(st.year) : null);
  if (map2 && map2Ready) showS2(map2, s2YearFor(st.year));
  overlay.setProps({ layers: [wet.layer({ year: st.year, visible: !st.holding && !st.hidden && closeEnough(), look, onTile: () => (ground.refresh(), status()), onViewportLoad: () => (ground.refresh(), status()) })] as never });
  drawPicked();
  // the outlines stay over the picture: deck's layer goes in under the labels after them
  for (const ms of [0, 250]) setTimeout(() => map.getLayer("nwi-line") && map.moveLayer("nwi-line", firstLabel), ms);
  status();
}
function paint() {
  $("year").textContent = String(st.year);
  $("loVal").textContent = st.lo.toFixed(2).slice(1);
  $("alphaVal").textContent = `${Math.round(strength() * 100)} %`;
  $<HTMLInputElement>("lo").value = String(Math.round(st.lo * 100));
  $<HTMLInputElement>("alpha").value = String(Math.round(strength() * 100));
  $("hy").textContent = String(st.year);
  // the imagery's year, big, over the map while Space is held; paired, the folded card on the imagery already gives the year
  $("s2card").classList.toggle("on", st.holding);
  $("s2card").innerHTML = `<div class="lab">Sentinel-2</div><div class="yr">${s2YearFor(st.year)}</div>`;
  const on = (b: boolean) => (b ? "on" : "off");
  const gr = st.view === "ground";
  $("title").textContent = gr ? "Low, draining, wet" : "Wetland and water";
  $("sub").textContent = gr
    ? "One value for every pixel: how low the ground sits in the view, how much ground drains through it, and the wet score from the AlphaEarth embedding of one year. It rescales as the map moves."
    : "Scored in your browser from the AlphaEarth embedding of one year, 10 m.";
  $("ticks").innerHTML = gr
    ? '<span style="left:0">high, sheds water, dry</span><span style="left:100%">low, collects water, wet</span>'
    : '<span style="left:0">barely wet</span><span style="left:50%">wetland</span><span style="left:100%">open water</span>';
  // the cutoff is the wetland view's own, and the mask's
  $("loRow").style.display = gr && !st.mask ? "none" : "";
  const rampCss = `linear-gradient(90deg, ${(gr ? GROUND_RAMP : BLUGRN).join(", ")})`;
  for (const e of document.querySelectorAll<HTMLElement>(".ramp")) e.style.background = rampCss;
  $("keys").innerHTML =
    `<kbd>[</kbd><kbd>]</kbd>&nbsp;year &nbsp; <kbd>V</kbd>&nbsp;view, ${gr ? "ground and wet as one" : "wetland and water alone"} &nbsp; ` +
    (gr ? `<kbd>M</kbd>&nbsp;only wet and water, ${on(st.mask)} &nbsp; ` : `<kbd>W</kbd>&nbsp;water, ${on(st.water)} &nbsp; `) + `<kbd>N</kbd>&nbsp;NWI outlines (${nwiColor() === "#ffffff" ? "white" : "orange"}), ${on(st.nwi)} &nbsp; <kbd>H</kbd>&nbsp;${st.hidden ? "show" : "hide"} the picture &nbsp; ` +
    `<kbd>Space</kbd>&nbsp;hold for Sentinel-2 &nbsp; <kbd>P</kbd>&nbsp;pair &nbsp; <kbd>F</kbd>&nbsp;full screen &nbsp; <kbd>/</kbd>&nbsp;search`;
  $<HTMLButtonElement>("prev").disabled = st.year === head.years[0];
  $<HTMLButtonElement>("next").disabled = st.year === head.years[head.years.length - 1];
  hint();
}
/** Too far out to score: say how far in to go, and where the map is now. */
function hint() {
  const z = map.getZoom();
  $("hint").style.display = z < MIN_ZOOM ? "block" : "none";
  if (z < MIN_ZOOM) $("hint").textContent = `Zoom in to score: zoom ${MIN_ZOOM} or closer (now ${(Math.floor(z * 10) / 10).toFixed(1)})`;
}
// the header folds the card to a line: title, ramp, year; remembered
const fold = (on: boolean) => {
  $("card").classList.toggle("collapsed", on);
  $("cb").title = on ? "open the card" : "fold the card";
  try {
    localStorage.setItem("wetland-fold", on ? "1" : "0");
  } catch {
    // no storage: the fold is not remembered
  }
};
$("hd").onclick = () => fold(!$("card").classList.contains("collapsed"));
fold((() => {
  try {
    return localStorage.getItem("wetland-fold") === "1";
  } catch {
    return false;
  }
})());

function step(d: number) {
  const i = head.years.indexOf(st.year) + d;
  if (i < 0 || i >= head.years.length) return;
  st.year = head.years[i]!;
  paint();
  draw();
}
function toggle(k: "water" | "nwi" | "labels") {
  st[k] = !st[k];
  if (k === "nwi") map.setLayoutProperty("nwi-line", "visibility", st.nwi ? "visible" : "none");
  else if (k === "labels") setLabels(map);
  else draw();
  paint();
}

// ---- the pair (P): a second map on the left with Sentinel-2 alone, made the first time the pair opens; either map moves the other
let map2: maplibregl.Map | null = null;
let foldBeforePair = false;
let map2Ready = false;
function boot2() {
  if (map2) return;
  const m2 = new maplibregl.Map({ container: mapEl2, style: { version: 8, sources: {}, layers: [] }, center: map.getCenter(), zoom: map.getZoom(), attributionControl: false, ...FLAT });
  map2 = m2;
  m2.keyboard.disable();
  m2.on("load", () => {
    map2Ready = true;
    showS2(m2, s2YearFor(st.year));
    drawPicked();
  });
  m2.on("click", pickAt);
  let syncing = false;
  const follow = (a: maplibregl.Map, b: maplibregl.Map) =>
    a.on("move", () => {
      if (syncing || !st.pair) return;
      syncing = true;
      b.jumpTo({ center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing() });
      syncing = false;
    });
  follow(map, m2);
  follow(m2, map);
  // one camera, so a point is the same place on both: the left side reads the scores too
  m2.on("mousemove", readAt);
  m2.on("mouseout", readOff);
  new ResizeObserver(() => m2.resize()).observe(mapEl2);
}
function setPair(on: boolean) {
  if (on === st.pair) return;
  if (on) endHold();
  st.pair = on;
  document.body.classList.toggle("pair", on);
  // paired, the card folds to its line (title, ramp, year) over the imagery; it still opens, and leaving the pair
  // puts it back as it was. The remembered fold is not touched.
  const card = $("card");
  if (on) {
    foldBeforePair = card.classList.contains("collapsed");
    card.classList.add("collapsed");
  } else card.classList.toggle("collapsed", foldBeforePair);
  if (on) boot2();
  map.resize();
  map2?.resize();
  if (on) map2!.jumpTo({ center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing() });
  paint();
  draw();
}

// ---- hold Space: Sentinel-2 over the whole map in place of the picture (one map only; paired, it is already on the left)
function beginHold() {
  if (st.holding || st.pair) return;
  st.holding = true;
  paint();
  draw();
}
function endHold() {
  if (!st.holding) return;
  st.holding = false;
  paint();
  draw();
}
window.addEventListener("keyup", (e) => {
  if (e.code === "Space") endHold();
});
// a lost window (Cmd-Tab with Space held) gets no keyup
window.addEventListener("blur", () => endHold());

function fullScreen() {
  if (document.fullscreenElement) void document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => say("full screen is not allowed here"));
}
document.addEventListener("fullscreenchange", () => setTimeout(() => map.resize(), 30));
window.addEventListener("resize", () => setTimeout(() => map.resize(), 30));

$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
$("lo").oninput = () => {
  st.lo = Number($<HTMLInputElement>("lo").value) / 100;
  paint();
  draw();
};
$("alpha").oninput = () => {
  if (st.view === "ground") st.galpha = Number($<HTMLInputElement>("alpha").value) / 100;
  else st.alpha = Number($<HTMLInputElement>("alpha").value) / 100;
  paint();
  draw();
};
window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || (e.target as HTMLElement)?.tagName === "INPUT") return;
  if (e.code === "Space") {
    e.preventDefault();
    if (!e.repeat) beginHold();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === "p") return setPair(!st.pair);
  if (k === "/") {
    e.preventDefault();
    return search.focus();
  }
  if (k === "[") step(-1);
  else if (k === "]") step(1);
  else if (k === "w") toggle("water");
  else if (k === "m") {
    st.mask = !st.mask;
    paint();
    draw();
  } else if (k === "h") {
    st.hidden = !st.hidden;
    paint();
    draw();
  } else if (k === "v") {
    st.view = st.view === "ground" ? "wet" : "ground";
    paint();
    draw();
  }
  else if (k === "n") toggle("nwi");
  else if (k === "f") fullScreen();
});

// a click outlines the pixel's boundary on both maps and holds its numbers in the card; with one outlined, a click
// anywhere lets it go
function pickAt(e: maplibregl.MapMouseEvent) {
  st.picked = st.picked ? null : [e.lngLat.lng, e.lngLat.lat];
  drawPicked();
  if (st.picked) readFor(st.picked[0], st.picked[1]);
}
function readAt(e: maplibregl.MapMouseEvent) {
  readFor(e.lngLat.lng, e.lngLat.lat);
}
function readFor(lng: number, lat: number) {
  if (st.view === "ground") {
    const g = groundOn() ? ground.valueAt(lng, lat) : null;
    $("read").innerHTML = g
      ? `<span>low <b>${g.low.toFixed(2)}</b></span><span>flow <b>${g.flow.toFixed(2)}</b></span><span>wet <b>${g.wet.toFixed(2)}</b></span><span>${Math.round(g.elev)} m</span>`
      : "&nbsp;";
    return;
  }
  const v = map.getZoom() >= MIN_ZOOM ? wet.valueAtPoint(lng, lat, st.year) : null;
  $("read").innerHTML = v
    ? `<span>wetland <b>${v.wetland.toFixed(2)}</b></span><span>water <b>${v.water.toFixed(2)}</b></span><span>upland <b>${v.upland.toFixed(2)}</b></span>`
    : "&nbsp;";
}
function readOff() {
  // off the map, the card goes back to the picked pixel's numbers
  if (st.picked) readFor(st.picked[0], st.picked[1]);
  else $("read").innerHTML = "&nbsp;";
}
map.on("click", pickAt);
map.on("mousemove", readAt);
map.on("mouseout", readOff);
map.on("zoom", hint);
map.on("moveend", () => {
  // crossing the minimum zoom turns the reading on or off
  if (closeEnough() !== wasClose) {
    wasClose = closeEnough();
    draw();
  }
  paint();
  ground.set({ on: groundOn() });
  ground.ask();
  status();
});
wet.pool.onMode = status;

paint();
draw();
