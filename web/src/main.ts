import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { MapLibreOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { Protocol } from "pmtiles";

import { MIN_ZOOM } from "./aef";
import type { FillNet } from "./fill";
import type { WetHead } from "./head";
import { cssGradient, EMRLD, WATER } from "./ramp";
import { createWetland, type Look } from "./wetland";

maplibregl.setWorkerUrl(maplibreWorkerUrl);
maplibregl.addProtocol("pmtiles", new Protocol().tile);

const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const NWI_TILES = "https://data.source.coop/cboettig/wetlands/nwi-v2.pmtiles";
const PHOTOS = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}";
// the Yazoo basin, Mississippi: bottomland forest, oxbows and fields that flood in some years and not others
const HOME = { center: [-90.9, 33.1] as [number, number], zoom: 12.6 };
// the map never tilts; the grid is drawn flat
const FLAT = { maxPitch: 0, pitchWithRotate: false, touchPitch: false, dragRotate: false };
/** The ramp's top: wetland probability at its light end. The low end is the reader's (the "from" slider). */
const HI = 0.95;

const [head, fill] = (await Promise.all([
  fetch(`${import.meta.env.BASE_URL}models/wetland_water_aef.json`).then((r) => r.json()),
  // the map runs without the fill (each chunk then draws once, from all its bytes)
  fetch(`${import.meta.env.BASE_URL}models/fill_aef_32p.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
])) as [WetHead, FillNet | null];

const st = { year: head.years.includes(2024) ? 2024 : head.years[head.years.length - 1]!, lo: 0.3, alpha: 0.9, head: true, water: true, nwi: true, photos: false, labels: true };

const mapEl = document.getElementById("map")!;
document.body.insertAdjacentHTML(
  "beforeend",
  `<div class="at-top">
  <div class="card at-glass">
    <h1>Wetland and water</h1>
    <p class="sub">Scored in your browser from the AlphaEarth embedding of one year, 10 m.</p>
    <div class="row year"><button id="prev" title="earlier year ([)" aria-label="earlier year">&#9664;</button><span id="year"></span><button id="next" title="later year (])" aria-label="later year">&#9654;</button></div>
    <div class="key"><div class="ramp" id="ramp"></div><div class="ends"><span id="loEnd"></span><span>vegetated wetland, probability</span><span>${HI.toFixed(2).slice(1)}</span></div></div>
    <div class="key"><div class="ramp" id="wramp"></div><div class="ends"><span>less sure</span><span>open water</span><span>sure</span></div></div>
    <label class="row"><span>Wetland from</span><input id="lo" type="range" min="5" max="90" step="1"><b id="loVal"></b></label>
    <label class="row"><span>Strength</span><input id="alpha" type="range" min="10" max="100" step="1"><b id="alphaVal"></b></label>
    <div class="row toggles">
      <button id="bHead" title="the head's picture (H)">Head</button><button id="bWater" title="open water (W)">Water</button><button id="bNwi" title="NWI outlines (N)">NWI</button>
      <button id="bPhotos" title="aerial photos under the picture (B)">Photos</button><button id="bLabels" title="place names (L)">Labels</button><button id="bFs" title="full screen (F)">Full screen</button>
    </div>
    <div class="read" id="read"></div>
    <div class="status" id="status"></div>
    <p class="foot">Head taught on ${head.taught_on} squares of the lower 48 states from NWI polygons; NWI is not an input. White lines are NWI.</p>
  </div>
</div>
<div class="hint at-glass" id="hint">Zoom in to score (zoom ${MIN_ZOOM} or closer)</div>`,
);
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const say = (t: string) => {
  $("status").textContent = t;
};

const map = new maplibregl.Map({ container: mapEl, style: STYLE, center: HOME.center, zoom: HOME.zoom, attributionControl: false, hash: true, ...FLAT });
map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: "Embeddings: AlphaEarth Foundations via Source Cooperative. NWI: USFWS via cboettig/wetlands. Photos: USGS The National Map." }), "bottom-left");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
const overlay = new MapLibreOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
new ResizeObserver(() => map.resize()).observe(mapEl);
await new Promise<void>((ok) => (map.loaded() ? ok() : map.once("load", () => ok())));

const firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
map.addSource("photos", { type: "raster", tiles: [PHOTOS], tileSize: 256, maxzoom: 16 });
map.addLayer({ id: "photos", type: "raster", source: "photos", layout: { visibility: "none" } }, firstLabel);
map.addSource("nwi", { type: "vector", url: `pmtiles://${NWI_TILES}` });
// one stroke in its own color, no casing
map.addLayer(
  { id: "nwi-line", type: "line", source: "nwi", "source-layer": "nwi-v2", minzoom: 10,
    paint: { "line-color": "#ffffff", "line-opacity": 0.75, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.3, 13, 0.8, 16, 1.6] } },
  firstLabel,
);

const wet = await createWetland(head, () => firstLabel, fill, () => {
  map.triggerRepaint();
  status();
});

const viewBox = (): [number, number, number, number] => {
  const b = map.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
};
function status() {
  if (map.getZoom() < MIN_ZOOM) return say("");
  const { loaded, rough, wanted } = wet.loadedIn(viewBox(), st.year);
  const how = wet.pool.modes[0] ? ` Scoring on the ${wet.pool.modes[0].mode === "gpu" ? "GPU" : "CPU"}.` : "";
  if (loaded < wanted) say(`Reading ${st.year}: ${loaded} of ${wanted} pieces in.${how}`);
  else if (rough) say(`${st.year}: drawn from half the bands, ${rough} of ${wanted} pieces still sharpening.${how}`);
  else say(`${st.year}: all ${wanted} pieces in.${how}`);
}
function draw() {
  const look: Look = { lo: st.lo, hi: HI, alpha: st.alpha, water: st.water };
  overlay.setProps({ layers: [wet.layer({ year: st.year, visible: st.head, look, onTile: status, onViewportLoad: status })] as never });
  // the outlines stay over the picture: deck's layer goes in under the labels after them
  for (const ms of [0, 250]) setTimeout(() => map.getLayer("nwi-line") && map.moveLayer("nwi-line", firstLabel), ms);
  status();
}
function paint() {
  $("year").textContent = String(st.year);
  $("loVal").textContent = st.lo.toFixed(2).slice(1);
  $("loEnd").textContent = st.lo.toFixed(2).slice(1);
  $("alphaVal").textContent = `${Math.round(st.alpha * 100)} %`;
  $<HTMLInputElement>("lo").value = String(Math.round(st.lo * 100));
  $<HTMLInputElement>("alpha").value = String(Math.round(st.alpha * 100));
  for (const [id, on] of [["bHead", st.head], ["bWater", st.water], ["bNwi", st.nwi], ["bPhotos", st.photos], ["bLabels", st.labels]] as const) {
    $(id).classList.toggle("on", on);
    $(id).setAttribute("aria-pressed", String(on));
  }
  $<HTMLButtonElement>("prev").disabled = st.year === head.years[0];
  $<HTMLButtonElement>("next").disabled = st.year === head.years[head.years.length - 1];
  $("hint").style.display = map.getZoom() < MIN_ZOOM ? "block" : "none";
}
$("ramp").style.background = cssGradient(EMRLD);
$("wramp").style.background = cssGradient(WATER);

function step(d: number) {
  const i = head.years.indexOf(st.year) + d;
  if (i < 0 || i >= head.years.length) return;
  st.year = head.years[i]!;
  paint();
  draw();
}
function toggle(k: "head" | "water" | "nwi" | "photos" | "labels") {
  st[k] = !st[k];
  if (k === "nwi") map.setLayoutProperty("nwi-line", "visibility", st.nwi ? "visible" : "none");
  else if (k === "photos") map.setLayoutProperty("photos", "visibility", st.photos ? "visible" : "none");
  else if (k === "labels") {
    for (const l of map.getStyle().layers) if (l.type === "symbol") map.setLayoutProperty(l.id, "visibility", st.labels ? "visible" : "none");
  } else draw();
  paint();
}
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
  st.alpha = Number($<HTMLInputElement>("alpha").value) / 100;
  paint();
  draw();
};
$("bHead").onclick = () => toggle("head");
$("bWater").onclick = () => toggle("water");
$("bNwi").onclick = () => toggle("nwi");
$("bPhotos").onclick = () => toggle("photos");
$("bLabels").onclick = () => toggle("labels");
$("bFs").onclick = fullScreen;
window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || (e.target as HTMLElement)?.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "[") step(-1);
  else if (k === "]") step(1);
  else if (k === "h") toggle("head");
  else if (k === "w") toggle("water");
  else if (k === "n") toggle("nwi");
  else if (k === "b") toggle("photos");
  else if (k === "l") toggle("labels");
  else if (k === "f") fullScreen();
});

map.on("mousemove", (e) => {
  const v = map.getZoom() >= MIN_ZOOM ? wet.valueAtPoint(e.lngLat.lng, e.lngLat.lat, st.year) : null;
  $("read").innerHTML = v
    ? `<span>wetland <b>${v.wetland.toFixed(2)}</b></span><span>water <b>${v.water.toFixed(2)}</b></span><span>upland <b>${v.upland.toFixed(2)}</b></span>`
    : "";
});
map.on("mouseout", () => {
  $("read").innerHTML = "";
});
map.on("moveend", () => {
  paint();
  status();
});
wet.pool.onMode = status;

paint();
draw();
