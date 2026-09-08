/** Hill Space ink vocabulary over the penname SVG builder.
 *
 * Print-safe ink for every SVG class in the vocabulary: WeasyPrint does not
 * apply CSS stroke/fill rules to inline SVG, so every element carries
 * presentation attributes in print colors; the dark web theme overrides
 * them by class. One markup string serves both outputs.
 */

import { makeStyled, MONO, text as esc, type Attrs } from "penname/svg";

export { frame, h, text, type Attrs, type Frame } from "penname/svg";

const INK = {
  tanh: "#b06f10",
  sigmoid: "#0b7285",
  weight: "#6d4fa1",
  good: "#1e7d43",
  bad: "#b3362c",
  dim: "#3d434e",
  faint: "#6a7180",
  ghost: "#9aa1ad",
};

const CLASS_ATTRS: Record<string, Attrs> = {
  "line-tanh": { stroke: INK.tanh, "stroke-width": 2 },
  "line-sigmoid": { stroke: INK.sigmoid, "stroke-width": 2 },
  "line-weight": { stroke: INK.weight, "stroke-width": 2 },
  "line-good": { stroke: INK.good, "stroke-width": 2 },
  "line-bad": { stroke: INK.bad, "stroke-width": 2 },
  "line-ghost": { stroke: INK.ghost, "stroke-width": 1.4, "stroke-dasharray": "5 5" },
  "line-axis": { stroke: INK.ghost, "stroke-width": 1 },
  "dot-tanh": { fill: INK.tanh },
  "dot-sigmoid": { fill: INK.sigmoid },
  "dot-weight": { fill: INK.weight },
  "dot-bad": { fill: INK.bad },
  "ring-target": { fill: "none", stroke: INK.ghost, "stroke-width": 1.4 },
  "bar-tanh": { fill: INK.tanh },
  "bar-sigmoid": { fill: INK.sigmoid },
  "bar-weight": { fill: INK.weight },
  "bar-ghost": { fill: INK.ghost },
  "bar-bad": { fill: INK.bad },
  "line-grad": { stroke: INK.good, "stroke-width": 2 },
  "circle-rim": { stroke: INK.ghost, "stroke-width": 1.4, fill: "none" },
  "circle-tick": { stroke: INK.ghost, "stroke-width": 1 },
  "proj-cos": { stroke: INK.tanh, "stroke-width": 1.4, "stroke-dasharray": "4 4" },
  "proj-sin": { stroke: INK.sigmoid, "stroke-width": 1.4, "stroke-dasharray": "4 4" },
  needle: { stroke: INK.weight, "stroke-width": 2.2, "stroke-linecap": "round" },
  "cross-line": { stroke: INK.dim, "stroke-width": 1, "stroke-dasharray": "3 3" },
  "ax-label": { fill: INK.faint, "font-family": MONO, "font-size": 11 },
  "ax-label ghost": { fill: INK.ghost, "font-family": MONO, "font-size": 11 },
  "pt-label": { fill: INK.dim, "font-family": MONO, "font-size": 11 },
  "pt-label small": { fill: INK.dim, "font-family": MONO, "font-size": 9.5 },
  "pt-label ghost": { fill: INK.ghost, "font-family": MONO, "font-size": 11 },
};

export const { s, axisLabel, polyline } = makeStyled(CLASS_ATTRS);

/** SVG text content with print-safe hats. The PDF renderer mispositions
 * combining marks inside SVG <text> (Ŵ has a precomposed glyph, M̂ does
 * not), so "M̂" is rebuilt as M plus a modifier-circumflex tspan nudged
 * over it. Use only for SVG text, never HTML — and only with the default
 * start anchor: the print renderer re-anchors every tspan chunk, so
 * middle/end anchoring scrambles the pieces (see hatLabel). */
export function hatSafe(label: string): string {
  const parts = label.split("M̂");
  if (parts.length === 1) return esc(label);
  let out = esc(parts[0]);
  for (let i = 1; i < parts.length; i++) {
    out += `M<tspan dx="-6.8" dy="-3">ˆ</tspan><tspan dx="0.2" dy="3">${esc(parts[i])}</tspan>`;
  }
  return out;
}

/** Anchored SVG label that may contain M̂: plain labels anchor natively;
 * hatted ones are converted to a start anchor at an x computed from the
 * mono glyph advance (0.602 em), because multi-chunk anchoring is unsafe
 * in print. */
export function hatLabel(
  x: number,
  y: number,
  label: string,
  cls = "ax-label",
  anchor: "start" | "middle" | "end" = "middle",
  size = 11,
): string {
  if (!label.includes("M̂")) return s("text", cls, { x, y, "text-anchor": anchor }, esc(label));
  const w = label.replace(/̂/g, "").length * size * 0.602;
  const xs = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
  return s("text", cls, { x: xs, y }, hatSafe(label));
}
