/** Static renderers for the closed figure vocabulary.
 *
 * Each renderer returns the figure's inner HTML as a string — used verbatim
 * at build time (web page and PDF share the exact same markup) and re-used
 * by the browser hydrators, which re-render with mutated params. All color
 * comes from CSS tokens (SVG carries print-ink presentation attributes; see
 * src/svg.ts); all data comes from `hillspace.ts` or from measured numbers
 * embedded in the spec by the report's author.
 *
 * The primitive figures share one layout language, inherited from the old
 * site's widgets but rebuilt: a large equation row of labeled value chips,
 * a verdict panel (model vs calculator vs error), and a step-by-step
 * walkthrough that ships expanded — the web page collapses it behind a
 * toggle; the PDF and no-JS readers always see every step.
 *
 * Note on glyphs: SVG text uses the precomposed Ŵ (U+0174) because the
 * print renderer positions combining marks poorly inside SVG; HTML text
 * keeps the combining forms, which lay out correctly everywhere.
 */

import {
  additive,
  constrain,
  dConstrain_dmHat,
  dConstrain_dwHat,
  exponential,
  neuralCalc,
  sampleCurve,
  sigmoid,
  trigProducts,
  unitCircle,
} from "./hillspace.js";
import type {
  AdditivePrimitive,
  ConstraintExplorer,
  ExponentialPrimitive,
  FigureSpec,
  GradientVanishing,
  HillSurface,
  NeuralCalculator,
  OptimizerMatrix,
  TrigProducts,
  UnitCircle,
} from "./spec.js";
import { axisLabel, frame, h, hatLabel, hatSafe, polyline, s, text } from "./svg.js";

const title = (s: string) => `<div class="fig-title">${text(s)}</div>`;
const note = (s: string) => `<div class="fig-note">${text(s)}</div>`;
const legend = (entries: [string, string][]) =>
  `<div class="leg-row">${entries
    .map(([label, cls]) => `<span class="leg"><span class="leg-swatch ${cls}"></span>${text(label)}</span>`)
    .join("")}</div>`;

const fmt = (v: number, d = 3) => (Object.is(v, -0) ? (0).toFixed(d) : v.toFixed(d));

// ── the chip/flow/steps layout language ─────────────────────────────────

type ChipCls = "in" | "w" | "out" | "ok" | "err" | "dim";

/** Small inline chip — used in step-by-step lines. */
const chip = (label: string, value: string, cls: ChipCls): string =>
  `<span class="chip chip-${cls}">${label ? `<span class="chip-label">${text(label)}</span>` : ""}${text(value)}</span>`;

/** Large labeled chip — the main equation row reads at a glance. */
const colChip = (label: string, value: string, cls: ChipCls): string =>
  `<span class="chip chip-col chip-${cls}"><span class="chip-label">${text(label)}</span><span class="chip-value">${text(value)}</span></span>`;

const op = (sym: string) => `<span class="flow-op">${text(sym)}</span>`;

const flow = (...parts: string[]) => `<div class="flow">${parts.join("")}</div>`;
const heroFlow = (...parts: string[]) => `<div class="flow flow-hero">${parts.join("")}</div>`;

/** Model output vs the calculator, with an error verdict — three cards,
 * echoing the old widgets' prediction column. */
function verdictCards(predicted: number, truth: number | null, route = ""): string {
  const card = (cls: string, label: string, value: string) =>
    `<div class="vcard ${cls}"><div class="v-label">${text(label)}</div><div class="v-value">${text(value)}</div></div>`;
  if (truth === null) {
    return `<div class="verdict">${card("v-model", "model", fmt(predicted, 6))}${
      route ? `<div class="v-route">${text(route)}</div>` : ""
    }</div>`;
  }
  const err = Math.abs(predicted - truth);
  const exact = err === 0;
  const good = err < 1e-6;
  return `<div class="verdict">${
    card("v-model", "model", fmt(predicted, 6)) +
    card("v-truth", "calculator", fmt(truth, 6)) +
    card(good ? "v-ok" : "v-bad", "error", exact ? "0 — exact" : err.toExponential(2))
  }${route ? `<div class="v-route">${text(route)}</div>` : ""}</div>`;
}

/** Step-by-step walkthrough. Ships expanded (PDF, no-JS); the web page's
 * hydration collapses it behind a "step by step" toggle so it never
 * gatekeeps and never forces itself on readers who don't need it. */
const steps = (items: string[]) =>
  `<div class="steps"><div class="steps-label">step by step</div><ol>${items
    .map((i) => `<li>${i}</li>`)
    .join("")}</ol></div>`;

const eq = (...parts: string[]) => parts.join("");

// ── shared color helpers (data colors, fixed across themes) ─────────────

const lerpC = (a: number[], b: number[], t: number): number[] =>
  a.map((x, i) => x + (b[i] - x) * t);
const rgb = (c: number[]) => `rgb(${c.map(Math.round).join(",")})`;

/** Diverging map for W ∈ [−1, 1]: the site's slate (sigmoid) through a
 * warm gray to its copper (tanh), the web-theme values rather than the
 * print inks so the hill sits in the page's own palette. */
function heatRGB(v: number): number[] {
  const neg = [69, 120, 140];
  const mid = [201, 192, 174];
  const pos = [156, 97, 25];
  return v < 0 ? lerpC(mid, neg, -v) : lerpC(mid, pos, v);
}

// ── constraint-explorer ──────────────────────────────────────────────────

function activationPanel(
  fn: (x: number) => number,
  x0: number,
  yd: [number, number],
  saturations: number[],
  curveCls: string,
  dotCls: string,
  label: string,
  paramName: string,
): string {
  const W = 260;
  const H = 178;
  const XD: [number, number] = [-8, 8];
  const f = frame(W, H, XD, yd, { l: 34, r: 12, t: 16, b: 26 });
  const xc = Math.max(XD[0], Math.min(XD[1], x0));
  const guides = saturations
    .map((y) =>
      s("line", "line-ghost", { x1: f.x(XD[0]), y1: f.y(y), x2: f.x(XD[1]), y2: f.y(y) }),
    )
    .join("");
  const curve = polyline(
    sampleCurve(fn, XD[0], XD[1]).map(([x, y]) => [f.x(x), f.y(y)] as [number, number]),
    curveCls,
  );
  return h(
    "svg",
    { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" },
    guides,
    s("line", "line-axis", { x1: f.x(0), y1: f.y(yd[0]), x2: f.x(0), y2: f.y(yd[1]) }),
    curve,
    s("circle", dotCls, { cx: f.x(xc), cy: f.y(fn(x0)), r: 4 }),
    s("text", "pt-label", { x: f.l, y: 12 }, hatSafe(`${label} = ${fmt(fn(x0))}`)),
    hatLabel(W - 12, 12, paramName, "ax-label ghost", "end"),
    axisLabel(f.x(XD[0]), H - 8, String(XD[0])),
    axisLabel(f.x(0), H - 8, "0"),
    axisLabel(f.x(XD[1]), H - 8, String(XD[1])),
    saturations.map((y) => axisLabel(f.l - 6, f.y(y) + 4, fmt(y).replace(/\.0+$/, ""), "end")),
  );
}

function weightStrip(w: number): string {
  const W = 560;
  const H = 78;
  const f = frame(W, H, [-1.18, 1.18], [0, 1], { l: 20, r: 20, t: 10, b: 10 });
  const yMid = H / 2 + 8;
  const targets = [-1, 0, 1]
    .map(
      (t) =>
        s("circle", "ring-target", { cx: f.x(t), cy: yMid, r: 7 }) +
        s("text", "pt-label ghost", { x: f.x(t), y: yMid + 24, "text-anchor": "middle" }, t > 0 ? `+${t}` : String(t)),
    )
    .join("");
  return h(
    "svg",
    { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" },
    s("line", "line-axis", { x1: f.x(-1.12), y1: yMid, x2: f.x(1.12), y2: yMid }),
    targets,
    s("circle", "dot-weight", { cx: f.x(w), cy: yMid, r: 5 }),
    s("text", "pt-label", { x: f.x(w), y: yMid - 14, "text-anchor": "middle" }, text(`W = ${fmt(w)}`)),
  );
}

export function renderConstraintExplorer(spec: ConstraintExplorer): string {
  const w = constrain(spec.wHat, spec.mHat);
  return (
    title(`the constraint W = tanh(Ŵ) · σ(M̂) — Ŵ = ${spec.wHat.toFixed(1)}, M̂ = ${spec.mHat.toFixed(1)} → W = ${fmt(w)}`) +
    `<div class="panel-row">` +
    activationPanel(Math.tanh, spec.wHat, [-1.15, 1.15], [-1, 1], "line-tanh", "dot-tanh", "tanh(Ŵ)", "Ŵ") +
    activationPanel(sigmoid, spec.mHat, [-0.12, 1.12], [0, 1], "line-sigmoid", "dot-sigmoid", "σ(M̂)", "M̂") +
    `</div>` +
    weightStrip(w) +
    legend([
      ["tanh — sign", "bar-tanh"],
      ["σ — magnitude gate", "bar-sigmoid"],
      ["W — constrained weight", "bar-weight"],
    ]) +
    note("On the web, drag Ŵ and M̂.")
  );
}

// ── hill-surface ─────────────────────────────────────────────────────────

/** The hill itself: W = tanh(Ŵ)·σ(M̂) as a rotated 3D surface, drawn
 * back-to-front in SVG with slope shading. The 3D view is what earned the
 * space its name — the flat heatmap never showed the hill. */
export function renderHillSurface(spec: HillSurface): string {
  const R = spec.range;
  const N = 32;
  const W = 640;
  const H = 470;
  const az = ((spec.azimuthDeg ?? 40) * Math.PI) / 180;
  const cosA = Math.cos(az);
  const sinA = Math.sin(az);
  const SX = 13;
  const SY = 5.6;
  const ZH = 86;
  const CX = 320;
  const CY = 252;
  const proj = (x: number, y: number, z: number): [number, number, number] => {
    const u = x * cosA - y * sinA;
    const v = x * sinA + y * cosA;
    return [CX + u * SX, CY - v * SY - z * ZH, v];
  };
  const d = (2 * R) / N;
  const grid = (i: number) => -R + i * d;
  // light for slope shading; z is exaggerated by ZH/SX on screen
  const K = ZH / (SX * 4);
  const L = [-0.42, 0.34, 0.84];
  const Llen = Math.hypot(...L);
  interface Quad {
    depth: number;
    path: string;
    fill: string;
  }
  const quads: Quad[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const [x0, x1, y0, y1] = [grid(i), grid(i + 1), grid(j), grid(j + 1)];
      const z00 = constrain(x0, y0);
      const z10 = constrain(x1, y0);
      const z11 = constrain(x1, y1);
      const z01 = constrain(x0, y1);
      const zc = (z00 + z10 + z11 + z01) / 4;
      const dzdx = ((z10 + z11 - z00 - z01) / 2 / d) * K;
      const dzdy = ((z01 + z11 - z00 - z10) / 2 / d) * K;
      const nLen = Math.hypot(dzdx, dzdy, 1);
      const dot = (-dzdx * L[0] - dzdy * L[1] + L[2]) / (nLen * Llen);
      const shade = 0.68 + 0.32 * Math.max(0, dot);
      const pts = [proj(x0, y0, z00), proj(x1, y0, z10), proj(x1, y1, z11), proj(x0, y1, z01)];
      const fill = rgb(heatRGB(zc).map((c) => c * shade));
      quads.push({
        depth: (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4,
        path: pts.map(([X, Y], k) => `${k ? "L" : "M"}${X.toFixed(1)} ${Y.toFixed(1)}`).join("") + "Z",
        fill,
      });
    }
  }
  quads.sort((a, b) => b.depth - a.depth);
  const surface = quads
    .map((q) => h("path", { d: q.path, fill: q.fill, stroke: q.fill, "stroke-width": 0.6 }))
    .join("");
  // z reference at the far-left corner
  const zAxis = [-1, 0, 1]
    .map((z) => {
      const [X, Y] = proj(-R, R, z);
      return s("text", "ax-label ghost", { x: X - 8, y: Y + 3, "text-anchor": "end" }, z > 0 ? `+${z}` : String(z));
    })
    .join("");
  const [mx, my] = spec.marker ?? [2, 2];
  const wAt = constrain(mx, my);
  const [dx, dy] = [proj(mx, my, wAt), proj(mx, my, -1.06)];
  const marker =
    s("line", "cross-line", { x1: dy[0], y1: dy[1], x2: dx[0], y2: dx[1] }) +
    s("circle", "dot-weight", { cx: dx[0], cy: dx[1], r: 4.5, stroke: "#f2f4f7", "stroke-width": 1.2 });
  const axisText = (x: number, y: number, label: string) => {
    const [X, Y] = proj(x, y, -1.15);
    return hatLabel(X, Y, label, "ax-label", "middle");
  };
  return (
    title(`the hill: W = tanh(Ŵ) · σ(M̂) over the raw parameters`) +
    h(
      "svg",
      { viewBox: `0 0 ${W} ${H}`, class: "chart hs-map", role: "img" },
      surface,
      zAxis,
      marker,
      axisText(R * 0.7, -R * 1.45, "Ŵ (sign)"),
      axisText(-R * 1.2, R * 0.55, "M̂ (gate)"),
    ) +
    `<div class="flow hs-readout">` +
    chip("Ŵ", fmt(mx, 1), "dim") +
    chip("M̂", fmt(my, 1), "dim") +
    chip("W", fmt(wAt), "out") +
    chip("∂W/∂Ŵ", fmt(dConstrain_dwHat(mx, my), 4), "ok") +
    chip("∂W/∂M̂", fmt(dConstrain_dmHat(mx, my), 4), "ok") +
    `</div>` +
    note(
      "Plateaus: W = +1 amber, −1 teal, 0 gray. The readout gives W and both derivatives at the marker. On the web, spin the hill and move the marker.",
    )
  );
}

// ── gradient-vanishing ───────────────────────────────────────────────────

export function renderGradientVanishing(spec: GradientVanishing): string {
  const W = 560;
  const H = 300;
  const XD: [number, number] = [-12, 12];
  const ceil = Math.tanh(spec.wHat);
  // Fixed frame across the full weight range: the three discrete targets are
  // permanent guides, and the live version morphs smoothly instead of
  // re-scaling and flipping when the sign of Ŵ changes.
  const yd: [number, number] = [-1.18, 1.18];
  const f = frame(W, H, XD, yd, { l: 46, r: 14, t: 16, b: 30 });
  const wCurve = sampleCurve((m) => constrain(spec.wHat, m), XD[0], XD[1]);
  const gCurve = sampleCurve((m) => dConstrain_dmHat(spec.wHat, m), XD[0], XD[1]);
  const toPts = (c: [number, number][]) => c.map(([x, y]) => [f.x(x), f.y(y)] as [number, number]);
  return (
    title(`W and its own gradient across M̂ — Ŵ = ${spec.wHat.toFixed(1)}, so the ceiling is tanh(Ŵ) = ${fmt(ceil)}`) +
    h(
      "svg",
      { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" },
      s("line", "line-ghost", { x1: f.x(XD[0]), y1: f.y(1), x2: f.x(XD[1]), y2: f.y(1) }),
      s("line", "line-ghost", { x1: f.x(XD[0]), y1: f.y(0), x2: f.x(XD[1]), y2: f.y(0) }),
      s("line", "line-ghost", { x1: f.x(XD[0]), y1: f.y(-1), x2: f.x(XD[1]), y2: f.y(-1) }),
      s("line", "line-axis", { x1: f.x(0), y1: f.y(yd[0]), x2: f.x(0), y2: f.y(yd[1]) }),
      polyline(toPts(wCurve), "line-weight"),
      polyline(toPts(gCurve), "line-grad"),
      s("text", "pt-label ghost", { x: f.l + 6, y: f.y(1) - 7 }, "W = +1 plateau"),
      s("text", "pt-label ghost", { x: W - 20, y: f.y(0) - 7, "text-anchor": "end" }, "W = 0 plateau"),
      s("text", "pt-label ghost", { x: f.l + 6, y: f.y(-1) + 16 }, "W = \u22121 plateau"),
      axisLabel(f.x(-10), H - 8, "-10"),
      axisLabel(f.x(0), H - 8, "0"),
      axisLabel(f.x(10), H - 8, "10"),
      hatLabel(W - 8, H - 8, "M̂", "ax-label", "end"),
    ) +
    legend([
      ["W = tanh(Ŵ)·σ(M̂)", "bar-weight"],
      ["∂W/∂M̂ — same scale, unscaled", "bar-good"],
    ]) +
    note(
      "The gradient is drawn on the same scale as W. It peaks mid-slope and fades toward every plateau; there is no particular M̂ where runs get stuck.",
    )
  );
}

// ── optimizer-matrix ─────────────────────────────────────────────────────

/** Five bins for log₁₀ MSE, built from the site's own inks so the table
 * sits inside the page instead of on top of it: two slates for runs at or
 * near the floating-point floor, a pale slate for runs that stalled short,
 * sand for a mild failure, and a muted brick for a clear one. The cutoff
 * between stalled and failed is the paper's MSE > 1e-2. Ink is chosen per
 * bin so every cell's number reads at 4.5:1 or better. */
function mseBin(v: number): { bg: string; fg: string } {
  const cream = "#f6f1e6";
  const dark = "#453425";
  if (v <= -20) return { bg: "#33525e", fg: cream };
  if (v <= -10) return { bg: "#4b6d78", fg: cream };
  if (v <= -2) return { bg: "#95aeb6", fg: dark };
  if (v <= 3) return { bg: "#c2a081", fg: dark };
  return { bg: "#8a4631", fg: cream };
}

export function renderOptimizerMatrix(spec: OptimizerMatrix): string {
  const { rows: specRows, cols, panels } = spec;
  if (!specRows || !cols || !panels)
    throw new Error(
      "optimizer-matrix: unresolved spec — the build resolves `src` into rows/cols/panels; hand-written cells are not supported",
    );
  const panel = (p: { label: string; cells: number[][] }) => {
    const head = `<tr><th></th>${cols.map((c) => `<th>${text(c)}</th>`).join("")}</tr>`;
    const rows = specRows
      .map(
        (r, i) =>
          `<tr><th>${text(r)}</th>${p.cells[i]
            .map((v) => {
              const c = mseBin(v);
              return `<td style="background:${c.bg};color:${c.fg}" title="log10 MSE = ${v}">${v}</td>`;
            })
            .join("")}</tr>`,
      )
      .join("");
    return `<div class="om-panel"><div class="om-label">${text(p.label)}</div><table class="om-table">${head}${rows}</table></div>`;
  };
  return (
    title("measured extrapolation MSE (log₁₀) by operation and optimizer") +
    `<div class="om-row">${panels.map(panel).join("")}</div>` +
    note(
      "Dark slate is the floating-point floor, paler slate stopped short of it, sand and brick failed (MSE above 1e-2).",
    )
  );
}

// ── additive-primitive ───────────────────────────────────────────────────

const near = (v: number, t: number) => Math.abs(v - t) < 0.05;

function additiveOpName(w1: number, w2: number): string {
  if (near(w1, 1) && near(w2, 1)) return "addition (a + b)";
  if (near(w1, 1) && near(w2, -1)) return "subtraction (a − b)";
  if (near(w1, 1) && near(w2, 0)) return "identity (a)";
  if (near(w1, -1) && near(w2, 0)) return "negation (−a)";
  return "a linear combination";
}

function additiveTruth(w1: number, w2: number, a: number, b: number): number | null {
  if (near(w1, 1) && near(w2, 1)) return a + b;
  if (near(w1, 1) && near(w2, -1)) return a - b;
  if (near(w1, 1) && near(w2, 0)) return a;
  if (near(w1, -1) && near(w2, 0)) return -a;
  return null;
}

export function renderAdditivePrimitive(spec: AdditivePrimitive): string {
  const [w1, w2] = spec.w;
  const out = additive(spec.a, spec.b, w1, w2);
  const truth = additiveTruth(w1, w2, spec.a, spec.b);
  return (
    title(`additive primitive — the weights select ${additiveOpName(w1, w2)}`) +
    heroFlow(
      colChip("a", fmt(spec.a, 2), "in"),
      op("×"),
      colChip("w₁", fmt(w1, 2), "w"),
      op("+"),
      colChip("b", fmt(spec.b, 2), "in"),
      op("×"),
      colChip("w₂", fmt(w2, 2), "w"),
      op("="),
      colChip("result", fmt(out, 4), "out"),
    ) +
    verdictCards(out, truth, truth === null ? "no named operation at these weights" : "") +
    steps([
      eq(chip("", fmt(spec.a, 2), "in"), op("×"), chip("", fmt(w1, 2), "w"), op("="), chip("", fmt(spec.a * w1, 4), "dim")),
      eq(chip("", fmt(spec.b, 2), "in"), op("×"), chip("", fmt(w2, 2), "w"), op("="), chip("", fmt(spec.b * w2, 4), "dim")),
      eq(chip("", fmt(spec.a * w1, 4), "dim"), op("+"), chip("", fmt(spec.b * w2, 4), "dim"), op("="), chip("", fmt(out, 6), "out")),
    ]) +
    note(
      "On the web, drive the inputs and weights or use the presets.",
    )
  );
}

// ── exponential-primitive ────────────────────────────────────────────────

function expOpName(w1: number, w2: number): string {
  if (near(w1, 1) && near(w2, 1)) return "multiplication (a × b)";
  if (near(w1, 1) && near(w2, -1)) return "division (a ÷ b)";
  if (near(w1, 1) && near(w2, 0)) return "identity (a)";
  if (near(w1, -1) && near(w2, 0)) return "reciprocal (1/a)";
  return "a product of powers";
}

function expTruth(w1: number, w2: number, a: number, b: number): number | null {
  if (near(w1, 1) && near(w2, 1)) return a * b;
  if (near(w1, 1) && near(w2, -1)) return a / b;
  if (near(w1, 1) && near(w2, 0)) return a;
  if (near(w1, -1) && near(w2, 0)) return 1 / a;
  return null;
}

export function renderExponentialPrimitive(spec: ExponentialPrimitive): string {
  const [w1, w2] = spec.w;
  const z = exponential(spec.a, spec.b, w1, w2);
  const truth = expTruth(w1, w2, spec.a, spec.b);
  const p1 = Math.pow(Math.abs(spec.a), w1) * (spec.a < 0 ? Math.cos(w1 * Math.PI) : 1);
  const p2 = Math.pow(Math.abs(spec.b), w2) * (spec.b < 0 ? Math.cos(w2 * Math.PI) : 1);
  const spin = Math.abs(z.im) > 1e-9;
  return (
    title(`exponential primitive — the weights select ${expOpName(w1, w2)}`) +
    heroFlow(
      colChip("a", fmt(spec.a, 2), "in"),
      op("^"),
      colChip("w₁", fmt(w1, 2), "w"),
      op("×"),
      colChip("b", fmt(spec.b, 2), "in"),
      op("^"),
      colChip("w₂", fmt(w2, 2), "w"),
      op("="),
      colChip("result (Re)", fmt(z.re, 4), "out"),
      spin ? colChip("Im residue", fmt(z.im, 4), "err") : "",
    ) +
    verdictCards(z.re, truth, truth === null ? "fractional weights, between the named selections" : "") +
    steps([
      eq(chip("", fmt(spec.a, 2), "in"), op("^"), chip("", fmt(w1, 2), "w"), op("="), chip("", fmt(p1, 4), "dim")),
      eq(chip("", fmt(spec.b, 2), "in"), op("^"), chip("", fmt(w2, 2), "w"), op("="), chip("", fmt(p2, 4), "dim")),
      eq(chip("", fmt(p1, 4), "dim"), op("×"), chip("", fmt(p2, 4), "dim"), op("="), chip("", fmt(z.re, 6), "out")),
    ]) +
    note(
      "An imaginary residue appears only when a negative base meets a fractional weight.",
    )
  );
}

// ── unit-circle ──────────────────────────────────────────────────────────

function circleFace(cos: number, sin: number, size = 210): string {
  const c = size / 2;
  const r = size * 0.4;
  const px = c + cos * r;
  const py = c - sin * r;
  const ticks = [0, 90, 180, 270]
    .map((deg) => {
      const a = (deg * Math.PI) / 180;
      return s("line", "circle-tick", {
        x1: c + Math.cos(a) * r,
        y1: c - Math.sin(a) * r,
        x2: c + Math.cos(a) * (r + 5),
        y2: c - Math.sin(a) * (r + 5),
      });
    })
    .join("");
  return h(
    "svg",
    { viewBox: `0 0 ${size} ${size}`, class: "circle-chart", role: "img" },
    s("circle", "circle-rim", { cx: c, cy: c, r }),
    ticks,
    s("line", "line-axis", { x1: c - r, y1: c, x2: c + r, y2: c }),
    s("line", "line-axis", { x1: c, y1: c - r, x2: c, y2: c + r }),
    s("line", "proj-cos", { x1: px, y1: py, x2: px, y2: c }),
    s("line", "proj-sin", { x1: px, y1: py, x2: c, y2: py }),
    s("line", "needle", { x1: c, y1: c, x2: px, y2: py }),
    s("circle", "dot-weight", { cx: px, cy: py, r: 4 }),
    s("text", "pt-label small", { x: px, y: c + 12, "text-anchor": "middle" }, text(`cos ${fmt(cos, 2)}`)),
    s("text", "pt-label small", { x: c + 4, y: py - 4 }, text(`sin ${fmt(sin, 2)}`)),
  );
}

function circleOpName(selector: number, phase: number): string {
  if (near(selector, 1) && near(phase, 0)) return "cos(θ)";
  if (near(selector, -1) && near(phase, 0)) return "sin(θ)";
  if (near(selector, 0) && near(phase, 0)) return "the equal mix (cos θ + sin θ)/2";
  return "a phase-shifted blend";
}

export function renderUnitCircle(spec: UnitCircle): string {
  const angle = (spec.angleDeg * Math.PI) / 180;
  const u = unitCircle(angle, spec.selector, spec.phase);
  const truth =
    near(spec.phase, 0) && near(spec.selector, 1)
      ? Math.cos(angle)
      : near(spec.phase, 0) && near(spec.selector, -1)
        ? Math.sin(angle)
        : null;
  return (
    title(`unit-circle primitive — the weights select ${circleOpName(spec.selector, spec.phase)}`) +
    `<div class="flow-cols">` +
    circleFace(u.cos, u.sin) +
    `<div>` +
    heroFlow(
      colChip("θ", `${fmt(spec.angleDeg, 0)}°`, "in"),
      op("+"),
      colChip("phase w₂·π", fmt(spec.phase * Math.PI, 2), "w"),
      op("→"),
      colChip("θ′", `${fmt(u.shifted, 3)} rad`, "dim"),
      op("→"),
      colChip("output", fmt(u.out, 4), "out"),
    ) +
    flow(
      chip("selector w₁", fmt(spec.selector, 2), "w"),
      op("→"),
      chip("(1+w₁)/2 · cos", fmt(((1 + spec.selector) / 2) * u.cos, 4), "dim"),
      op("+"),
      chip("(1−w₁)/2 · sin", fmt(((1 - spec.selector) / 2) * u.sin, 4), "dim"),
    ) +
    verdictCards(u.out, truth) +
    `</div></div>` +
    steps([
      eq(chip("θ", `${fmt(spec.angleDeg, 0)}°`, "in"), op("="), chip("", `${fmt(angle, 3)} rad`, "dim"), op("+"), chip("phase", fmt(spec.phase * Math.PI, 3), "w"), op("="), chip("θ′", fmt(u.shifted, 3), "dim")),
      eq(chip("cos θ′", fmt(u.cos, 4), "dim"), op(","), chip("sin θ′", fmt(u.sin, 4), "dim")),
      eq(
        chip("", fmt((1 + spec.selector) / 2, 2), "w"), op("×"), chip("", fmt(u.cos, 4), "dim"),
        op("+"),
        chip("", fmt((1 - spec.selector) / 2, 2), "w"), op("×"), chip("", fmt(u.sin, 4), "dim"),
        op("="), chip("", fmt(u.out, 6), "out"),
      ),
    ]) +
    note(
      "The dashed drops are cos and sin of the shifted angle; the phase weight rotates first, then the selector blends them.",
    )
  );
}

// ── trig-products ────────────────────────────────────────────────────────

function trigOpName(w0: number, w1: number): string {
  if (near(w0, 1) && near(w1, 1)) return "cos(θ₁ − θ₂)";
  if (near(w0, 1) && near(w1, 0)) return "cos(θ₁ + θ₂)";
  if (near(w0, 0) && near(w1, 1)) return "sin(θ₁ − θ₂)";
  if (near(w0, 0) && near(w1, 0)) return "sin(θ₁ + θ₂)";
  return "a blend of the four products";
}

export function renderTrigProducts(spec: TrigProducts): string {
  const t1 = (spec.theta1Deg * Math.PI) / 180;
  const t2 = (spec.theta2Deg * Math.PI) / 180;
  const [w0, w1] = spec.w;
  const r = trigProducts(t1, t2, w0, w1);
  const truthMap: [boolean, number][] = [
    [near(w0, 1) && near(w1, 1), Math.cos(t1 - t2)],
    [near(w0, 1) && near(w1, 0), Math.cos(t1 + t2)],
    [near(w0, 0) && near(w1, 1), Math.sin(t1 - t2)],
    [near(w0, 0) && near(w1, 0), Math.sin(t1 + t2)],
  ];
  const truth = truthMap.find(([hit]) => hit)?.[1] ?? null;
  const cell = (label: string, value: number, coeff: number) =>
    `<div class="tp-cell${coeff > 0.5 ? " tp-selected" : ""}"><div class="tp-name">${text(label)}</div><div class="tp-val">${fmt(value, 4)}</div><div class="tp-coeff">× ${fmt(coeff, 2)}</div></div>`;
  return (
    title(`trigonometric products primitive — two selection weights choose ${trigOpName(w0, w1)}`) +
    heroFlow(
      colChip("θ₁", `${fmt(spec.theta1Deg, 0)}°`, "in"),
      colChip("θ₂", `${fmt(spec.theta2Deg, 0)}°`, "in"),
      op("→"),
      colChip("w₀", fmt(w0, 2), "w"),
      colChip("w₁", fmt(w1, 2), "w"),
    ) +
    `<div class="tp-grid">` +
    cell("cos(θ₁−θ₂)", r.products.cosDiff, r.coeffs.cosDiff) +
    cell("cos(θ₁+θ₂)", r.products.cosSum, r.coeffs.cosSum) +
    cell("sin(θ₁−θ₂)", r.products.sinDiff, r.coeffs.sinDiff) +
    cell("sin(θ₁+θ₂)", r.products.sinSum, r.coeffs.sinSum) +
    `</div>` +
    verdictCards(r.out, truth) +
    steps([
      eq(chip("cos θ₁", fmt(Math.cos(t1), 4), "dim"), op(","), chip("sin θ₁", fmt(Math.sin(t1), 4), "dim"), op(","), chip("cos θ₂", fmt(Math.cos(t2), 4), "dim"), op(","), chip("sin θ₂", fmt(Math.sin(t2), 4), "dim")),
      `all four sum/difference products form at once from those parts: ${eq(chip("cos(θ₁−θ₂)", fmt(r.products.cosDiff, 4), "dim"), chip("cos(θ₁+θ₂)", fmt(r.products.cosSum, 4), "dim"), chip("sin(θ₁−θ₂)", fmt(r.products.sinDiff, 4), "dim"), chip("sin(θ₁+θ₂)", fmt(r.products.sinSum, 4), "dim"))}`,
      `the weight matrix mixes them: ${eq(
        chip("", fmt(r.coeffs.cosDiff, 2), "w"), op("×"), chip("", fmt(r.products.cosDiff, 4), "dim"), op("+"),
        chip("", fmt(r.coeffs.cosSum, 2), "w"), op("×"), chip("", fmt(r.products.cosSum, 4), "dim"), op("+"),
        chip("", fmt(r.coeffs.sinDiff, 2), "w"), op("×"), chip("", fmt(r.products.sinDiff, 4), "dim"), op("+"),
        chip("", fmt(r.coeffs.sinSum, 2), "w"), op("×"), chip("", fmt(r.products.sinSum, 4), "dim"),
        op("="), chip("", fmt(r.out, 6), "out"),
      )}`,
    ]) +
    note(
      "All four products are computed every time; the highlighted cell is the one the weights select.",
    )
  );
}

// ── neural-calculator ────────────────────────────────────────────────────

const OP_SYM: Record<NeuralCalculator["op"], string> = { add: "+", sub: "−", mul: "×", div: "÷" };

export function renderNeuralCalculator(spec: NeuralCalculator): string {
  const { w, predicted, truth } = neuralCalc(spec.a, spec.b, spec.op);
  const family = spec.op === "add" || spec.op === "sub" ? "additive (a·w₁ + b·w₂)" : "exponential (a^w₁ · b^w₂)";
  return (
    title(`a calculator with enumerated weights — no training, ever`) +
    heroFlow(
      colChip("a", fmt(spec.a, 3), "in"),
      op(OP_SYM[spec.op]),
      colChip("b", fmt(spec.b, 3), "in"),
      op("→"),
      colChip("Ŵ, M̂", "±15", "dim"),
      op("→"),
      colChip("w₁", fmt(w[0], 1), "w"),
      colChip("w₂", fmt(w[1], 1), "w"),
    ) +
    verdictCards(predicted, truth, `route: ${family}`) +
    steps([
      `the operation is looked up, not learned: deep saturation ±15 puts tanh and σ at their fixed points, so ${eq(chip("tanh(±15)·σ(15)", "", "dim"), op("="), chip("", `${fmt(w[0], 1)}, ${fmt(w[1], 1)}`, "w"))}`,
      `those weights route the inputs through the ${family} primitive`,
      eq(chip("model", fmt(predicted, 6), "out"), op("vs"), chip("calculator", fmt(truth, 6), "dim")),
    ]) +
    note(
      "On the web, type your own numbers.",
    )
  );
}

// ── dispatch ─────────────────────────────────────────────────────────────

export function renderFigure(spec: FigureSpec): string {
  switch (spec.kind) {
    case "constraint-explorer":
      return renderConstraintExplorer(spec);
    case "hill-surface":
      return renderHillSurface(spec);
    case "gradient-vanishing":
      return renderGradientVanishing(spec);
    case "optimizer-matrix":
      return renderOptimizerMatrix(spec);
    case "additive-primitive":
      return renderAdditivePrimitive(spec);
    case "exponential-primitive":
      return renderExponentialPrimitive(spec);
    case "unit-circle":
      return renderUnitCircle(spec);
    case "trig-products":
      return renderTrigProducts(spec);
    case "neural-calculator":
      return renderNeuralCalculator(spec);
  }
}
