/** Browser entry: upgrade static figures to interactive ones.
 *
 * Every figure ships working static SVG/HTML; hydration is progressive
 * enhancement. Most hydrators re-render through the same `renderFigure`
 * pipeline with mutated params — there is exactly one rendering path. The
 * hill-surface is the one exception: its heatmap is rendered once and only
 * the crosshair and readouts move under the pointer.
 *
 * A few page-wide enhancements live here too, all additive: long code
 * listings fold behind a "show all" button, step-by-step walkthroughs
 * (shipped expanded for the PDF and no-JS readers) collapse behind their
 * label, the floating contents panel tracks the reading position, and
 * the # after a heading copies its link, and a marked term shows its
 * plain description on a click. Without JavaScript everything is simply
 * visible.
 */

import {
  ctlRow,
  enableCodeFold,
  enableGlossary,
  enableHeadingLinks,
  enableStepsToggle,
  enableToc,
  numberInput,
  presetButtons,
  runHydrators,
  slider,
  valueOut,
  type Hydrator,
} from "penname/hydrate";

import { renderFigure } from "./figures.js";
import type {
  AdditivePrimitive,
  ConstraintExplorer,
  ExponentialPrimitive,
  FigureSpec,
  GradientVanishing,
  HillSurface,
  NeuralCalculator,
  TrigProducts,
  UnitCircle,
} from "./spec.js";

const nearW = (w: [number, number], t: [number, number]) =>
  Math.abs(w[0] - t[0]) < 0.05 && Math.abs(w[1] - t[1]) < 0.05;

/** First preset whose weight pair matches the live weights, else null. */
function matchPreset<T extends string>(w: [number, number], table: [T, [number, number]][]): T | null {
  return table.find(([, tw]) => nearW(w, tw))?.[0] ?? null;
}

const hydrators: Partial<Record<FigureSpec["kind"], Hydrator<FigureSpec>>> = {
  "constraint-explorer": (body, spec0) => {
    const spec = { ...(spec0 as ConstraintExplorer) };
    const stage = document.createElement("div");
    const wOut = valueOut();
    const mOut = valueOut();
    const render = () => {
      wOut.textContent = `Ŵ = ${spec.wHat.toFixed(1)}`;
      mOut.textContent = `M̂ = ${spec.mHat.toFixed(1)}`;
      stage.innerHTML = renderFigure(spec);
    };
    body.replaceChildren(
      ctlRow(
        "sign",
        slider(-15, 15, 0.1, spec.wHat, (v) => ((spec.wHat = v), render())),
        wOut,
        "gate",
        slider(-15, 15, 0.1, spec.mHat, (v) => ((spec.mHat = v), render())),
        mOut,
      ),
      stage,
    );
    render();
  },

  "hill-surface": (body, spec0) => {
    const spec = {
      ...(spec0 as HillSurface),
      marker: [...((spec0 as HillSurface).marker ?? [2, 2])] as [number, number],
    };
    const stage = document.createElement("div");
    const azOut = valueOut();
    const wOut = valueOut();
    const mOut = valueOut();
    // full re-render per input, throttled to animation frames — the surface
    // is ~1000 shaded quads, cheap enough to rebuild but not per-event
    let raf = 0;
    const render = () => {
      raf = 0;
      azOut.textContent = `${(spec.azimuthDeg ?? 40).toFixed(0)}°`;
      wOut.textContent = `Ŵ = ${spec.marker[0].toFixed(1)}`;
      mOut.textContent = `M̂ = ${spec.marker[1].toFixed(1)}`;
      stage.innerHTML = renderFigure(spec);
    };
    const queue = () => {
      if (!raf) raf = requestAnimationFrame(render);
    };
    body.replaceChildren(
      ctlRow(
        "spin",
        slider(5, 85, 1, spec.azimuthDeg ?? 40, (v) => ((spec.azimuthDeg = v), queue())),
        azOut,
      ),
      ctlRow(
        "marker",
        slider(-15, 15, 0.1, spec.marker[0], (v) => ((spec.marker[0] = v), queue())),
        wOut,
        slider(-15, 15, 0.1, spec.marker[1], (v) => ((spec.marker[1] = v), queue())),
        mOut,
      ),
      stage,
    );
    render();
  },

  "gradient-vanishing": (body, spec0) => {
    const spec = { ...(spec0 as GradientVanishing) };
    const stage = document.createElement("div");
    const out = valueOut();
    const render = () => {
      out.textContent = `Ŵ = ${spec.wHat.toFixed(1)}`;
      stage.innerHTML = renderFigure(spec);
    };
    body.replaceChildren(
      ctlRow("sign strength", slider(-15, 15, 0.1, spec.wHat, (v) => ((spec.wHat = v), render())), out),
      stage,
    );
    render();
  },

  "additive-primitive": (body, spec0) => {
    const spec = { ...(spec0 as AdditivePrimitive), w: [...(spec0 as AdditivePrimitive).w] as [number, number] };
    const stage = document.createElement("div");
    const w1Out = valueOut();
    const w2Out = valueOut();
    const w1 = slider(-1, 1, 0.01, spec.w[0], (v) => ((spec.w[0] = v), render()));
    const w2 = slider(-1, 1, 0.01, spec.w[1], (v) => ((spec.w[1] = v), render()));
    const TABLE: ["add" | "sub" | "id" | "neg", [number, number]][] = [
      ["add", [1, 1]],
      ["sub", [1, -1]],
      ["id", [1, 0]],
      ["neg", [-1, 0]],
    ];
    const render = () => {
      w1Out.textContent = `w₁ = ${spec.w[0].toFixed(2)}`;
      w2Out.textContent = `w₂ = ${spec.w[1].toFixed(2)}`;
      stage.innerHTML = renderFigure(spec);
      pb.refresh();
    };
    const setW = (a: number, b: number) => {
      spec.w = [a, b];
      w1.value = String(a);
      w2.value = String(b);
      render();
    };
    const pb = presetButtons(
      [
        ["add", "a+b"],
        ["sub", "a−b"],
        ["id", "a"],
        ["neg", "−a"],
      ],
      (k) => setW(...(Object.fromEntries(TABLE)[k] as [number, number])),
      () => matchPreset(spec.w, TABLE),
    );
    body.replaceChildren(
      ctlRow(
        "a",
        numberInput(spec.a, (v) => ((spec.a = v), render())),
        "b",
        numberInput(spec.b, (v) => ((spec.b = v), render())),
        pb.el,
      ),
      ctlRow("w₁", w1, w1Out, "w₂", w2, w2Out),
      stage,
    );
    render();
  },

  "exponential-primitive": (body, spec0) => {
    const spec = { ...(spec0 as ExponentialPrimitive), w: [...(spec0 as ExponentialPrimitive).w] as [number, number] };
    const stage = document.createElement("div");
    const w1Out = valueOut();
    const w2Out = valueOut();
    const w1 = slider(-1, 1, 0.01, spec.w[0], (v) => ((spec.w[0] = v), render()));
    const w2 = slider(-1, 1, 0.01, spec.w[1], (v) => ((spec.w[1] = v), render()));
    const TABLE: ["mul" | "div" | "id" | "rec", [number, number]][] = [
      ["mul", [1, 1]],
      ["div", [1, -1]],
      ["id", [1, 0]],
      ["rec", [-1, 0]],
    ];
    const render = () => {
      w1Out.textContent = `w₁ = ${spec.w[0].toFixed(2)}`;
      w2Out.textContent = `w₂ = ${spec.w[1].toFixed(2)}`;
      stage.innerHTML = renderFigure(spec);
      pb.refresh();
    };
    const setW = (a: number, b: number) => {
      spec.w = [a, b];
      w1.value = String(a);
      w2.value = String(b);
      render();
    };
    const pb = presetButtons(
      [
        ["mul", "a×b"],
        ["div", "a÷b"],
        ["id", "a"],
        ["rec", "1/a"],
      ],
      (k) => setW(...(Object.fromEntries(TABLE)[k] as [number, number])),
      () => matchPreset(spec.w, TABLE),
    );
    body.replaceChildren(
      ctlRow(
        "a",
        numberInput(spec.a, (v) => ((spec.a = v), render())),
        "b",
        numberInput(spec.b, (v) => ((spec.b = v), render())),
        pb.el,
      ),
      ctlRow("w₁", w1, w1Out, "w₂", w2, w2Out),
      stage,
    );
    render();
  },

  "unit-circle": (body, spec0) => {
    const spec = { ...(spec0 as UnitCircle) };
    const stage = document.createElement("div");
    const aOut = valueOut();
    const selOut = valueOut();
    const phOut = valueOut();
    const sel = slider(-1, 1, 0.01, spec.selector, (v) => ((spec.selector = v), render()));
    const ph = slider(-1, 1, 0.01, spec.phase, (v) => ((spec.phase = v), render()));
    const TABLE: ["cos" | "sin" | "mix", [number, number]][] = [
      ["cos", [1, 0]],
      ["sin", [-1, 0]],
      ["mix", [0, 0]],
    ];
    const render = () => {
      aOut.textContent = `θ = ${spec.angleDeg.toFixed(0)}°`;
      selOut.textContent = `w₁ = ${spec.selector.toFixed(2)}`;
      phOut.textContent = `w₂ = ${spec.phase.toFixed(2)}`;
      stage.innerHTML = renderFigure(spec);
      pb.refresh();
    };
    const setW = (selector: number, phase: number) => {
      spec.selector = selector;
      spec.phase = phase;
      sel.value = String(selector);
      ph.value = String(phase);
      render();
    };
    const pb = presetButtons(
      [
        ["cos", "cos"],
        ["sin", "sin"],
        ["mix", "mix"],
      ],
      (k) => setW(...(Object.fromEntries(TABLE)[k] as [number, number])),
      () => matchPreset([spec.selector, spec.phase], TABLE),
    );
    body.replaceChildren(
      ctlRow(
        "angle",
        slider(0, 360, 1, spec.angleDeg, (v) => ((spec.angleDeg = v), render())),
        aOut,
        pb.el,
      ),
      ctlRow("selector", sel, selOut, "phase", ph, phOut),
      stage,
    );
    render();
  },

  "trig-products": (body, spec0) => {
    const spec = { ...(spec0 as TrigProducts), w: [...(spec0 as TrigProducts).w] as [number, number] };
    const stage = document.createElement("div");
    const t1Out = valueOut();
    const t2Out = valueOut();
    const TABLE: ["cd" | "cs" | "sd" | "ss", [number, number]][] = [
      ["cd", [1, 1]],
      ["cs", [1, 0]],
      ["sd", [0, 1]],
      ["ss", [0, 0]],
    ];
    const render = () => {
      t1Out.textContent = `θ₁ = ${spec.theta1Deg.toFixed(0)}°`;
      t2Out.textContent = `θ₂ = ${spec.theta2Deg.toFixed(0)}°`;
      stage.innerHTML = renderFigure(spec);
      pb.refresh();
    };
    const setW = (w0: number, w1: number) => {
      spec.w = [w0, w1];
      render();
    };
    const pb = presetButtons(
      [
        ["cd", "cos(θ₁−θ₂)"],
        ["cs", "cos(θ₁+θ₂)"],
        ["sd", "sin(θ₁−θ₂)"],
        ["ss", "sin(θ₁+θ₂)"],
      ],
      (k) => setW(...(Object.fromEntries(TABLE)[k] as [number, number])),
      () => matchPreset(spec.w, TABLE),
    );
    body.replaceChildren(
      ctlRow(
        "θ₁",
        slider(0, 360, 1, spec.theta1Deg, (v) => ((spec.theta1Deg = v), render())),
        t1Out,
        "θ₂",
        slider(0, 360, 1, spec.theta2Deg, (v) => ((spec.theta2Deg = v), render())),
        t2Out,
      ),
      ctlRow("select", pb.el),
      stage,
    );
    render();
  },

  "neural-calculator": (body, spec0) => {
    const spec = { ...(spec0 as NeuralCalculator) };
    const stage = document.createElement("div");
    const render = () => {
      stage.innerHTML = renderFigure(spec);
      pb.refresh();
    };
    const pb = presetButtons(
      [
        ["add", "+"],
        ["sub", "−"],
        ["mul", "×"],
        ["div", "÷"],
      ],
      (k) => ((spec.op = k), render()),
      () => spec.op,
    );
    body.replaceChildren(
      ctlRow(
        "a",
        numberInput(spec.a, (v) => ((spec.a = v), render())),
        pb.el,
        "b",
        numberInput(spec.b, (v) => ((spec.b = v), render())),
      ),
      stage,
    );
    render();
  },
};

runHydrators(hydrators);
enableStepsToggle();
enableCodeFold();
enableToc();
enableHeadingLinks();
enableGlossary();
