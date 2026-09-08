/** The Hill Space geometry, mirrored from the study code.
 *
 * Pure functions over plain numbers and arrays — no DOM, no Node APIs — so
 * the same code computes figure data at build time (static SVG for the PDF)
 * and live in the browser (hydrated widgets). Everything here is the
 * closed-form mathematics the report characterizes: the NALU constraint
 * W = tanh(Ŵ) ⊙ σ(M̂), its own derivatives, and the primitives built on it.
 */

export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

/** The constraint: sign from tanh, magnitude gate from sigmoid. */
export const constrain = (wHat: number, mHat: number): number =>
  Math.tanh(wHat) * sigmoid(mHat);

/** Structural derivatives of the constraint itself — the vanishing that
 * pins converged weights to their saturation values. */
export const dConstrain_dwHat = (wHat: number, mHat: number): number =>
  (1 - Math.tanh(wHat) ** 2) * sigmoid(mHat);

export const dConstrain_dmHat = (wHat: number, mHat: number): number =>
  Math.tanh(wHat) * sigmoid(mHat) * (1 - sigmoid(mHat));

/** Sample y = f(x) over [x0, x1] as polyline-ready pairs. */
export function sampleCurve(
  f: (x: number) => number,
  x0: number,
  x1: number,
  steps = 160,
): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const x = x0 + ((x1 - x0) * i) / steps;
    out.push([x, f(x)]);
  }
  return out;
}

// ── primitives ───────────────────────────────────────────────────────────

/** Additive primitive: matrix multiply against the weight column —
 * implicit addition is what turns weight selection into add/sub/identity/negate. */
export const additive = (a: number, b: number, w1: number, w2: number): number =>
  a * w1 + b * w2;

export interface Complex {
  re: number;
  im: number;
}

/** x^w over the complex plane — a negative base with a fractional exponent
 * is a rotation, not a NaN. This mirrors the complex128 evaluation the
 * report's error analysis validates. */
export function cpow(x: number, w: number): Complex {
  if (x === 0) return { re: w === 0 ? 1 : 0, im: 0 };
  const r = Math.pow(Math.abs(x), w);
  const theta = x < 0 ? w * Math.PI : 0;
  return { re: r * Math.cos(theta), im: r * Math.sin(theta) };
}

export const cmul = (p: Complex, q: Complex): Complex => ({
  re: p.re * q.re - p.im * q.im,
  im: p.re * q.im + p.im * q.re,
});

/** Exponential primitive: a^w₁ · b^w₂ evaluated in the complex plane;
 * the real part is the output, the imaginary part the rotation residue. */
export const exponential = (a: number, b: number, w1: number, w2: number): Complex =>
  cmul(cpow(a, w1), cpow(b, w2));

/** Unit-circle primitive: selector in [−1, 1] blends cos toward sin;
 * phase in [−1, 1] shifts the angle by a multiple of π. */
export function unitCircle(
  angle: number,
  selector: number,
  phase: number,
): { shifted: number; cos: number; sin: number; out: number } {
  const shifted = angle + phase * Math.PI;
  const cos = Math.cos(shifted);
  const sin = Math.sin(shifted);
  return { shifted, cos, sin, out: (cos * (1 + selector) + sin * (1 - selector)) / 2 };
}

/** Trigonometric products primitive: the four angle-sum/difference products,
 * mixed by two selector weights (w₀ picks cos vs sin, w₁ picks diff vs sum). */
export function trigProducts(
  t1: number,
  t2: number,
  w0: number,
  w1: number,
): {
  products: { cosDiff: number; cosSum: number; sinDiff: number; sinSum: number };
  coeffs: { cosDiff: number; cosSum: number; sinDiff: number; sinSum: number };
  out: number;
} {
  const c1 = Math.cos(t1);
  const s1 = Math.sin(t1);
  const c2 = Math.cos(t2);
  const s2 = Math.sin(t2);
  const products = {
    cosDiff: c1 * c2 + s1 * s2,
    cosSum: c1 * c2 - s1 * s2,
    sinDiff: s1 * c2 - c1 * s2,
    sinSum: s1 * c2 + c1 * s2,
  };
  const coeffs = {
    cosDiff: w0 * w1,
    cosSum: w0 * (1 - w1),
    sinDiff: (1 - w0) * w1,
    sinSum: (1 - w0) * (1 - w1),
  };
  const out =
    coeffs.cosDiff * products.cosDiff +
    coeffs.cosSum * products.cosSum +
    coeffs.sinDiff * products.sinDiff +
    coeffs.sinSum * products.sinSum;
  return { products, coeffs, out };
}

// ── the enumerated calculator (Experiment 4.1) ───────────────────────────

/** float16 rounding, matching the calculator listing's dtype: tanh(15)
 * rounds to exactly 1.0 in half precision. Uses Math.f16round where the
 * runtime has it; otherwise quantizes to the 11-bit half significand. */
const f16Fallback = (x: number): number => {
  if (!Number.isFinite(x) || x === 0) return x;
  const e = Math.floor(Math.log2(Math.abs(x)));
  const scale = Math.pow(2, Math.max(-14, Math.min(15, e)) - 10);
  return Math.round(x / scale) * scale;
};
export const f16: (x: number) => number =
  (Math as unknown as { f16round?: (x: number) => number }).f16round ?? f16Fallback;

export type CalcOp = "add" | "sub" | "mul" | "div";

/** Enumerated raw parameters: ±15 is deep saturation for both activations. */
export const CALC_RAW: Record<CalcOp, [number, number]> = {
  add: [15, 15],
  sub: [15, -15],
  mul: [15, 15],
  div: [15, -15],
};

/** Run the calculator with enumerated (never trained) weights. */
export function neuralCalc(
  a: number,
  b: number,
  op: CalcOp,
): { w: [number, number]; predicted: number; truth: number } {
  const [r1, r2] = CALC_RAW[op];
  const w: [number, number] = [
    f16(Math.tanh(r1)) * f16(sigmoid(15)),
    f16(Math.tanh(r2)) * f16(sigmoid(15)),
  ];
  const predicted =
    op === "add" || op === "sub" ? additive(a, b, w[0], w[1]) : exponential(a, b, w[0], w[1]).re;
  const truth = op === "add" ? a + b : op === "sub" ? a - b : op === "mul" ? a * b : a / b;
  return { w, predicted, truth };
}

/** Deterministic PRNG (mulberry32) — figure data must be identical between
 * the build-time render and the browser re-render. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
