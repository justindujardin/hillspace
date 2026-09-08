/** The closed figure vocabulary.
 *
 * Every ```figure fence in paper/report.md must parse into exactly one member
 * of `FigureSpec`. Unknown kinds and malformed params fail the BUILD, not the
 * reader — the vocabulary is closed on purpose: each kind has a static
 * renderer (used in both outputs) and optionally a hydrator (web).
 */

export interface ConstraintExplorer {
  kind: "constraint-explorer";
  /** Raw parameters before the constraint; the figure shows tanh(Ŵ), σ(M̂),
   * and where their product lands among the discrete targets {−1, 0, +1}. */
  wHat: number;
  mHat: number;
  caption?: string;
}

export interface HillSurface {
  kind: "hill-surface";
  /** Half-width of the raw-parameter square: W is drawn over [−range, range]². */
  range: number;
  /** Optional highlighted point (Ŵ, M̂); the live figure drives it with sliders. */
  marker?: [number, number];
  /** View rotation in degrees; the live figure lets the reader spin the hill. */
  azimuthDeg?: number;
  caption?: string;
}

export interface GradientVanishing {
  kind: "gradient-vanishing";
  /** Fixed sign parameter; the figure plots W and ∂W/∂M̂ across M̂. */
  wHat: number;
  caption?: string;
}

export interface OptimizerMatrix {
  kind: "optimizer-matrix";
  /** Repo-relative path to the experiment's raw JSON. The BUILD resolves it
   * into rows/cols/panels, so figure data is never hand-transcribed — a
   * rerun plus a rebuild is the whole update path. */
  src?: string;
  /** op -> row label, in display order (used with src). */
  rowMap?: Record<string, string>;
  /** optimizer -> column label, in display order (used with src). */
  colMap?: Record<string, string>;
  /** Which snap conditions become panels, with their display labels. */
  select?: { condition: number | null; label: string }[];
  /** Resolved form (filled by the build when src is present) —
   * cells are log₁₀ extrapolation MSE, one row per operation. */
  rows?: string[];
  cols?: string[];
  panels?: { label: string; cells: number[][] }[];
  caption?: string;
}

export interface AdditivePrimitive {
  kind: "additive-primitive";
  a: number;
  b: number;
  /** Effective weights [w₁, w₂] after the constraint. */
  w: [number, number];
  caption?: string;
}

export interface ExponentialPrimitive {
  kind: "exponential-primitive";
  a: number;
  b: number;
  w: [number, number];
  caption?: string;
}

export interface UnitCircle {
  kind: "unit-circle";
  angleDeg: number;
  /** Selector weight in [−1, 1]: +1 = cos, −1 = sin, 0 = equal mix. */
  selector: number;
  /** Phase-shift weight in [−1, 1], applied as a multiple of π. */
  phase: number;
  caption?: string;
}

export interface TrigProducts {
  kind: "trig-products";
  theta1Deg: number;
  theta2Deg: number;
  /** Selector weights [w₀, w₁]: w₀ mixes cos/sin, w₁ mixes diff/sum. */
  w: [number, number];
  caption?: string;
}

export interface NeuralCalculator {
  kind: "neural-calculator";
  a: number;
  b: number;
  op: "add" | "sub" | "mul" | "div";
  caption?: string;
}

export type FigureSpec =
  | ConstraintExplorer
  | HillSurface
  | GradientVanishing
  | OptimizerMatrix
  | AdditivePrimitive
  | ExponentialPrimitive
  | UnitCircle
  | TrigProducts
  | NeuralCalculator;

export const FIGURE_KINDS = [
  "constraint-explorer",
  "hill-surface",
  "gradient-vanishing",
  "optimizer-matrix",
  "additive-primitive",
  "exponential-primitive",
  "unit-circle",
  "trig-products",
  "neural-calculator",
] as const;

export function parseFigureSpec(json: string): FigureSpec {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (e) {
    throw new Error(`figure fence is not valid JSON: ${(e as Error).message}\n${json}`);
  }
  const spec = raw as { kind?: string };
  if (!spec.kind || !(FIGURE_KINDS as readonly string[]).includes(spec.kind)) {
    throw new Error(
      `unknown figure kind ${JSON.stringify(spec.kind)} — the vocabulary is closed: ${FIGURE_KINDS.join(", ")}`,
    );
  }
  return raw as FigureSpec;
}
