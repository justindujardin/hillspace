/** paper/report.md in, two artifacts out — the machinery lives in
 * penname; this file wires in the Hill Space vocabulary and the one
 * build-time data resolution the paper needs.
 *
 *   npm run build           # web + pdf
 *   npm run build:web       # skip the pdf
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { buildNote } from "penname";

import { renderFigure } from "../src/figures.js";
import { parseFigureSpec } from "../src/spec.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Resolve figure specs that reference experiment output. The renderer never
 * reads files (it runs in the browser too), so the build loads the raw JSON
 * an experiment wrote, aggregates it into the spec's resolved form, and
 * embeds THAT — figure data is what the experiment printed, never a human
 * transcription. A rerun plus a rebuild is the whole update path. */
function resolveFigureData(spec: ReturnType<typeof parseFigureSpec>): ReturnType<typeof parseFigureSpec> {
  if (spec.kind !== "optimizer-matrix" || !spec.src) return spec;
  if (!spec.rowMap || !spec.colMap || !spec.select)
    throw new Error(`optimizer-matrix with src needs rowMap, colMap, and select (${spec.src})`);
  const blob = JSON.parse(readFileSync(join(root, spec.src), "utf-8")) as {
    results: { op: string; optimizer: string; threshold: number | null; mse: number }[];
  };
  const cell = new Map(blob.results.map((r) => [`${r.op}|${r.optimizer}|${r.threshold}`, r.mse]));
  const ops = Object.keys(spec.rowMap);
  const opts = Object.keys(spec.colMap);
  const panels = spec.select.map(({ condition, label }) => ({
    label,
    cells: ops.map((op) =>
      opts.map((o) => {
        const mse = cell.get(`${op}|${o}|${condition}`);
        if (mse === undefined)
          throw new Error(`optimizer-matrix: no cell for ${op}/${o}/threshold=${condition} in ${spec.src}`);
        return Math.round(Math.log10(Math.max(mse, 1e-40)));
      }),
    ),
  }));
  return {
    ...spec,
    rows: Object.values(spec.rowMap),
    cols: Object.values(spec.colMap),
    panels,
  };
}

await buildNote({
  root,
  source: "paper/report.md",
  refs: "paper/refs.json",
  vocabulary: { parse: parseFigureSpec, render: renderFigure, resolve: resolveFigureData },
  styles: { web: "styles/web.css", pdf: "styles/pdf.css" },
  hydrate: "src/hydrate.ts",
  codeLanguages: ["python"],
  pdfName: "hill-space.pdf",
  toc: { web: "float" },
  site: "https://hillspace.justindujardin.com",
  // the page this site belongs to, linked above the masthead
  home: { label: "Justin DuJardin", href: "https://justindujardin.com" },
  // the field's words, marked in the prose with a plain description each
  glossary: "paper/glossary.yaml",
  colophon:
    "Math is cool, and I'm usually out of my depth. Corrections welcome.",
  noPdf: process.argv.includes("--no-pdf"),
});
