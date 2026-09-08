# Hill Space

Source for the report _Stable Because Stuck: NALU Selects, It Doesn't Compute_: the text, the interactive figures, and the code behind the experiments.

Read it at https://hillspace.justindujardin.com. The PDF is linked from the page.

## The report

The report is one markdown file, `paper/report.md`. A build turns it into the web page, with figures you can drag, and the print PDF, from the same source. The build tool is [penname](https://github.com/justindujardin/penname).

```bash
npm install
npm run build        # dist/index.html and dist/hill-space.pdf
npm run build:web    # the page only, no WeasyPrint needed
npm run dev          # rebuild on save, served at http://localhost:8787
```

The PDF is rendered by WeasyPrint, which the build starts through `uv run --with weasyprint`, so `uv` has to be on your path for the full build.

Pushes to `main` deploy to Netlify through the workflow in `.github/workflows/deploy.yml`, which builds both artifacts on a runner that has WeasyPrint's libraries. A pull request gets its own preview, linked from a comment on the PR.

The figures live in `src/`. `spec.ts` names the kinds a figure fence may declare, `figures.ts` renders each kind to SVG, `hydrate.ts` adds the sliders in the browser, and `hillspace.ts` is the constraint and the primitives as plain functions, shared by both. The two optimizer tables are not typed in; `tools/build.ts` reads them from
`results/optimizer_snapping/optimizer_snapping_raw.json` at build time, so rerunning that experiment and rebuilding updates them.

## The experiments

Python 3.10 or newer, managed with uv:

```bash
uv sync
```

Each script matches a section of the report.

```bash
# 4.1 enumerated weights, no training
uv run python -m hillspace.experiments.experiment_neural_calc 1339.7364 - 2.7364

# 4.2 division learned in about a minute on a CPU
uv run python -m hillspace.experiments.experiment_train_division

# 2.2 the optimizer sweep behind the two matrix figures
uv run python -m hillspace.experiments.experiment_optimizer_snapping

# 4.3 iNALU comparison, 10 runs
uv run python -m hillspace.experiments.experiment_inalu

# 4.4 error floors, 100M samples per operation; hours of CPU
uv run python -m hillspace.experiments.experiment_error
uv run python -m hillspace.experiments.experiment_error --tables-only

# 4.5 initialization scales, 10 runs
uv run python -m hillspace.experiments.experiment_init
```

The runs that produced the tables in the report are checked in under `results/`, and the aggregation step of each script reads from there. The sweep, iNALU, and initialization scripts retrain before they aggregate; comment out the run call at the bottom of the file to rebuild the tables from the saved results alone.

`hillspace/train.py` is the general trainer I used while working things out. It trains every arithmetic and trigonometric operator in the report with snapping and logs to Weights and Biases. Nothing in the report depends on it, but it is where the model and dataset code get exercised.

## License

Code is MIT, see LICENSE.md. The report and its figures are CC BY 4.0.

## Citation

```bibtex
@misc{dujardin2026hillspace,
  author = {DuJardin, Justin},
  title = {Stable Because Stuck: NALU Selects, It Doesn't Compute},
  year = {2026},
  howpublished = {\url{https://hillspace.justindujardin.com}},
  note = {Supersedes the 2025 TechRxiv preprint, doi:10.36227/techrxiv.175339930.03949307/v2}
}
```
