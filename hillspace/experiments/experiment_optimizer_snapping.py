"""Does snapping make Hill Space converge under ANY optimizer, where plain hill
is hit-and-miss?

Matched grid: same optimizers x same ops, snapping OFF vs snapping ON. The only
variable is snapping. Persuasive if the plain-hill column is a patchwork of fails
and the snapped column is uniformly at the floor.

IMPORTANT: this uses the REAL MathyUnit model and the REAL data pipeline (the same
code the paper's other experiments use), NOT a from-scratch reimplementation, so
there is no risk of a primitive being subtly wrong. Snapping is controlled by the
model's space: "hill" (plain) vs "hill_snap" (snapped). Training always uses plain
gradient flow; snapping is applied only at EVAL by swapping the space, so the hard
snap threshold never kills a training gradient.

Snapping uses the model's default threshold (SNAP_THRESHOLD = 1e-2). This is
derived from what optimizers actually reach: a converged weight lands within ~2e-3
of a target, while the stable selections {-1,0,+1} are ~0.5-1.0 apart, so 1e-2
catches any converged weight and can never reach a neighbor.

Usage:
    uv run python -m hillspace.experiments.experiment_optimizer_snapping
"""
import json
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..model import MathyUnit, SNAP_THRESHOLD
from ..dataset.operator_dataset import create_mathy_dataloaders
from ..dataset.operator_specs import OPERATION_REGISTRY

RESULTS_DIR = Path("results/optimizer_snapping")
RAW_JSON = RESULTS_DIR / "optimizer_snapping_raw.json"
SUMMARY_CSV = RESULTS_DIR / "optimizer_snapping_summary.csv"

OPS = ["add", "subtract", "multiply", "divide", "identity", "reciprocal", "sin", "cos_add"]
# Uses the model's default snap threshold (SNAP_THRESHOLD = 1e-2). See model.py for
# the safe-band rationale (converged weights land within ~2e-3 of a target; the
# stable selections are ~0.5-1.0 apart, so 1e-2 catches any converged weight and
# can never reach a neighbor).
EPOCHS = 100

# optimizer name -> factory. lr chosen per-family to be a reasonable default.
# RMSProp joins the sweep because iNALU trains with it (nalu_syn_simple_arith.py
# line 151: RMSPropOptimizer lr=0.01); alpha=0.9 mirrors TF's decay default.
OPTIMIZERS = {
    "Adam(def)": lambda p: torch.optim.Adam(p, lr=0.1),
    "Adam(β₂=0.5)": lambda p: torch.optim.Adam(p, lr=0.1, betas=(0.9, 0.5)),
    "AdamW": lambda p: torch.optim.AdamW(p, lr=0.1),
    "RAdam": lambda p: torch.optim.RAdam(p, lr=0.1),
    "NAdam": lambda p: torch.optim.NAdam(p, lr=0.1),
    "RMSProp": lambda p: torch.optim.RMSprop(p, lr=0.1, alpha=0.9),
    "Adagrad": lambda p: torch.optim.Adagrad(p, lr=0.3),
    "Adadelta": lambda p: torch.optim.Adadelta(p, lr=1.0),
    "Rprop": lambda p: torch.optim.Rprop(p, lr=0.01),
    "SGD+mom": lambda p: torch.optim.SGD(p, lr=0.3, momentum=0.9),
}


# Snap conditions swept: None = plain hill, then eval-snap at two thresholds.
# 1e-6 is where snapping started historically — right at the edge where default
# optimizers park (~1e-6 short of a target), below the safe band (~3e-3..5e-2),
# so it catches some runs and misses others. 1e-2 is the derived default inside
# the band. The figure reads: patchwork -> lighter patchwork -> clean sweep.
SNAP_CONDITIONS = (None, 1e-6, SNAP_THRESHOLD)


def run(op, opt_name, threshold, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    spec = OPERATION_REGISTRY[op]
    train_loader, _ = create_mathy_dataloaders(
        operator_spec=spec, train_samples=64_000, test_samples=1000, batch_size=64,
        train_range=(1e-8, 10.0), test_range=(1e-8, 10.0),
        distribution="uniform", dtype=torch.float64, seed=seed)
    _, test_loader = create_mathy_dataloaders(
        operator_spec=spec, train_samples=1000, test_samples=16_000, batch_size=64,
        train_range=(1e-8, 10.0), test_range=(-1e4, 1e4),
        distribution="uniform", dtype=torch.float64, seed=seed)

    device = "cpu"
    # Train with PLAIN hill (standard gradient flow, no snapping).
    model = MathyUnit(input_size=2, output_size=1, dtype=torch.float64,
                      space="hill", init_scale=0.0).to(device)
    opt = OPTIMIZERS[opt_name](model.get_arithmetic_parameters())
    crit = nn.MSELoss()
    for _ in range(EPOCHS):
        for bx, by in train_loader:
            opt.zero_grad()
            crit(model.forward(bx.to(device), op), by.to(device)).backward()
            opt.step()

    # Eval: optionally switch to snapped space (weights unchanged, just snapped).
    if threshold is not None:
        model.space = "hill_snap"
        model.snap_threshold = threshold
    errs = []
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model.forward(bx.to(device), op)
            errs.append(((pred - by.to(device)) ** 2).mean().item())
    return float(np.mean(errs))


def job(args):
    op, opt_name, threshold = args
    return {
        "op": op,
        "optimizer": opt_name,
        "threshold": threshold,
        "mse": run(op, opt_name, threshold),
    }


def make_data():
    """Run the full grid and write raw per-cell results to results/. Slow.
    Comment this out in the footer after one run to re-aggregate without re-running."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(OPTIMIZERS.keys())
    jobs = [(op, n, thr) for thr in SNAP_CONDITIONS for n in names for op in OPS]
    print(
        f"Running {len(jobs)} cells "
        f"({len(OPS)} ops x {len(names)} optimizers x {len(SNAP_CONDITIONS)} snap conditions)..."
    )
    with Pool(min(cpu_count(), 12)) as pool:
        results = pool.map(job, jobs)
    with open(RAW_JSON, "w") as f:
        json.dump(
            {"epochs": EPOCHS, "snap_conditions": list(SNAP_CONDITIONS), "results": results},
            f,
            indent=2,
        )
    print(f"Raw results -> {RAW_JSON}")


def aggregate_results():
    """Read raw results, print one table per snap condition, and write a tidy CSV. Fast."""
    with open(RAW_JSON) as f:
        blob = json.load(f)
    results = blob["results"]
    names = list(OPTIMIZERS.keys())
    cell = {(r["op"], r["optimizer"], r["threshold"]): r["mse"] for r in results}

    print(f"Optimizer x snapping grid (extrap MSE, {blob['epochs']} epochs, real MathyUnit)")
    print("'!' = fail (>1e-2), '?' = marginal, bare = converged to floor.")

    def render(threshold):
        tag = "SNAP OFF (plain hill)" if threshold is None else f"SNAP ON (eval, thr={threshold:.0e})"
        print(f"\n=== {tag} ===")
        print(f"{'op':>10} | " + " | ".join(f"{n:>10}" for n in names))
        print("-" * (12 + 13 * len(names)))
        fails = {n: 0 for n in names}
        for op in OPS:
            row = []
            for n in names:
                m = cell[(op, n, threshold)]
                mark = "" if m < 1e-6 else ("?" if m < 1e-2 else "!")
                row.append(f"{m:.0e}{mark}")
                if m >= 1e-2:
                    fails[n] += 1
            print(f"{op:>10} | " + " | ".join(f"{c:>10}" for c in row))
        print("-" * (12 + 13 * len(names)))
        print(f"{'FAILS':>10} | " + " | ".join(f"{fails[n]:>10}" for n in names))

    for thr in blob["snap_conditions"]:
        render(thr)

    # Tidy long-form CSV: one row per (op, optimizer, threshold).
    df = pd.DataFrame(results)[["op", "optimizer", "threshold", "mse"]]
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary CSV -> {SUMMARY_CSV}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    make_data()          # comment out after first run to re-aggregate without re-running
    aggregate_results()
