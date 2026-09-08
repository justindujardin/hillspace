"""Floating-point error floors for exact hill-space selections, paper §4.4.

The question: once a unit's selection is EXACTLY [1, ±1], what error remains,
and where does it come from? Each operation/dtype is evaluated through up to
four pathways against high-precision ground truth:

  native      -- the literal IEEE op (x+y, x-y, x*y, x/y) in the working
                 dtype. This is THE floating-point floor: a correctly rounded
                 op sits within 0.5 ulp of real arithmetic by construction.
  analytical  -- the hill-space formulation with weights fixed at the exact
                 selection: matmul for add/sub, prod(pow(x, w)) for mul/div.
                 Measures what the formulation adds over the native op.
                 (Verified bitwise-identical to native for add/sub/mul on
                 torch 2.7 CPU; divide computes x*(1/y), two roundings
                 instead of one. The JSON records the equality fraction.)
  complex128  -- exponential-primitive stabilization via complex arithmetic,
                 cast back to the working dtype. Multiply/divide only.
  logspace    -- exponential-primitive stabilization via log/exp using
                 iNALU's published guards (magnitude floor eps=1e-7,
                 exponent cap omega=20 [@schlor2020]) plus exact sign
                 recovery: sign(x)*sign(y), exact for weights ±1. iNALU must
                 LEARN its sign mechanism; recovering it exactly here
                 isolates the precision cost of the log/exp round trip from
                 that separate hazard. Multiply/divide only.

Ground truth and metric (v2, 2026-07 rework):
  The first version of this experiment had two measurement artifacts.
  Decimal(str(x)) perturbed each input by up to half an ulp, so the float64
  "floor" rows reported decimal-string round-trip noise; and rounding the
  exact result to float64 before differencing makes any correctly rounded
  float64 op measure exactly zero. v2 converts inputs exactly (Decimal(x)
  is exact for binary floats), computes the reference in 50-digit Decimal,
  and carries it as a two-term value gt64 + resid, where gt64 is the
  reference rounded to float64 and resid the exact remainder. Per-sample
  error is then (result - gt64) - resid in float64 arithmetic, accurate to
  ~1 part in 1e16 of the error itself and non-degenerate at float64.

Inputs: U(-1e4, 1e4)^2, the extreme extrapolation range of §4.3; divide
masks |y| > 1e-10. Seeding: SeedSequence(20260724, config_index, batch).
CPU only. The Decimal loop dominates: expect roughly 10-30 minutes per
config at 100M samples, with the 8 configs spread over --processes workers.
Memory is ~3 GB per worker at 100M samples (squared errors are kept for
exact percentiles).

A second mode (--sweep) asks the scale question instead of the precision
question: log-uniform magnitudes spanning the dtype's whole normal range
(random sign, exponent uniform over the format's normal exponents). MSE is
meaningless there — squared errors at 1e300 scale overflow float64 — so
each pathway is scored against the native op in the format's own currency:
bitwise agreement and ulp distance, with non-finite results counted on
both sides. No Decimal reference is involved, so the sweep runs in minutes.
"""

import argparse
import glob
import json
import os
import time
from decimal import Decimal, getcontext
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

getcontext().prec = 50

SEED_ROOT = 20260724
INPUT_RANGE = 1e4
DIVIDE_Y_MASK = 1e-10
LOGSPACE_EPS = 1e-7  # iNALU magnitude floor
LOGSPACE_OMEGA = 20.0  # iNALU exponent cap

OP_ORDER = ["add", "subtract", "multiply", "divide"]
WEIGHTS = {
    "add": [1.0, 1.0],
    "subtract": [1.0, -1.0],
    "multiply": [1.0, 1.0],
    "divide": [1.0, -1.0],
}
RESULT_GLOB = "results/error_floor_*.json"
SWEEP_GLOB = "results/error_sweep_*.json"
# smallest/largest normal binary exponents per dtype (subnormal inputs and
# overflow are then produced by the operations themselves, not the sampler)
SWEEP_EXPONENTS = {torch.float32: (-126.0, 127.0), torch.float64: (-1022.0, 1023.0)}


def native_op(inputs: torch.Tensor, operation: str) -> torch.Tensor:
    x, y = inputs[:, 0], inputs[:, 1]
    if operation == "add":
        return x + y
    if operation == "subtract":
        return x - y
    if operation == "multiply":
        return x * y
    return x / y


def analytical_primitive(inputs: torch.Tensor, operation: str) -> torch.Tensor:
    weights = torch.tensor(WEIGHTS[operation], dtype=inputs.dtype)
    if operation in ("add", "subtract"):
        return torch.matmul(inputs, weights)
    return torch.prod(torch.pow(inputs, weights.unsqueeze(0)), dim=1)


def complex_primitive(inputs: torch.Tensor, operation: str) -> torch.Tensor:
    weights = torch.tensor(WEIGHTS[operation], dtype=torch.complex128)
    powered = torch.pow(inputs.to(torch.complex128), weights.unsqueeze(0))
    return torch.prod(powered, dim=1).real.to(inputs.dtype)


def logspace_primitive(inputs: torch.Tensor, operation: str) -> torch.Tensor:
    weights = torch.tensor(WEIGHTS[operation], dtype=inputs.dtype)
    log_mag = torch.log(torch.clamp(torch.abs(inputs), min=LOGSPACE_EPS))
    exponent = torch.clamp(torch.matmul(log_mag, weights), max=LOGSPACE_OMEGA)
    # sign(x)^w for w = ±1 is sign(x) either way, so the exact sign of the
    # result is the plain product of input signs
    sign = torch.prod(torch.sign(inputs), dim=1)
    return sign * torch.exp(exponent)


def generate_batch(
    num_samples: int, dtype: torch.dtype, config_index: int, batch_index: int
) -> torch.Tensor:
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED_ROOT, config_index, batch_index])
    )
    samples = rng.uniform(-INPUT_RANGE, INPUT_RANGE, (num_samples, 2))
    return torch.tensor(samples, dtype=dtype)


def ground_truth_two_term(
    inputs: np.ndarray, operation: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact reference per sample as gt64 + resid.

    Decimal(x) is an exact conversion for binary floats. gt64 is the exact
    result rounded once to float64; resid = exact - gt64 satisfies
    |resid| <= 0.5 ulp(gt64), so (r - gt64) - resid recovers a method's true
    error to ~1e-16 relative accuracy without keeping Decimals around.
    """
    n = len(inputs)
    gt64 = np.empty(n)
    resid = np.empty(n)
    for i in range(n):
        x = Decimal(float(inputs[i, 0]))
        y = Decimal(float(inputs[i, 1]))
        if operation == "add":
            exact = x + y
        elif operation == "subtract":
            exact = x - y
        elif operation == "multiply":
            exact = x * y
        else:
            exact = x / y
        g = float(exact)
        gt64[i] = g
        resid[i] = float(exact - Decimal(g)) if np.isfinite(g) else np.nan
    return gt64, resid


def methods_for(operation: str) -> Dict[str, Callable[[torch.Tensor, str], torch.Tensor]]:
    methods = {"native": native_op, "analytical": analytical_primitive}
    if operation in ("multiply", "divide"):
        methods["complex128"] = complex_primitive
        methods["logspace"] = logspace_primitive
    return methods


def analyze_config(
    config_index: int,
    operation: str,
    dtype: torch.dtype,
    num_samples: int,
    batch_size: int,
) -> Dict[str, Any]:
    tag = f"{operation}/{str(dtype).replace('torch.', '')}"
    methods = methods_for(operation)

    squared = {name: np.empty(num_samples) for name in methods}
    counts = {name: {"n": 0, "nan": 0, "inf": 0} for name in methods}
    native_equal = {name: 0 for name in methods}
    total_valid = 0
    start = time.time()

    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_index in range(num_batches):
        n = min(batch_size, num_samples - batch_index * batch_size)
        batch = generate_batch(n, dtype, config_index, batch_index)
        if operation == "divide":
            batch = batch[torch.abs(batch[:, 1]) > DIVIDE_Y_MASK]
        if len(batch) == 0:
            continue

        gt64, resid = ground_truth_two_term(batch.double().numpy(), operation)
        gt_valid = np.isfinite(gt64) & np.isfinite(resid)
        total_valid += int(gt_valid.sum())

        native_result = None
        for name, fn in methods.items():
            result = fn(batch, operation).double().numpy()
            if name == "native":
                native_result = result
            counts[name]["nan"] += int(np.isnan(result).sum())
            counts[name]["inf"] += int(np.isinf(result).sum())
            native_equal[name] += int((result == native_result).sum())

            valid = gt_valid & np.isfinite(result)
            err = (result[valid] - gt64[valid]) - resid[valid]
            block = err * err
            k = counts[name]["n"]
            squared[name][k : k + len(block)] = block
            counts[name]["n"] = k + len(block)

        if (batch_index + 1) % 25 == 0 or batch_index + 1 == num_batches:
            elapsed = time.time() - start
            print(
                f"[{tag}] batch {batch_index + 1}/{num_batches} ({elapsed:.0f}s)",
                flush=True,
            )

    stats = {}
    for name in methods:
        se = squared[name][: counts[name]["n"]]
        if len(se) == 0:
            stats[name] = {"num_samples": 0}
            continue
        stats[name] = {
            "num_samples": int(len(se)),
            "nan_count": counts[name]["nan"],
            "inf_count": counts[name]["inf"],
            "bitwise_equal_native_frac": native_equal[name] / max(total_valid, 1),
            "mse": float(np.mean(se)),
            "median_se": float(np.median(se)),
            "q99_se": float(np.percentile(se, 99)),
            "q99_99_se": float(np.percentile(se, 99.99)),
            "max_se": float(np.max(se)),
        }

    return {
        "operation": operation,
        "dtype": str(dtype),
        "total_samples": total_valid,
        "elapsed_s": round(time.time() - start, 1),
        "seed_root": SEED_ROOT,
        "config_index": config_index,
        "methods": stats,
    }


def run_config(params: Tuple[int, str, torch.dtype, int, int]) -> Dict[str, Any]:
    config_index, operation, dtype, num_samples, batch_size = params
    tag = f"{operation}/{str(dtype).replace('torch.', '')}"
    print(f"[{tag}] starting: {num_samples:,} samples", flush=True)
    result = analyze_config(config_index, operation, dtype, num_samples, batch_size)
    path = f"results/error_floor_{operation}_{str(dtype).replace('torch.', '')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] done in {result['elapsed_s']}s -> {path}", flush=True)
    return result


def format_se(value: float) -> str:
    if value == 0.0:
        return "0.0"
    return f"{value:.1e}" if abs(value) < 1e-15 else f"{value:.2e}"


def load_results() -> List[Dict[str, Any]]:
    results = []
    for path in glob.glob(RESULT_GLOB):
        with open(path) as f:
            results.append(json.load(f))
    order = {op: i for i, op in enumerate(OP_ORDER)}
    results.sort(key=lambda r: (order[r["operation"]], r["dtype"]))
    return results


def print_tables() -> None:
    results = load_results()
    if not results:
        print(f"No results matching {RESULT_GLOB}; run the analysis first.")
        return

    print("\n**Table 4.4.1: Floating-Point Precision Baseline (native IEEE ops)**")
    print("| Operation | Precision | Mean Squared Error | Max Error | 99.99%ile Error |")
    print("| --------- | --------- | ------------------ | --------- | --------------- |")
    for r in results:
        m = r["methods"]["native"]
        dtype = "Float32" if "float32" in r["dtype"] else "Float64"
        print(
            f"| {r['operation']} | {dtype} | {format_se(m['mse'])} "
            f"| {format_se(m['max_se'])} | {format_se(m['q99_99_se'])} |"
        )

    print("\n**Table 4.4.2: Additional MSE Beyond the Native Floor**")
    print("| Operation | Precision | Method | Additional MSE | Max Error | 99.99%ile Error |")
    print("| --------- | --------- | ------ | -------------- | --------- | --------------- |")
    label = {"analytical": "Real", "complex128": "Complex128", "logspace": "Log-space"}
    for r in results:
        baseline = r["methods"]["native"]["mse"]
        dtype = "Float32" if "float32" in r["dtype"] else "Float64"
        for key in ("analytical", "complex128", "logspace"):
            if key not in r["methods"]:
                continue
            m = r["methods"][key]
            print(
                f"| {r['operation']} | {dtype} | {label[key]} "
                f"| {format_se(m['mse'] - baseline)} | {format_se(m['max_se'])} "
                f"| {format_se(m['q99_99_se'])} |"
            )

    print("\nBitwise agreement with the native op (fraction of samples):")
    for r in results:
        dtype = "Float32" if "float32" in r["dtype"] else "Float64"
        parts = [
            f"{name}={m['bitwise_equal_native_frac']:.4f}"
            for name, m in r["methods"].items()
            if name != "native" and m.get("num_samples")
        ]
        print(f"  {r['operation']:9s} {dtype}: {', '.join(parts)}")


def generate_sweep_batch(
    num_samples: int, dtype: torch.dtype, config_index: int, batch_index: int
) -> torch.Tensor:
    """Log-uniform magnitudes across the dtype's full normal range."""
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED_ROOT, 1, config_index, batch_index])
    )
    lo, hi = SWEEP_EXPONENTS[dtype]
    exponent = rng.uniform(lo, hi, (num_samples, 2))
    sign = rng.integers(0, 2, (num_samples, 2)) * 2 - 1
    return torch.tensor(sign * np.exp2(exponent), dtype=dtype)


def float_ordinal(values: np.ndarray) -> np.ndarray:
    """Map floats to integers so consecutive representable values are
    consecutive integers (two's-complement trick); ulp distance becomes a
    subtraction. ±0 share an ordinal."""
    itype = np.int32 if values.dtype == np.float32 else np.int64
    ordinals = values.view(itype).astype(np.int64)
    negative = ordinals < 0
    ordinals[negative] = np.int64(np.iinfo(itype).min) - ordinals[negative]
    return ordinals


def sweep_config(
    config_index: int,
    operation: str,
    dtype: torch.dtype,
    num_samples: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Score every pathway against the native op across the full range.

    Divide is unmasked here — tiny divisors and the over/underflow they
    cause are the point. Ulp distances are exact below 2^53 and
    approximate above (float64 differencing of ordinals)."""
    tag = f"{operation}/{str(dtype).replace('torch.', '')}"
    methods = {k: v for k, v in methods_for(operation).items() if k != "native"}

    agree = {name: 0 for name in methods}
    nonfinite = {name: 0 for name in methods}
    ulps: Dict[str, list] = {name: [] for name in methods}
    nonfinite_native = 0
    start = time.time()

    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_index in range(num_batches):
        n = min(batch_size, num_samples - batch_index * batch_size)
        batch = generate_sweep_batch(n, dtype, config_index, batch_index)
        r_nat = native_op(batch, operation).numpy()
        nat_ord = float_ordinal(r_nat).astype(np.float64)
        nat_finite = np.isfinite(r_nat)
        nonfinite_native += int((~nat_finite).sum())

        for name, fn in methods.items():
            r = fn(batch, operation).numpy()
            agree[name] += int(
                ((r == r_nat) | (np.isnan(r) & np.isnan(r_nat))).sum()
            )
            nonfinite[name] += int((~np.isfinite(r)).sum())
            both = nat_finite & np.isfinite(r)
            d = np.abs(float_ordinal(r[both]).astype(np.float64) - nat_ord[both])
            ulps[name].append(d.astype(np.float32))

        if (batch_index + 1) % 25 == 0 or batch_index + 1 == num_batches:
            print(f"[sweep {tag}] batch {batch_index + 1}/{num_batches} "
                  f"({time.time() - start:.0f}s)", flush=True)

    stats: Dict[str, Any] = {}
    for name in methods:
        d = np.concatenate(ulps[name]) if ulps[name] else np.empty(0, np.float32)
        stats[name] = {
            "bitwise_frac": agree[name] / num_samples,
            "nonfinite_frac": nonfinite[name] / num_samples,
            "both_finite": int(len(d)),
            "ulp_le1_frac": float((d <= 1).mean()) if len(d) else float("nan"),
            "ulp_q99_99": float(np.percentile(d, 99.99)) if len(d) else float("nan"),
            "ulp_max": float(np.max(d)) if len(d) else float("nan"),
        }

    return {
        "operation": operation,
        "dtype": str(dtype),
        "num_samples": num_samples,
        "nonfinite_native_frac": nonfinite_native / num_samples,
        "elapsed_s": round(time.time() - start, 1),
        "seed_root": SEED_ROOT,
        "config_index": config_index,
        "methods": stats,
    }


def print_sweep_tables() -> None:
    results = []
    for path in glob.glob(SWEEP_GLOB):
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        print(f"No results matching {SWEEP_GLOB}; run with --sweep first.")
        return
    order = {op: i for i, op in enumerate(OP_ORDER)}
    results.sort(key=lambda r: (order[r["operation"]], r["dtype"]))

    label = {"analytical": "Hill Space", "complex128": "Complex128", "logspace": "Log-space"}
    print("\n**Full-Range Agreement Sweep (log-uniform over the dtype's normal range)**")
    print("| Operation | Precision | Method | Bitwise = native | ≤1 ulp | 99.99%ile ulp | Max ulp | Non-finite (method / native) |")
    print("| --------- | --------- | ------ | ---------------- | ------ | ------------- | ------- | ---------------------------- |")
    for r in results:
        dtype = "Float32" if "float32" in r["dtype"] else "Float64"
        for key in ("analytical", "complex128", "logspace"):
            if key not in r["methods"]:
                continue
            m = r["methods"][key]
            print(
                f"| {r['operation']} | {dtype} | {label[key]} "
                f"| {m['bitwise_frac'] * 100:.4f}% | {m['ulp_le1_frac'] * 100:.4f}% "
                f"| {m['ulp_q99_99']:.3g} | {m['ulp_max']:.3g} "
                f"| {m['nonfinite_frac'] * 100:.3f}% / {r['nonfinite_native_frac'] * 100:.3f}% |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="regenerate tables from existing results without recomputing",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="full-dynamic-range agreement sweep (log-uniform exponents, ulp metrics)",
    )
    args = parser.parse_args()

    if args.sweep:
        os.makedirs("results", exist_ok=True)
        configs = [
            (i, operation, dtype)
            for i, (dtype, operation) in enumerate(
                (d, o) for d in (torch.float32, torch.float64) for o in OP_ORDER
            )
        ]
        for config_index, operation, dtype in configs:
            result = sweep_config(
                config_index, operation, dtype, args.samples, args.batch_size
            )
            path = (
                f"results/error_sweep_{operation}_"
                f"{str(dtype).replace('torch.', '')}.json"
            )
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        print_sweep_tables()
        return

    if not args.tables_only:
        os.makedirs("results", exist_ok=True)
        configs = [
            (i, operation, dtype, args.samples, args.batch_size)
            for i, (dtype, operation) in enumerate(
                (d, o) for d in (torch.float32, torch.float64) for o in OP_ORDER
            )
        ]
        print(f"{len(configs)} configs, {args.samples:,} samples each, "
              f"{args.processes} processes", flush=True)
        with Pool(processes=args.processes) as pool:
            pool.map(run_config, configs)

    print_tables()


if __name__ == "__main__":
    main()
