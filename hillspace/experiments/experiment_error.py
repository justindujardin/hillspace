import json
import os
import glob
import decimal
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
import torch

# Set high precision for ground truth calculations
getcontext().prec = 50


class ComprehensiveHillSpace:
    """Hill Space with all 4 implementation variants for error analysis."""

    def __init__(
        self, dtype=torch.float64, eps: float = 1e-7, target_range: float = 5.0
    ):
        self.dtype = dtype
        self.eps = eps
        self.target_range = target_range

        # Perfect analytical weights (bypass hill space entirely) - baseline method
        self.analytical_weights = {
            "add": torch.tensor([1.0, 1.0], dtype=dtype),
            "subtract": torch.tensor([1.0, -1.0], dtype=dtype),
            "multiply": torch.tensor([1.0, 1.0], dtype=dtype),
            "divide": torch.tensor([1.0, -1.0], dtype=dtype),
        }

    def analytical_primitive(
        self, inputs: torch.Tensor, operation: str
    ) -> torch.Tensor:
        """Method 1: Pure analytical weights (baseline)"""
        weights = self.analytical_weights[operation]

        if operation in ["add", "subtract"]:
            return torch.matmul(inputs, weights)
        elif operation in ["multiply", "divide"]:
            return torch.prod(torch.pow(inputs, weights.unsqueeze(0)), dim=1)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def complex_primitive(
        self, inputs: torch.Tensor, operation: str, use_complex128: bool = False
    ) -> torch.Tensor:
        """Method 2: Complex number conversion"""
        weights = self.analytical_weights[operation]

        if operation in ["add", "subtract"]:
            return torch.matmul(inputs, weights)
        elif operation in ["multiply", "divide"]:
            # Convert to complex to handle negative bases with fractional exponents
            complex_dtype = torch.complex128 if use_complex128 else torch.complex64
            x_complex = inputs.to(complex_dtype)
            # Compute x^w using complex arithmetic 
            powered = torch.pow(x_complex, weights.unsqueeze(0))
            # Take product across input dimensions
            result_complex = torch.prod(powered, dim=1)
            # Convert back to real
            return result_complex.real.to(self.dtype)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def logspace_primitive(self, inputs: torch.Tensor, operation: str) -> torch.Tensor:
        """Method 3: iNALU log-space computation"""
        weights = self.analytical_weights[operation]

        if operation in ["add", "subtract"]:
            return torch.matmul(inputs, weights)
        elif operation in ["multiply", "divide"]:
            # Log-space computation with stability
            x_abs = torch.abs(inputs) + self.eps
            log_x = torch.clamp(torch.log(x_abs), min=-10, max=10)
            result = torch.exp(torch.matmul(log_x, weights))
            result = torch.clamp(result, min=-1e6, max=1e6)
            return result
        else:
            raise ValueError(f"Unknown operation: {operation}")


def compute_ground_truth(inputs_np: np.ndarray, operation: str) -> np.ndarray:
    """Compute ground truth using high-precision Decimal arithmetic."""
    results = []

    for i in range(len(inputs_np)):
        try:
            x_val = float(np.float64(inputs_np[i, 0]))
            y_val = (
                float(np.float64(inputs_np[i, 1])) if inputs_np.shape[1] > 1 else 0.0
            )

            if not np.isfinite(x_val) or not np.isfinite(y_val):
                results.append(np.nan)
                continue

            x_dec = Decimal(str(x_val))
            y_dec = Decimal(str(y_val))

            if operation == "add":
                result = x_dec + y_dec
            elif operation == "subtract":
                result = x_dec - y_dec
            elif operation == "multiply":
                result = x_dec * y_dec
            elif operation == "divide":
                if abs(y_dec) > Decimal("1e-20"):
                    result = x_dec / y_dec
                else:
                    results.append(np.nan)
                    continue
            else:
                raise ValueError(f"Unknown operation: {operation}")

            result_float64 = float(result)
            if not np.isfinite(result_float64):
                results.append(np.nan)
            else:
                results.append(result_float64)

        except (decimal.InvalidOperation, decimal.Overflow, ValueError, OverflowError):
            results.append(np.nan)

    return np.array(results)


def generate_test_inputs(
    num_samples: int, dtype: torch.dtype, seed: int = 42
) -> torch.Tensor:
    """Generate test inputs from U(-1e4, 1e4) to match other experiments."""
    np.random.seed(seed)
    samples = np.random.uniform(-1e4, 1e4, (num_samples, 2))
    np.random.shuffle(samples)
    return torch.tensor(samples, dtype=dtype)


def analyze_method_errors(
    operation: str,
    dtype: torch.dtype,
    num_samples: int = 100_000_000,
    seed: int = 42,
    batch_size: int = 100_000,
) -> Dict[str, float]:
    """Analyze errors for all 4 methods against ground truth."""

    hill_space = ComprehensiveHillSpace(dtype)

    # Storage for all errors
    method_errors = {
        "analytical": {"squared_errors": [], "nan_count": 0, "inf_count": 0},
        "complex64": {"squared_errors": [], "nan_count": 0, "inf_count": 0},
        "complex128": {"squared_errors": [], "nan_count": 0, "inf_count": 0},
        "logspace": {"squared_errors": [], "nan_count": 0, "inf_count": 0},
    }

    total_processed = 0

    print(f"Processing {num_samples:,} samples for {operation} with {dtype}")

    for batch_start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - batch_start)
        batch_seed = seed + (batch_start // batch_size)

        if batch_start % (batch_size * 10) == 0:
            print(f"  Processed {batch_start:,}/{num_samples:,} samples...")

        # Generate batch
        batch_inputs = generate_test_inputs(current_batch_size, dtype, batch_seed)

        # Skip problematic cases for division
        if operation == "divide":
            mask = torch.abs(batch_inputs[:, 1]) > 1e-10
            batch_inputs = batch_inputs[mask]

        if len(batch_inputs) == 0:
            continue

        # Compute ground truth
        batch_inputs_np = batch_inputs.detach().cpu().numpy()
        ground_truth = compute_ground_truth(batch_inputs_np, operation)
        ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float64)

        # Compute results for all methods
        methods = {
            "analytical": lambda x: hill_space.analytical_primitive(x, operation),
            "complex64": lambda x: hill_space.complex_primitive(
                x, operation, use_complex128=False
            ),
            "complex128": lambda x: hill_space.complex_primitive(
                x, operation, use_complex128=True
            ),
            "logspace": lambda x: hill_space.logspace_primitive(x, operation),
        }

        for method_name, method_func in methods.items():
            try:
                result = method_func(batch_inputs)
                result_f64 = result.double()

                # Count NaN/Inf
                nan_mask = torch.isnan(result_f64)
                inf_mask = torch.isinf(result_f64)
                method_errors[method_name]["nan_count"] += nan_mask.sum().item()
                method_errors[method_name]["inf_count"] += inf_mask.sum().item()

                # Calculate squared errors (excluding NaN/Inf from ground truth)
                valid_ground_truth = torch.isfinite(ground_truth_tensor)
                valid_result = torch.isfinite(result_f64)
                valid_mask = valid_ground_truth & valid_result

                if valid_mask.sum() > 0:
                    squared_errors = (
                        result_f64[valid_mask] - ground_truth_tensor[valid_mask]
                    ) ** 2
                    method_errors[method_name]["squared_errors"].extend(
                        squared_errors.detach().cpu().numpy().astype(np.float64)
                    )

            except Exception as e:
                print(f"Error in {method_name}: {e}")
                # Count as NaN
                method_errors[method_name]["nan_count"] += len(batch_inputs)

        total_processed += len(batch_inputs)

    print(f"  Completed! Processed {total_processed:,} valid samples")

    # Calculate statistics for each method
    results = {}
    for method_name, errors in method_errors.items():
        if len(errors["squared_errors"]) > 0:
            squared_errors_np = np.array(errors["squared_errors"])

            results[method_name] = {
                "num_samples": len(squared_errors_np),
                "nan_count": errors["nan_count"],
                "inf_count": errors["inf_count"],
                "nan_rate": (
                    errors["nan_count"] / total_processed if total_processed > 0 else 0
                ),
                "inf_rate": (
                    errors["inf_count"] / total_processed if total_processed > 0 else 0
                ),
                "mean_squared_error": float(np.mean(squared_errors_np)),
                "max_squared_error": float(np.max(squared_errors_np)),
                "median_squared_error": float(np.median(squared_errors_np)),
                "q99_squared_error": float(np.percentile(squared_errors_np, 99)),
                "q99_9_squared_error": float(np.percentile(squared_errors_np, 99.9)),
                "q99_99_squared_error": float(np.percentile(squared_errors_np, 99.99)),
                "std_squared_error": float(np.std(squared_errors_np)),
            }
        else:
            results[method_name] = {
                "num_samples": 0,
                "nan_count": errors["nan_count"],
                "inf_count": errors["inf_count"],
                "nan_rate": (
                    errors["nan_count"] / total_processed if total_processed > 0 else 0
                ),
                "inf_rate": (
                    errors["inf_count"] / total_processed if total_processed > 0 else 0
                ),
                "mean_squared_error": float("nan"),
                "max_squared_error": float("nan"),
                "median_squared_error": float("nan"),
                "q99_squared_error": float("nan"),
                "q99_9_squared_error": float("nan"),
                "q99_99_squared_error": float("nan"),
                "std_squared_error": float("nan"),
            }

    # Add metadata
    final_results = {
        "operation": operation,
        "dtype": str(dtype),
        "total_samples_processed": total_processed,
        "methods": results,
    }

    return final_results


def run_single_comprehensive_analysis(run_params: Tuple[int, str, torch.dtype]) -> Dict:
    """Run comprehensive analysis for one operation/dtype combination."""
    run_id, operation, dtype = run_params
    delay = min(30, run_id * 5)  # Stagger starts by 2x their run_id in seconds
    # Sleep run_id seconds to stagger starts
    print(
        f"[{run_id:02d}] Preparing to run comprehensive analysis: {operation} with {dtype} in {delay}s"
    )
    import time

    time.sleep(delay)  # Stagger starts by 2x their run_id in seconds
    print(f"[{run_id:02d}] Starting comprehensive analysis: {operation} with {dtype}")

    try:
        start_time = time.time()

        stats = analyze_method_errors(
            operation=operation,
            dtype=dtype,
            seed=42 + run_id,
        )

        elapsed = time.time() - start_time
        print(f"[{run_id:02d}] Completed {operation} with {dtype} in {elapsed:.1f}s")

        stats["run_id"] = run_id
        stats["success"] = True
        stats["elapsed_time"] = elapsed

        # Save individual result
        result_file = f"results/comprehensive_analysis_{operation}_{str(dtype).replace('torch.', '')}.json"
        with open(result_file, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    except Exception as e:
        import traceback

        print(traceback.format_exc())

        elapsed = time.time() - start_time if "start_time" in locals() else 0
        error_msg = f"Error analyzing {operation} with {dtype}: {str(e)}"
        print(f"[{run_id:02d}] {error_msg}")

        error_result = {
            "run_id": run_id,
            "operation": operation,
            "dtype": str(dtype),
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed,
        }

        return error_result


def run_comprehensive_analysis(num_processes: int = None):
    """Run comprehensive error analysis across all methods."""

    if num_processes is None:
        num_processes = min(cpu_count(), 4)

    operations = ["add", "subtract", "multiply", "divide"]
    dtypes = [torch.float32, torch.float64]

    # Create parameter combinations
    run_params = []
    run_id = 0
    for dtype in dtypes:
        for operation in operations:
            run_params.append((run_id, operation, dtype))
            run_id += 1

    print(f"Running comprehensive analysis with {num_processes} processes")
    print(f"Total combinations: {len(run_params)}")
    print(f"Testing all 4 methods per combination")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Run analysis in parallel
    try:
        with Pool(processes=num_processes) as pool:
            result = pool.map_async(run_single_comprehensive_analysis, run_params)
            results = result.get(timeout=7200)  # 2 hour timeout
    except Exception as e:
        print(f"Multiprocessing error: {e}")
        results = []

    return results


def generate_comprehensive_error_table():
    """Generate publication-ready comprehensive error comparison tables."""

    # Load all results
    individual_files = glob.glob("results/comprehensive_analysis_*.json")
    if not individual_files:
        print("No comprehensive analysis results found! Run the analysis first.")
        return

    all_results = []
    for file in individual_files:
        try:
            with open(file, "r") as f:
                result = json.load(f)
                if result.get("success", False):
                    all_results.append(result)
        except:
            continue

    if not all_results:
        print("No successful results found!")
        return

    # Format function
    def format_error(value):
        if np.isnan(value) or value == 0.0:
            return "0.0"
        elif value < 1e-15:
            return f"{value:.1e}"
        else:
            return f"{value:.2e}"

    print("\n" + "=" * 100)
    print("HILL SPACE ERROR ANALYSIS - FLOATING-POINT BASELINE")
    print("=" * 100)
    print()

    # Table 1: Analytical baseline (floating-point precision limits)
    print("**Table 1: Floating-Point Precision Baseline**")
    print(
        "*100M+ samples per operation/dtype, analytical weights vs high-precision ground truth*"
    )
    print()

    print(
        "| Operation | Precision | Mean Squared Error | Max Error | 99.99%ile Error |"
    )
    print("|-----------|-----------|-------------------|-----------|-----------------|")

    for result in sorted(all_results, key=lambda x: (x["operation"], x["dtype"])):
        operation = result["operation"]
        dtype_str = "Float32" if "float32" in result["dtype"] else "Float64"

        analytical_stats = result["methods"]["analytical"]
        mean_error = format_error(analytical_stats["mean_squared_error"])
        max_error = format_error(analytical_stats["max_squared_error"])
        q99_99_error = format_error(analytical_stats["q99_99_squared_error"])

        print(
            f"| {operation:9} | {dtype_str:9} | {mean_error:>17} | {max_error:>9} | {q99_99_error:>15} |"
        )

    print()

    # Table 2: Method comparison (streamlined)
    print("**Table 2: Additional Error Beyond Floating-Point Baseline**")
    print(
        "*100M+ samples per operation/dtype/method, additional error introduced by each implementation method*"
    )
    print()

    print(
        "| Operation | Precision | Method | Additional MSE | Max Error | 99.99%ile Error |"
    )
    print(
        "|-----------|-----------|--------|----------------|-----------|-----------------|"
    )

    for result in sorted(all_results, key=lambda x: (x["operation"], x["dtype"])):
        operation = result["operation"]
        dtype_str = "Float32" if "float32" in result["dtype"] else "Float64"

        # Get analytical baseline
        analytical_stats = result["methods"]["analytical"]
        baseline_mse = analytical_stats["mean_squared_error"]

        # Compare streamlined methods (exclude complex64 and normalized)
        methods_to_compare = [
            ("Complex128", "complex128"),
            ("Log-space", "logspace"),
        ]

        for method_display, method_key in methods_to_compare:
            if method_key not in result["methods"]:
                continue

            method_stats = result["methods"][method_key]

            # Calculate additional error
            method_mse = method_stats["mean_squared_error"]
            additional_mse = (
                method_mse - baseline_mse if not np.isnan(method_mse) else float("nan")
            )

            additional_mse_str = format_error(additional_mse)
            max_error_str = format_error(method_stats["max_squared_error"])
            q99_99_str = format_error(method_stats["q99_99_squared_error"])

            print(
                f"| {operation:9} | {dtype_str:9} | {method_display:8} | {additional_mse_str:>14} | {max_error_str:>9} | {q99_99_str:>15} |"
            )

    print()
    print("**Key Findings:**")
    print()

    # Generate summary insights for multiplication (the problematic operation)
    for result in all_results:
        operation = result["operation"]
        if operation == "multiply":
            print(f"*{operation.title()} Operation Analysis:*")

            analytical_mse = result["methods"]["analytical"]["mean_squared_error"]
            dtype_str = "Float32" if "float32" in result["dtype"] else "Float64"

            for method_name, method_key in [
                ("Complex128", "complex128"),
                ("Log-space", "logspace"),
            ]:
                if method_key in result["methods"]:
                    method_stats = result["methods"][method_key]
                    additional_error = (
                        method_stats["mean_squared_error"] - analytical_mse
                    )

                    if analytical_mse != 0:
                        ratio = additional_error / analytical_mse
                        ratio_str = (
                            f" ({ratio:.1e}x baseline)"
                            if abs(ratio) > 1e-10
                            else " (≈0x baseline)"
                        )
                    else:
                        ratio_str = ""

                    print(
                        f"- {dtype_str} {method_name}: {format_error(additional_error)} additional MSE{ratio_str}"
                    )
            print()

    print("**Methodology:**")
    print("- Ground truth computed with 50-digit precision Decimal arithmetic")
    print("- Additional error = Method MSE - Analytical MSE (floating-point baseline)")
    print("- Input range: U(-1e4, 1e4) matching extreme extrapolation experiments")
    print("- Complex128 uses 128-bit complex arithmetic (two 64-bit floats)")
    print(
        "- Log-space uses iNALU-style log/exp transformations with stability clamping"
    )

    # Save detailed results
    summary_data = []
    for result in all_results:
        analytical_baseline = result["methods"]["analytical"]["mean_squared_error"]

        # Include analytical baseline
        analytical_stats = result["methods"]["analytical"]
        summary_data.append(
            {
                "operation": result["operation"],
                "dtype": result["dtype"],
                "method": "analytical",
                "additional_mse": 0.0,
                "absolute_mse": analytical_stats["mean_squared_error"],
                "max_error": analytical_stats["max_squared_error"],
                "q99_99_error": analytical_stats["q99_99_squared_error"],
                "num_samples": analytical_stats["num_samples"],
            }
        )

        # Include comparison methods
        for method_name in ["complex128", "logspace"]:
            if method_name in result["methods"]:
                method_stats = result["methods"][method_name]
                additional_error = (
                    method_stats["mean_squared_error"] - analytical_baseline
                )

                summary_data.append(
                    {
                        "operation": result["operation"],
                        "dtype": result["dtype"],
                        "method": method_name,
                        "additional_mse": additional_error,
                        "absolute_mse": method_stats["mean_squared_error"],
                        "max_error": method_stats["max_squared_error"],
                        "q99_99_error": method_stats["q99_99_squared_error"],
                        "num_samples": method_stats["num_samples"],
                    }
                )

    df = pd.DataFrame(summary_data)
    df.to_csv("results/comprehensive_error_analysis.csv", index=False)
    print(f"\nDetailed results saved to: results/comprehensive_error_analysis.csv")


if __name__ == "__main__":
    # Run comprehensive analysis
    print("Starting comprehensive Hill Space error analysis...")
    results = run_comprehensive_analysis(num_processes=2)

    print("\nGenerating comprehensive error table...")
    generate_comprehensive_error_table()
