import csv
import glob
import json
import os
import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree

from ..dataset.operator_dataset import create_mathy_dataloaders
from ..dataset.operator_specs import OPERATION_REGISTRY
from ..model import MathyUnit


def periodic_eval(
    model: MathyUnit,
    test_loaders: Dict[str, DataLoader],
    operations: List[str],
    device: str,
    converged_loss: float = 1e-10,
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Run evaluation on test set and return metrics and convergence status."""
    model.eval()
    eval_losses = {op: {"math": 0.0, "mae": 0.0, "mse": 0.0} for op in operations}

    with torch.no_grad():
        for operation in operations:
            if operation not in test_loaders:
                continue

            test_loader = test_loaders[operation]
            total_math_loss = 0.0
            total_mae = 0.0
            total_mse = 0.0
            num_batches = 0

            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Get predictions
                final_result = model.forward(batch_x, operation)

                # Calculate losses same way as training
                math_loss = torch.pow(final_result - batch_y, 2).mean()  # MSE

                # MAE and MSE on original scale
                mae = torch.abs(final_result - batch_y).mean()
                mse = torch.pow(final_result - batch_y, 2).mean()

                total_math_loss += math_loss.item()
                total_mae += mae.item()
                total_mse += mse.item()
                num_batches += 1

            if num_batches > 0:
                eval_losses[operation]["math"] = total_math_loss / num_batches
                eval_losses[operation]["mae"] = total_mae / num_batches
                eval_losses[operation]["mse"] = total_mse / num_batches

    # Check convergence - all operations must be below threshold
    math_losses = [eval_losses[op]["math"] for op in operations if op in eval_losses]
    all_eval_converged = (
        all(loss < converged_loss for loss in math_losses) if math_losses else False
    )

    return eval_losses, all_eval_converged


def evaluate_model(
    model: MathyUnit,
    test_loaders: Dict[str, DataLoader],
    operations: List[str],
    device: str,
) -> Dict[str, float]:
    """Evaluate model and return MSE for each operation."""
    model.eval()
    results = {}

    with torch.no_grad():
        for operation in operations:
            if operation not in test_loaders:
                continue

            total_mse = 0.0
            num_samples = 0

            for batch_x, batch_y in test_loaders[operation]:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model.forward(batch_x, operation)
                mse = torch.pow(pred - batch_y, 2).mean()

                total_mse += mse.item() * batch_x.size(0)
                num_samples += batch_x.size(0)

            results[operation] = (
                total_mse / num_samples if num_samples > 0 else float("inf")
            )

    return results


def train_initialization_model(
    operations: List[str],
    init_scale: float,
    device: str = "cpu",
    seed: int = 42,
    max_epochs: int = 10,
    use_schedulefree: bool = True,
    eval_frequency: int = 5,
) -> Tuple[MathyUnit, Dict[str, float], int]:
    """Train Hill Space model with specific initialization scale."""

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create dataloaders - train on Goldilocks, test on extreme range
    train_loaders = {}
    quick_eval_loaders = {}
    test_loaders = {}

    for operation in operations:
        if operation not in OPERATION_REGISTRY:
            continue

        spec = OPERATION_REGISTRY[operation]

        # Train on Goldilocks distribution
        train_loader, _ = create_mathy_dataloaders(
            operator_spec=spec,
            train_samples=64_000,
            test_samples=1000,  # dummy
            batch_size=64,
            train_range=(1e-8, 10.0),  # Goldilocks
            test_range=(1e-8, 10.0),  # dummy
            distribution="uniform",
            dtype=torch.float64,
            seed=seed,
        )

        # Test on extreme extrapolation range
        _, test_loader = create_mathy_dataloaders(
            operator_spec=spec,
            train_samples=1000,  # dummy
            test_samples=64_000,
            batch_size=64,
            train_range=(1e-8, 10.0),  # dummy
            test_range=(-1e4, 1e4),  # extreme range
            distribution="uniform",
            dtype=torch.float64,
            seed=seed,
        )

        # Quick eval loader on extreme range (smaller for speed)
        quick_eval_loader, _ = create_mathy_dataloaders(
            operator_spec=spec,
            train_samples=5_000,
            test_samples=5_000,
            batch_size=64,
            train_range=(-1e4, 1e4),  # extreme range for eval
            test_range=(-1e4, 1e4),
            distribution="uniform",
            dtype=torch.float64,
            seed=seed + 1000,
        )

        train_loaders[operation] = train_loader
        quick_eval_loaders[operation] = quick_eval_loader
        test_loaders[operation] = test_loader

    # Initialize model with specific init_scale
    model = MathyUnit(
        input_size=2,
        output_size=1,
        dtype=torch.float64,
        space="hill_snap",
        init_scale=init_scale,
    )
    model.to(device)

    # Training setup
    betas = (0.9, 0.96)
    weight_decay = 1e-8
    eps = 1e-8

    if use_schedulefree:
        optimizer = AdamWScheduleFree(
            model.get_arithmetic_parameters(),
            lr=0.1,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.get_arithmetic_parameters(),
            lr=0.1,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )

    criterion = nn.MSELoss()

    converged_epoch = None
    convergence_threshold = 1e-8
    recent_losses = []

    if use_schedulefree:
        optimizer.train()

    # Training loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []

        # Collect all batches and shuffle
        all_batches = []
        for operation, loader in train_loaders.items():
            for batch_x, batch_y in loader:
                all_batches.append((operation, batch_x.to(device), batch_y.to(device)))
        random.shuffle(all_batches)

        # Training step
        for operation, batch_x, batch_y in all_batches:
            optimizer.zero_grad()
            pred = model.forward(batch_x, operation)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Track recent performance for convergence
        avg_epoch_loss = np.mean(epoch_losses)
        recent_losses.append(avg_epoch_loss)
        if len(recent_losses) > 5:
            recent_losses.pop(0)

        # Check convergence: last 5 epochs all below threshold
        if len(recent_losses) == 5 and all(
            loss < convergence_threshold for loss in recent_losses
        ):
            converged_epoch = epoch
            break

        # Periodic evaluation
        if epoch % eval_frequency == 0 or epoch <= 10:
            if use_schedulefree:
                optimizer.eval()

            eval_losses, eval_all_converged = periodic_eval(
                model, quick_eval_loaders, operations, device, convergence_threshold
            )

            if use_schedulefree:
                optimizer.train()

            model.train()

            # Print detailed progress
            print(f"  Epoch {epoch:03d}, Train Loss: {avg_epoch_loss:.10f}")
            for operation in operations:
                if operation in eval_losses:
                    eval_mse = eval_losses[operation]["mse"]
                    weights = model.inspect_weights(operation)
                    print(
                        f"    {operation:8s}: eval_mse={eval_mse:.10f}, weights={weights}"
                    )

            # Early stopping if converged on eval
            if eval_all_converged:
                converged_epoch = epoch
                print(f"    Converged on eval at epoch {epoch}")
                break

        elif epoch % 10 == 0:  # Print every 10 epochs if no eval
            print(f"  Epoch {epoch:03d}, Loss: {avg_epoch_loss:.10f}")

    if use_schedulefree:
        optimizer.eval()

    # Final evaluation on extreme extrapolation range
    results = evaluate_model(model, test_loaders, operations, device)

    return model, results, converged_epoch or max_epochs


def run_single_initialization_analysis(run_id: int) -> str:
    """Run a single initialization analysis run - designed for multiprocessing."""

    operations = [
        # Additive operations
        "add",
        "subtract",
        # "negation",
        # Exponential operations
        "multiply",
        "divide",
        "identity",
        "reciprocal",
        # Trigonometric operations
        "cos",
        "sin",
        # Trigonometric product operations
        "cos_add",
        "sin_add",
        "cos_sub",
        "sin_sub",
    ]

    # Initialization scales to test
    init_scales = [0.0, 0.02, 1e-8, 1.0, 3.0, 10.0]
    device = "cpu"
    use_schedulefree = True  # Set to False to use standard AdamW

    # Generate a unique seed for this run
    seed = random.randint(0, 100000)
    optimizer_name = "schedulefree" if use_schedulefree else "adamw"
    results_file = (
        f"results/init_analysis_run_{run_id:02d}_seed_{seed}_{optimizer_name}.json"
    )

    print(f"Starting RUN {run_id + 1} (seed: {seed}, optimizer: {optimizer_name})")

    run_results = {
        "run_id": run_id,
        "seed": seed,
        "optimizer": optimizer_name,
        "init_scales": init_scales,
        "operations": operations,
        "results": {},
    }

    for init_scale in init_scales:
        print(f"  Run {run_id + 1}: Testing init_scale {init_scale}")

        model, results, epochs = train_initialization_model(
            operations=operations,
            init_scale=init_scale,
            device=device,
            seed=seed,
            use_schedulefree=use_schedulefree,
            eval_frequency=5,  # Evaluate every 5 epochs
        )

        # Store results for this init_scale
        run_results["results"][str(init_scale)] = {
            "mse_results": results,
            "convergence_epochs": epochs,
        }

        # Brief progress update
        avg_mse = np.mean(list(results.values()))
        print(
            f"  Run {run_id + 1}: init_scale {init_scale} -> {epochs} epochs, avg MSE: {avg_mse:.2e}"
        )

    # Save run results to JSON
    with open(results_file, "w") as f:
        json.dump(run_results, f, indent=2)

    print(f"Completed RUN {run_id + 1}, saved to {results_file}")
    return results_file


def run_initialization_analysis(num_processes: int = None):
    """Run the weight initialization analysis experiment with multiprocessing."""

    if num_processes is None:
        num_processes = min(cpu_count(), 10)  # Don't use more cores than runs

    print(f"Running initialization analysis with {num_processes} processes")

    num_runs = 10

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Run experiments in parallel
    with Pool(processes=num_processes) as pool:
        result_files = pool.map(run_single_initialization_analysis, range(num_runs))

    print(f"\nAll {num_runs} runs completed!")
    print("Saved files:")
    for file in result_files:
        print(f"  - {file}")
    print("\nRun 'aggregate_initialization_results()' to generate the final table.")


def aggregate_initialization_results():
    """Aggregate all initialization run results and generate markdown table."""

    # Load all run files
    run_files = glob.glob("results/init_analysis_run_*.json")
    if not run_files:
        print("No initialization analysis run files found!")
        return

    print(f"Found {len(run_files)} run files to aggregate")

    # Load and combine all results
    all_results = []
    for file in run_files:
        with open(file, "r") as f:
            run_data = json.load(f)
            all_results.append(run_data)

    # Extract operations and init_scales from first run
    operations = all_results[0]["operations"]
    init_scales = [float(scale) for scale in all_results[0]["init_scales"]]

    # Build aggregate data structure (and calcualate STD)
    aggregate_data = {}
    for operation in operations:
        aggregate_data[operation] = {}
        for init_scale in init_scales:
            mse_values = []
            std_values = []
            for run_data in all_results:

                mse = run_data["results"][str(init_scale)]["mse_results"][operation]
                mse_values.append(mse)

                std = np.std(mse_values)
                std_values.append(std)

            # Calculate mean MSE across all runs
            aggregate_data[operation][init_scale] = np.mean(mse_values)
            # Store standard deviation for reference
            aggregate_data[operation][f"{init_scale}_std"] = np.mean(std_values)

    # Format numbers for markdown table
    def format_mse(value):
        if value == 0.0:
            return "0.0"
        elif value < 1e-15:
            return f"{value:.0e}"
        elif value < 1e-10:
            return f"{value:.0e}"
        elif value < 1e-5:
            return f"{value:.0e}"
        elif value < 1e-2:
            return f"{value:.0e}"
        elif value < 1:
            return f"{value:.2f}"
        elif value < 1000:
            return f"{value:.1f}"
        else:
            return f"{value:.0e}"

    # Create operation name mapping for display
    operation_names = {
        "add": "a + b",
        "subtract": "a - b",
        "negation": "-a",
        "multiply": "a × b",
        "divide": "a ÷ b",
        "identity": "a",
        "reciprocal": "1/a",
        "cos": "cos(θ)",
        "sin": "sin(θ)",
        "cos_add": "cos(θ₁+θ₂)",
        "sin_add": "sin(θ₁+θ₂)",
        "cos_sub": "cos(θ₁-θ₂)",
        "sin_sub": "sin(θ₁-θ₂)",
    }

    # Generate markdown table
    print("\n" + "=" * 80)
    print("WEIGHT INITIALIZATION ANALYSIS RESULTS")
    print("=" * 80)
    print()

    # Table header
    header_scales = [str(scale) if scale != 0.0 else "0" for scale in init_scales]
    print(
        "| Operation  | " + " | ".join(f"{scale:>10}" for scale in header_scales) + " |"
    )
    print("| " + "-" * 10 + " | " + " | ".join("-" * 10 for _ in header_scales) + " |")

    # Table rows
    for operation in operations:
        display_name = operation_names.get(operation, operation)
        row_values = []

        for init_scale in init_scales:
            mse_value = aggregate_data[operation][init_scale]
            std_value = aggregate_data[operation][f"{init_scale}_std"]
            formatted_mse = format_mse(mse_value)
            formatted_std = format_mse(std_value)
            # Bold significantly bad values (heuristic: > 1e-2 for large scales)
            if mse_value > 1e-2:
                formatted_mse = f"**{formatted_mse}**"

            # Format with std if available
            if std_value > 0:
                formatted_mse = f"{formatted_mse} ± {formatted_std}"

            # If it's 1e-16 or less, format as 0.0
            if mse_value <= 1e-16:
                formatted_mse = "0.0"

            row_values.append(f"{formatted_mse:>10}")

        print(f"| {display_name:10} | " + " | ".join(row_values) + " |")

    print()
    # Add optimizer info to the note
    optimizer_info = all_results[0].get("optimizer", "unknown")
    print(
        f"*Note: Results averaged over {len(all_results)}. Values in bold indicate degraded performance.*"
    )
    print()

    # Save detailed results to CSV for further analysis
    detailed_results = []
    for run_data in all_results:
        for init_scale_str, scale_results in run_data["results"].items():
            for operation, mse in scale_results["mse_results"].items():
                detailed_results.append(
                    {
                        "run_id": run_data["run_id"],
                        "seed": run_data["seed"],
                        "optimizer": run_data.get("optimizer", "unknown"),
                        "init_scale": float(init_scale_str),
                        "operation": operation,
                        "mse": mse,
                        "epochs": scale_results["convergence_epochs"],
                    }
                )

    df = pd.DataFrame(detailed_results)
    df.to_csv("results/initialization_analysis_detailed.csv", index=False)

    # Save summary table
    summary_data = []
    for operation in operations:
        row = {"operation": operation_names.get(operation, operation)}
        for init_scale in init_scales:
            col_name = str(init_scale) if init_scale != 0.0 else "0"
            row[col_name] = aggregate_data[operation][init_scale]
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results/initialization_analysis_summary.csv", index=False)

    print("Files saved:")
    print("  - results/initialization_analysis_detailed.csv (all run data)")
    print("  - results/initialization_analysis_summary.csv (mean values)")

    return aggregate_data


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    # run_initialization_analysis(num_processes=6)
    aggregate_initialization_results()
