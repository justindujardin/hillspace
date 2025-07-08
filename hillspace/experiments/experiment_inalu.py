import csv
import glob
import os
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader

from ..dataset.operator_dataset import create_mathy_dataloaders
from ..dataset.operator_specs import OPERATION_REGISTRY
from ..model import MathyUnit


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


def periodic_eval(
    model: MathyUnit,
    test_loaders: Dict[str, DataLoader],
    operations: List[str],
    device: str,
    converged_loss: float = 1e-12,
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


def train_hill_space_model(
    operations: List[str],
    train_range: Tuple[float, float],
    test_range: Tuple[float, float],
    distribution: str = "uniform",
    device: str = "cpu",
    seed: int = 42,
    eval_frequency: int = 10,
) -> Tuple[MathyUnit, Dict[str, float], int]:
    """Train Hill Space model and return model, results, and convergence epoch."""

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create dataloaders
    train_loaders = {}
    quick_eval_loaders = {}
    test_loaders = {}

    for operation in operations:
        if operation not in OPERATION_REGISTRY:
            continue

        spec = OPERATION_REGISTRY[operation]
        train_loader, test_loader = create_mathy_dataloaders(
            operator_spec=spec,
            train_samples=64_000,
            test_samples=64_000,
            batch_size=64,
            train_range=train_range,
            test_range=test_range,
            distribution=distribution,
            dtype=torch.float64,
            seed=seed,
        )

        # Quick eval loader on test range (smaller for speed)
        quick_eval_loader, _ = create_mathy_dataloaders(
            operator_spec=spec,
            train_samples=5_000,
            test_samples=5_000,
            batch_size=64,
            train_range=test_range,  # Use test range for eval distribution
            test_range=test_range,
            distribution=distribution,
            dtype=torch.float64,
            seed=seed + 1000,
        )

        train_loaders[operation] = train_loader
        quick_eval_loaders[operation] = quick_eval_loader
        test_loaders[operation] = test_loader

    # Initialize model
    model = MathyUnit(
        input_size=2, output_size=1, dtype=torch.float64, space="hill_snap"
    )
    model.to(device)

    # Training setup
    betas = (0.9, 0.96)
    weight_decay = 1e-8
    eps = 1e-8
    schedule_free = True  # Use schedule-free optimizer
    optimizer = AdamWScheduleFree(
        model.get_arithmetic_parameters(),
        lr=0.1,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        warmup_steps=100,
    )
    criterion = nn.MSELoss()

    converged_epoch = None
    convergence_threshold = 1e-10
    recent_losses = []  # Track recent epoch losses

    # Max 10 epochs (we find they converge or don't within ~3 so this is conservative)
    for epoch in range(10):

        model.train()
        if schedule_free:
            optimizer.train()
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
        if len(recent_losses) > 5:  # Keep only last 5 epochs
            recent_losses.pop(0)

        # Check convergence: last 5 epochs all below threshold
        if len(recent_losses) == 5 and all(
            loss < convergence_threshold for loss in recent_losses
        ):
            converged_epoch = epoch
            break

        # Periodic evaluation
        if epoch % eval_frequency == 0 or epoch <= 10:
            if schedule_free:
                optimizer.eval()

            eval_losses, eval_all_converged = periodic_eval(
                model, quick_eval_loaders, operations, device, convergence_threshold
            )
            if schedule_free:
                optimizer.train()

            model.train()

            # Print detailed progress
            print(f"  Epoch {epoch:03d}, Train Loss: {avg_epoch_loss:.16f}")
            for operation in operations:
                if operation in eval_losses:
                    eval_mse = eval_losses[operation]["mse"]
                    weights = model.inspect_weights(operation)
                    print(
                        f"    {operation:8s}: eval_mse={eval_mse:.16f}, weights={weights}"
                    )

            # Early stopping if converged on eval
            if eval_all_converged:
                converged_epoch = epoch
                print(f"    Converged on eval at epoch {epoch}")
                break

        elif epoch % 10 == 0:  # Print every 10 epochs if no eval
            print(f"  Epoch {epoch:03d}, Loss: {avg_epoch_loss:.16f}")

    # Final evaluation
    if schedule_free:
        optimizer.train()
    results = evaluate_model(model, test_loaders, operations, device)

    return model, results, converged_epoch or 100


def run_single_inalu_comparison(run_id: int) -> str:
    """Run a single iNALU comparison run - designed for multiprocessing."""

    # Generate a unique seed for this run
    seed = random.randint(0, 100000)
    results_file = f"results/inalu_comparison_{run_id:02d}_seed_{seed}.csv"

    print(f"Starting iNALU comparison RUN {run_id + 1} (seed: {seed})")

    # iNALU test configurations
    test_configs = [
        ("exponential_0.8_0.5", (0.8, 0.5), (0.8, 0.5), "exponential"),
        ("uniform_-5_5_to_-10_-5", (-5.0, 5.0), (-10.0, -5.0), "uniform"),
        ("normal_-3_3_to_8_10", (-3.0, 3.0), (8.0, 10.0), "truncated_normal"),
    ]

    operations = ["add", "subtract", "multiply", "divide"]
    device = "cpu"

    # STEP 1: Train ONE universal model on Goldilocks distribution for ALL operations
    print(f"  Run {run_id + 1}: Training universal model...")
    universal_model, _, universal_epochs = train_hill_space_model(
        operations=operations,
        train_range=(1e-8, 10.0),
        test_range=(1e-8, 10.0),
        distribution="uniform",
        device=device,
        seed=seed,
        eval_frequency=10,
    )
    print(f"  Run {run_id + 1}: Universal model trained in {universal_epochs} epochs")

    results = []

    # STEP 2: For each iNALU configuration, do matched training AND universal evaluation
    for config_name, train_range, test_range, distribution in test_configs:
        print(f"  Run {run_id + 1}: Processing {config_name}...")

        for operation in operations:
            # MATCHED: Train operation-specific model
            model_matched, results_matched, epochs_matched = train_hill_space_model(
                operations=[operation],
                train_range=train_range,
                test_range=test_range,
                distribution=distribution,
                device=device,
                seed=seed,
                eval_frequency=10,
            )

            # UNIVERSAL: Evaluate the pre-trained universal model
            spec = OPERATION_REGISTRY[operation]
            _, test_loader = create_mathy_dataloaders(
                operator_spec=spec,
                train_samples=1000,
                test_samples=64_000,
                batch_size=64,
                train_range=train_range,
                test_range=test_range,
                distribution=distribution,
                dtype=torch.float64,
                seed=seed,
            )

            universal_results = evaluate_model(
                universal_model, {operation: test_loader}, [operation], device
            )

            # Store results
            results.append(
                {
                    "config": config_name,
                    "operation": operation,
                    "distribution": distribution,
                    "train_range": f"{train_range[0]},{train_range[1]}",
                    "test_range": f"{test_range[0]},{test_range[1]}",
                    "hill_space_matched_mse": results_matched[operation],
                    "hill_space_matched_epochs": epochs_matched,
                    "hill_space_universal_mse": universal_results[operation],
                    "hill_space_universal_epochs": universal_epochs,
                }
            )

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    with open(results_file, "w", newline="") as f:
        fieldnames = [
            "config",
            "operation",
            "distribution",
            "train_range",
            "test_range",
            "hill_space_matched_mse",
            "hill_space_matched_epochs",
            "hill_space_universal_mse",
            "hill_space_universal_epochs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Completed iNALU comparison RUN {run_id + 1}, saved to {results_file}")
    return results_file


def run_inalu_comparison_parallel(num_processes: int = None):
    """Run the iNALU comparison experiment with multiprocessing."""

    if num_processes is None:
        num_processes = min(cpu_count(), 10)

    print(f"Running iNALU comparison with {num_processes} processes")

    num_runs = 10

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Run experiments in parallel
    with Pool(processes=num_processes) as pool:
        result_files = pool.map(run_single_inalu_comparison, range(num_runs))

    print(f"\nAll {num_runs} iNALU comparison runs completed!")
    print("Saved files:")
    for file in result_files:
        print(f"  - {file}")
    print("\nRun 'aggregate_results()' to generate the final comparison table.")


def aggregate_results():
    """Aggregate all run results into summary statistics with markdown table output."""

    # Load all run files
    run_files = glob.glob("results/inalu_comparison_*.csv")
    if not run_files:
        print("No run files found!")
        return

    print(f"Found {len(run_files)} run files to aggregate")

    # Combine all runs
    all_data = []
    for file in run_files:
        df = pd.read_csv(file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Calculate summary statistics
    summary = (
        combined_df.groupby(["config", "operation"])
        .agg(
            {
                "hill_space_matched_mse": ["mean", "std"],
                "hill_space_matched_epochs": ["mean", "std"],
                "hill_space_universal_mse": ["mean", "std"],
                "hill_space_universal_epochs": ["mean", "std"],
            }
        )
        .round(10)
    )

    # Format numbers for markdown table - match initialization script format
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

    def format_epochs(value):
        return f"{value:.1f}"

    # Create operation name mapping for display
    operation_names = {
        "add": "a + b",
        "subtract": "a - b",
        "multiply": "a × b",
        "divide": "a ÷ b",
    }

    # Configuration name mapping for display
    config_names = {
        "exponential_0.8_0.5": "E(0.8,0.5)",
        "uniform_-5_5_to_-10_-5": "U(-5,5)",
        "normal_-3_3_to_8_10": "N(-3,3)",
    }

    # iNALU baseline results (from paper)

    # iNALU baseline results (real means from source data)
    inalu_baselines = {
        ("exponential_0.8_0.5", "add"): (2e-15, 3e-17),
        # exponential_0.8_0.5       add      | 2e-15 ± 3e-17
        ("exponential_0.8_0.5", "subtract"): (1e-15, 2e-17),
        # exponential_0.8_0.5       subtract | 1e-15 ± 2e-17
        ("exponential_0.8_0.5", "multiply"): (1e-15, 6e-17),
        # exponential_0.8_0.5       multiply | 1e-15 ± 6e-17
        ("exponential_0.8_0.5", "divide"): (362.4, 1078.7),
        # exponential_0.8_0.5       divide   | 362.4 ± 1078.7
        ("uniform_-5_5_to_-10_-5", "add"): (4e-13, 3e-15),
        # uniform_-5_5_to_-10_-5    add      | 4e-13 ± 3e-15
        ("uniform_-5_5_to_-10_-5", "subtract"): (9e-14, 5e-16),
        # uniform_-5_5_to_-10_-5    subtract | 9e-14 ± 5e-16
        ("uniform_-5_5_to_-10_-5", "multiply"): (1e-10, 9e-13),
        # uniform_-5_5_to_-10_-5    multiply | 1e-10 ± 9e-13
        ("uniform_-5_5_to_-10_-5", "divide"): (0.23, 0.34),
        # uniform_-5_5_to_-10_-5    divide   | 0.23 ± 0.34
        ("normal_-3_3_to_8_10", "add"): (9e-13, 5e-15),
        # normal_-3_3_to_8_10       add      | 9e-13 ± 5e-15
        ("normal_-3_3_to_8_10", "subtract"): (2e-13, 8e-16),
        # normal_-3_3_to_8_10       subtract | 2e-13 ± 8e-16
        ("normal_-3_3_to_8_10", "multiply"): (3e-10, 2e-12),
        # normal_-3_3_to_8_10       multiply | 3e-10 ± 2e-12
        ("normal_-3_3_to_8_10", "divide"): (2.7, 4.1),
        # normal_-3_3_to_8_10       divide   | 2.7 ± 4.1
    }

    # Generate markdown table
    print("\n" + "=" * 80)
    print("HILL SPACE vs iNALU COMPARISON RESULTS")
    print("=" * 80)
    print()

    # Create comprehensive table with both matched and universal results
    # Group data for table
    table_data = {}
    for (config, operation), group in combined_df.groupby(["config", "operation"]):
        matched_mse_mean = group["hill_space_matched_mse"].mean()
        matched_mse_std = group["hill_space_matched_mse"].std()
        universal_mse_mean = group["hill_space_universal_mse"].mean()
        universal_mse_std = group["hill_space_universal_mse"].std()
        matched_epochs_mean = group["hill_space_matched_epochs"].mean()
        universal_epochs_mean = group["hill_space_universal_epochs"].mean()

        if config not in table_data:
            table_data[config] = {}

        table_data[config][operation] = {
            "matched_mse": matched_mse_mean,
            "matched_mse_std": matched_mse_std,
            "universal_mse": universal_mse_mean,
            "universal_mse_std": universal_mse_std,
            "matched_epochs": matched_epochs_mean,
            "universal_epochs": universal_epochs_mean,
        }

    # Print table header
    print("| Distribution | Operation | iNALU MSE | Matched MSE | Universal MSE |")
    print(
        "| "
        + "-" * 12
        + " | "
        + "-" * 9
        + " | "
        + "-" * 9
        + " | "
        + "-" * 11
        + " | "
        + "-" * 13
        + " | "
    )

    # Print table rows
    for config in [
        "exponential_0.8_0.5",
        "uniform_-5_5_to_-10_-5",
        "normal_-3_3_to_8_10",
    ]:
        for operation in ["add", "subtract", "multiply", "divide"]:
            if config in table_data and operation in table_data[config]:
                data = table_data[config][operation]

                config_display = config_names.get(config, config)
                op_display = operation_names.get(operation, operation)

                # Get iNALU baseline
                inalu_mse, inalu_std = inalu_baselines.get((config, operation), 0.0)
                inalu_std_formatted = (
                    f" ± {format_mse(inalu_std)}" if inalu_std > 0 else ""
                )
                inalu_mse_formatted = format_mse(inalu_mse)
                inalu_mse_formatted += inalu_std_formatted
                if inalu_mse > 1e-2:
                    inalu_mse_formatted = f"**{inalu_mse_formatted}**"

                matched_mse = format_mse(data["matched_mse"])
                matched_mse_std = format_mse(data["matched_mse_std"])
                matched_mse = f"{matched_mse} ± {matched_mse_std}"
                universal_mse = format_mse(data["universal_mse"])
                universal_mse_std = format_mse(data["universal_mse_std"])
                universal_mse = f"{universal_mse} ± {universal_mse_std}"

                # Bold values that indicate poor performance (heuristic: > 1e-2)
                if data["matched_mse"] > 1e-2:
                    matched_mse = f"**{matched_mse}**"
                if data["universal_mse"] > 1e-2:
                    universal_mse = f"**{universal_mse}**"

                print(
                    f"| {config_display:12} | {op_display:9} | {inalu_mse_formatted:>15} | {matched_mse:>11} | {universal_mse:>13} |"
                )

    print()
    num_runs = len(run_files)
    print(
        f"*Note: Results averaged over {num_runs} runs. Values in bold indicate degraded performance (MSE > 1e-2).*"
    )
    print(
        f"*Universal model trains once on Goldilocks distribution then evaluates across all test configurations.*"
    )
    print()

    # Save CSV files
    combined_df.to_csv("results/inalu_all_runs.csv", index=False)

    # Create clean summary for CSV
    clean_summary = []
    for config in table_data:
        for operation in table_data[config]:
            data = table_data[config][operation]
            clean_summary.append(
                {
                    "config": config,
                    "operation": operation,
                    "matched_mse_mean": data["matched_mse"],
                    "matched_mse_std": data["matched_mse_std"],
                    "universal_mse_mean": data["universal_mse"],
                    "universal_mse_std": data["universal_mse_std"],
                    "matched_epochs_mean": data["matched_epochs"],
                    "universal_epochs_mean": data["universal_epochs"],
                }
            )

    summary_df = pd.DataFrame(clean_summary)
    summary_df.to_csv("results/inalu_summary.csv", index=False)

    print("Files saved:")
    print("  - results/inalu_all_runs.csv (raw data from all runs)")
    print("  - results/inalu_summary.csv (mean values)")

    return combined_df, summary_df


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    print("Starting iNALU comparison experiment...")
    # run_inalu_comparison_parallel(6)
    aggregate_results()
