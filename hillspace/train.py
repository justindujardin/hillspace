import multiprocessing as mp
import os
import random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader

from .dataset.operator_dataset import create_mathy_dataloaders
from .dataset.operator_specs import OPERATION_REGISTRY, OPERATION_SYMBOLS
from .model import MathySpace, MathyUnit


def periodic_eval(
    model: MathyUnit,
    test_loaders: Dict[str, DataLoader[Tuple[torch.Tensor, torch.Tensor]]],
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
                final_result = model.forward(batch_x, operation)
                math_loss = F.l1_loss(final_result, batch_y)
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


def train(
    model: MathyUnit,
    *,
    math_lr: float,
    train_loaders: Dict[str, DataLoader[Tuple[torch.Tensor, torch.Tensor]]],
    quick_eval_loaders: Dict[str, DataLoader[Tuple[torch.Tensor, torch.Tensor]]],
    epochs: int,
    math_weight: float,
    operations: List[str],
    optimizer_type: Literal["schedulefree", "adamw", "adam"],
    device: str = "auto",
    project_name: str = "jac-arithmetic",
    run_name: Optional[str] = None,
    model_save_path: Optional[str] = None,
    eval_frequency: int = 10,
) -> MathyUnit:

    betas = (0.9, 0.96)
    weight_decay = 1e-8
    eps = 1e-8
    if optimizer_type == "schedulefree":
        # Use ScheduleFree AdamW optimizer for better convergence
        optimizer = AdamWScheduleFree(
            model.get_arithmetic_parameters(),
            lr=math_lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optimizer_type == "adamw":
        # Use standard AdamW optimizer
        optimizer = torch.optim.AdamW(
            model.get_arithmetic_parameters(),
            lr=math_lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optimizer_type == "adam":
        # Use standard Adam optimizer
        optimizer = torch.optim.Adam(
            model.get_arithmetic_parameters(),
            lr=math_lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )

    config = {
        "math_lr": math_lr,
        "epochs": epochs,
        "math_weight": math_weight,
        "device": device,
        "total_params": sum(p.numel() for p in model.parameters()),
        "eval_frequency": eval_frequency,
    }
    wandb.init(project=project_name, name=run_name, config=config, tags=[])
    wandb.watch(model, log="all", log_freq=50)

    math_criterion = nn.MSELoss()
    model.train()
    print(f"Training MathyUnit on ALL operations: {operations}")
    print(f"Loss weights - Math: {math_weight}")
    print(f"Eval frequency: every {eval_frequency} epochs")
    print("-" * 60)

    if isinstance(optimizer, AdamWScheduleFree):
        optimizer.train()
    converged_loss = 1e-12

    for epoch in range(epochs):
        total_losses = {
            op: {"math": 0.0, "combined": 0.0, "mae": 0.0} for op in operations
        }
        num_batches = {op: 0 for op in operations}

        # Randomly interleave operations within each epoch for richer training
        all_batches = []
        for operation, loader in train_loaders.items():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                all_batches.append((operation, batch_x, batch_y))

        random.shuffle(all_batches)

        epoch_gradients = []

        for operation, batch_x, batch_y in all_batches:
            optimizer.zero_grad()

            # Model Forward pass
            final_result = model.forward(batch_x, operation)
            math_loss = math_criterion(final_result, batch_y)
            mae = torch.abs(final_result - batch_y).mean()
            # Any Nans?
            if torch.isnan(math_loss):
                print(
                    f"Warning: NaN loss detected for operation '{operation}' at epoch {epoch}."
                )
                for i in range(batch_x.size(0)):
                    if torch.isnan(final_result[i]).any():
                        print(
                            f"  NaN in final result for input {batch_x[i].cpu().numpy()}"
                        )
                    if torch.isnan(batch_y[i]).any():
                        print(f"  NaN in target for input {batch_x[i].cpu().numpy()}")
                raise ValueError(
                    f"NaN loss detected for operation '{operation}' at epoch {epoch}."
                )
            math_loss = math_loss * math_weight
            total_loss = math_loss
            total_loss.backward()

            # Collect gradient norms for analysis
            math_grad_norm = 0.0
            # Math circuit gradients (w_hat parameters)
            for param in model.get_arithmetic_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    math_grad_norm += grad_norm**2
            epoch_gradients.append(math_grad_norm**0.5)
            optimizer.step()
            total_losses[operation]["math"] += math_loss.item()
            total_losses[operation]["combined"] += total_loss.item()
            total_losses[operation]["mae"] += mae.item()
            num_batches[operation] += 1

        # Calculate average losses
        avg_losses = {}
        for operation in operations:
            if num_batches[operation] > 0:
                avg_losses[operation] = {
                    "math": total_losses[operation]["math"] / num_batches[operation],
                    "mae": total_losses[operation]["mae"] / num_batches[operation],
                    "combined": total_losses[operation]["combined"]
                    / num_batches[operation],
                }
            else:
                avg_losses[operation] = {
                    "math": 0.0,
                    "combined": 0.0,
                    "mae": 0.0,
                }

        log_dict = {"epoch": epoch}

        # Operation-specific losses
        for operation in operations:
            log_dict[f"loss/{operation}_math"] = avg_losses[operation]["math"]
            log_dict[f"loss/{operation}_mae"] = avg_losses[operation]["mae"]
            log_dict[f"loss/{operation}_combined"] = avg_losses[operation]["combined"]

        # Aggregate metrics
        math_losses = [
            avg_losses[op]["math"] for op in operations if num_batches[op] > 0
        ]
        log_dict["loss/avg_mae"] = (
            np.mean([avg_losses[op]["mae"] for op in operations if num_batches[op] > 0])
            if any(num_batches.values())
            else 0.0
        )

        log_dict["loss/avg_math"] = np.mean(math_losses) if math_losses else 0.0

        # Gradient norms
        if epoch_gradients:
            log_dict["grads/math_circuits"] = np.mean(epoch_gradients)

        # Convergence indicators
        log_dict["convergence/math_below_threshold"] = sum(
            1 for loss in math_losses if loss < converged_loss
        )
        train_all_converged = (
            all(loss < converged_loss for loss in math_losses) if math_losses else False
        )
        log_dict["convergence/all_math_converged"] = train_all_converged

        # Run periodic evaluation
        eval_all_converged = False
        if epoch % eval_frequency == 0 or epoch <= 10:
            # Toggle optimizer to eval mode
            if isinstance(optimizer, AdamWScheduleFree):
                optimizer.eval()
            eval_losses, eval_all_converged = periodic_eval(
                model, quick_eval_loaders, operations, device, converged_loss
            )
            # Toggle back to train mode
            if isinstance(optimizer, AdamWScheduleFree):
                optimizer.train()
            model.train()

            # Log eval metrics
            for operation in operations:
                if operation in eval_losses:
                    log_dict[f"eval/{operation}_math"] = eval_losses[operation]["math"]
                    log_dict[f"eval/{operation}_mae"] = eval_losses[operation]["mae"]
                    log_dict[f"eval/{operation}_mse"] = eval_losses[operation]["mse"]

            # Aggregate eval metrics
            eval_math_losses = [
                eval_losses[op]["math"] for op in operations if op in eval_losses
            ]
            if eval_math_losses:
                log_dict["eval/avg_math"] = np.mean(eval_math_losses)
                log_dict["eval/avg_mae"] = np.mean(
                    [eval_losses[op]["mae"] for op in operations if op in eval_losses]
                )
                log_dict["eval/avg_mse"] = np.mean(
                    [eval_losses[op]["mse"] for op in operations if op in eval_losses]
                )

            log_dict["convergence/eval_all_converged"] = eval_all_converged
            log_dict["convergence/eval_below_threshold"] = sum(
                1 for loss in eval_math_losses if loss < converged_loss
            )

        wandb.log(log_dict)

        # Print progress every 10 epochs after the first 10 epochs
        if epoch % 10 == 0 or epoch <= 10:
            # Save every epoch in a subfolder
            if model_save_path:
                epoch_save_path = os.path.join(model_save_path, f"epoch_{epoch}")
                os.makedirs(epoch_save_path, exist_ok=True)
                if isinstance(optimizer, AdamWScheduleFree):
                    optimizer.eval()
                model.save_pretrained(epoch_save_path)
                if isinstance(optimizer, AdamWScheduleFree):
                    optimizer.train()

            print(f"Epoch {epoch:3d}:")
            for operation in operations:
                train_losses = avg_losses[operation]
                if epoch % eval_frequency == 0 or epoch <= 10:
                    eval_op_losses = eval_losses.get(
                        operation, {"math": 0.0, "mae": 0.0, "mse": 0.0}
                    )
                    print(
                        f"  {operation:8s}: train_math={train_losses['math']:.12f} eval_math={eval_op_losses['math']:.12f} "
                        f"train_mae={train_losses['mae']:.12f} eval_mae={eval_op_losses['mae']:.12f}"
                    )
                else:
                    print(
                        f"  {operation:8s}: train_math={train_losses['math']:.12f} train_mae={train_losses['mae']:.12f}"
                    )

        # Early stopping if all operations converged on BOTH train and eval
        if train_all_converged and (eval_all_converged or epoch < eval_frequency):
            print(f"All operations converged at epoch {epoch}")
            wandb.log({"convergence/converged_epoch": epoch})
            break

    if isinstance(optimizer, AdamWScheduleFree):
        optimizer.eval()
    return model


def test_model(
    model: MathyUnit,
    test_loaders: Dict[str, DataLoader[Tuple[torch.Tensor, torch.Tensor]]],
    device: str = "auto",
):
    print(f"\nTesting MathyUnit on ALL operations")
    print("=" * 60)
    model.eval()
    for operation, test_loader in test_loaders.items():
        print(f"\n{operation.upper()} Operation:")
        print("-" * 40)
        # Statistics across ALL test examples
        all_math_errors = []
        all_targets = []
        all_math_preds = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # Move data to device
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Get JAC'd predictions
                math_results = model.forward(batch_x, operation)

                # Collect statistics for ALL examples in this batch
                for i in range(batch_x.size(0)):
                    expected = batch_y[i].item()
                    math_val = (
                        math_results[i].item()
                        if math_results[i].numel() == 1
                        else math_results[i][0].item()
                    )

                    # Track errors
                    math_error = abs(math_val - expected)
                    all_math_errors.append(math_error)
                    all_targets.append(expected)
                    all_math_preds.append(math_val)

        # Calculate comprehensive statistics
        if all_math_errors:
            math_mae = np.mean(all_math_errors)
            math_rmse = np.sqrt(np.mean([e**2 for e in all_math_errors]))

            print(f"FULL TEST SET ANALYSIS ({len(all_math_errors)} examples):")
            print(f"  Math circuit - MAE: {math_mae:.4f}, RMSE: {math_rmse:.4f}")


def main():
    mp.set_start_method("spawn")
    dev_quick_test = (
        False  # Set to True for quick testing with fewer epochs and samples
    )

    torch.use_deterministic_algorithms(True)
    device = "cpu"
    train_dtype = torch.float32
    train_dtype = torch.float64
    print(f"Using device: {device} with dtype: {train_dtype}")
    space: MathySpace = "hill_snap"
    optimizer_type = "schedulefree"  # "adamw" or "adam" or "schedulefree"

    # Create dataloaders for ALL operations
    operations = [
        # Additive operations
        "add",
        "subtract",
        "negation",
        # "identity", # Same as exponential, need a diff name to train both (pass)
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
    train_loaders = {}
    quick_eval_loaders = {}
    test_loaders = {}

    print(f"Creating dataloaders for operations: {operations}")
    print()

    test_cases: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
        # "uniform:extreme_hundreds": ((-100.0, 100.0), (-1e5, 1e5)),
        # "uniform:extreme_tens_positive_test_neg": ((1e-8, 10.0), (-1e5, 10.0)),
        # "uniform:extreme_tens": ((-10.0, 10.0), (-1e5, 1e5)),
        # "uniform:extreme_unit": ((-1.0, 1.0), (-1e5, 1e5)),
        # "uniform:goldilocks_test_neg1e4_pos1e4": ((1e-8, 25.0), (-1e4, 1e4)),
        "uniform:goldilocks": ((1e-8, 10.0), (-1e4, 1e4)),
        # "truncated_normal:inalu_neg2t4": ((-2.0, 4.0), (-1e5, 1e5)),
        # "truncated_normal:inalu_neg3t3": ((-3.0, 3.0), (-1e5, 1e5)),
        # "truncated_normal:inalu_neg4t2": ((-4.0, 2.0), (-1e5, 1e5)),
        # "uniform:inalu_neg1t1": ((-1.0, 1.0), (-1e5, 1e5)),
        # "uniform:inalu_neg5t5": ((-5.0, 5.0), (-1e5, 1e5)),
    }

    for key, (train_range, test_range) in test_cases.items():
        seed = 1337
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        distribution = key.split(":")[0]

        if ":" in key:
            # Replace ':' with '_' for compatibility with WandB
            key = key.replace(":", "_")

        if distribution not in ["truncated_normal", "uniform"]:
            raise ValueError(
                f"Distribution '{distribution}' not supported. Use 'truncated_normal' or 'uniform'."
            )

        print(f"\n🎯 {key}: Train range: {train_range}, Test range: {test_range}")
        print(f"📊 Distribution: {distribution}")

        # Reset loaders for this test case
        train_loaders = {}
        quick_eval_loaders = {}
        test_loaders = {}

        for operation in operations:
            if operation not in OPERATION_REGISTRY:
                print(f"⚠️  Operation '{operation}' not found in registry, skipping...")
                continue

            spec = OPERATION_REGISTRY[operation]

            # Use operation-specific optimal ranges if the global ranges don't work well
            # For most operations, use the provided ranges, but some may need adjustment
            op_train_range = train_range
            op_test_range = test_range

            print(f"  🔧 Creating datasets for {operation}...")

            try:
                train_loader, test_loader = create_mathy_dataloaders(
                    operator_spec=spec,
                    train_samples=64_000 if not dev_quick_test else 10_000,
                    test_samples=64_000,
                    batch_size=64,
                    train_range=op_train_range,
                    test_range=op_test_range,
                    distribution=distribution,
                    dtype=train_dtype,
                    seed=seed,
                )

                # Create quick eval loader
                quick_eval_loader, _ = create_mathy_dataloaders(
                    operator_spec=spec,
                    train_samples=1_000,
                    test_samples=1_000,
                    batch_size=64,
                    train_range=op_test_range,  # Use test_range for eval distribution
                    test_range=op_test_range,
                    distribution=distribution,
                    dtype=train_dtype,
                    seed=seed + 1000,
                )

                train_loaders[operation] = train_loader
                quick_eval_loaders[operation] = quick_eval_loader
                test_loaders[operation] = test_loader

                print(
                    f"  ✅ {operation}: {len(train_loader)} train, {len(quick_eval_loader)} eval, {len(test_loader)} test batches"
                )

            except Exception as e:
                print(f"  ❌ Failed to create {operation} datasets: {e}")
                continue

        if not train_loaders:
            raise ValueError(
                f"No valid training loaders created for operations in test case '{key}'."
            )
        print(
            f"\n🚀 Training on {len(train_loaders)} operations: {list(train_loaders.keys())}"
        )

        # Create the MathyUnit model with the specified space
        model = MathyUnit(2, output_size=1, dtype=train_dtype, space=space)
        model.to(device)

        p = "32" if train_dtype == torch.float32 else "64"
        run_name = f"{optimizer_type}-{space}-{key}-float{p}-seed{seed}"
        model_save_path = f"./trained/{run_name}"

        model = train(
            model,
            optimizer_type=optimizer_type,
            train_loaders=train_loaders,
            quick_eval_loaders=quick_eval_loaders,
            epochs=5 if dev_quick_test else 200,
            math_lr=0.1,
            math_weight=1.0,
            device=device,
            operations=list(train_loaders.keys()),
            project_name="mathy-arithmetic",
            run_name=run_name,
            model_save_path=model_save_path,
            eval_frequency=10,
        )

        model.eval()
        model.save_pretrained(model_save_path)
        print(f"\n💾 Final model saved to: {model_save_path}")

        # Comprehensive testing across all operations
        print(f"\n🧪 Testing model performance...")
        test_model(model, test_loaders, device=device)

        # Pretty manual tests with emojis and symbols
        print(f"\n🎲 Manual test examples:")
        with torch.no_grad():
            test_inputs = torch.tensor(
                [[2.0, 3.0], [10.0, 5.0], [20.0, 18.0], [1.234567, -6.25]],
                device=device,
                dtype=train_dtype,
            )
            for i, inputs in enumerate(test_inputs):
                a, b = inputs[0].item(), inputs[1].item()
                print(f"\n  📋 Test {i+1}: inputs = ({a:.3f}, {b:.3f})")

                # Only test operations that were actually trained
                for operation in train_loaders.keys():
                    try:
                        math_result = model.forward(inputs.unsqueeze(0), operation)
                        math_val = math_result[0].item()
                        symbol = OPERATION_SYMBOLS.get(operation, operation)

                        # For unary operations, don't show the second input
                        if OPERATION_REGISTRY[operation].arity == 1:
                            print(f"    {symbol}({a:.3f}) = {math_val:.3f}")
                        else:
                            print(f"    {a:.3f} {symbol} {b:.3f} = {math_val:.3f}")
                    except Exception as e:
                        print(f"    ❌ {operation}: error = {e}")

        print(f"\n🏁 Finished test case: {key}")
        wandb.finish()


if __name__ == "__main__":
    main()
