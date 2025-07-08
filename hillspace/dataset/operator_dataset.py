import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Tuple, List, Optional
from scipy import stats


def bound_precision(data: np.ndarray, precision_decimals: int = 4) -> np.ndarray:
    """Bound precision of data to specified decimal places"""
    multiplier = 10**precision_decimals
    return np.round(data * multiplier) / multiplier


class MathyOperatorDataset(Dataset):
    def __init__(
        self,
        operator_spec,  # OperatorSpec
        distribution: list[Literal["uniform", "truncated_normal", "exponential"]] = [
            "uniform"
        ],
        n_samples: int = 64000,
        train_range: Tuple[float, float] = (1e-8, 10.0),
        test_range: Tuple[float, float] = (-100.0, 100.0),
        normal_params: Tuple[float, float] = (0, 1),
        exp_lambda: float = 0.2,
        precision_limit: Optional[float] = None,
        seed: Optional[int] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dtype: torch.dtype = torch.float32,
        is_test: bool = False,
        strict_constraints: bool = False,
        verbose: bool = False,  # Whether to print generation details
    ):
        super().__init__()
        self.spec = operator_spec
        self.distribution = distribution
        self.n_samples = n_samples
        self.train_range = train_range
        self.test_range = test_range
        self.normal_params = normal_params
        self.exp_lambda = exp_lambda
        self.device = device
        self.is_test = is_test
        self.precision_limit = precision_limit
        self.dtype = dtype
        self.strict_constraints = strict_constraints
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Use test range if this is extrapolation data
        self.active_range = test_range if is_test else train_range

        # Pre-generate all data
        self.data_x, self.data_y = self._generate_data()

    def _generate_base_inputs(
        self, n_samples: int, distribution: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base inputs according to specified distribution"""

        if distribution == "uniform":
            a, b = self.active_range
            x1 = np.random.uniform(a, b, size=n_samples)
            x2 = np.random.uniform(a, b, size=n_samples)

        elif distribution == "truncated_normal":
            a, b = self.active_range
            mu, sigma = self.normal_params
            truncated_norm = stats.truncnorm(
                (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
            )
            x1 = truncated_norm.rvs(size=n_samples)
            x2 = truncated_norm.rvs(size=n_samples)
        elif distribution == "exponential":
            # For exponential, interpret "range" as (lambda1, lambda2) parameters
            lambda1, lambda2 = self.active_range
            x1 = np.random.exponential(1 / lambda1, size=n_samples)
            x2 = np.random.exponential(1 / lambda2, size=n_samples)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return x1, x2

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data using the operator specification"""

        # Support both single distribution and multiple distributions
        if isinstance(self.distribution, str):
            distributions = [self.distribution]
        else:
            distributions = self.distribution

        # Validate distributions
        supported_distributions = ["uniform", "truncated_normal", "exponential"]
        for dist in distributions:
            if dist not in supported_distributions:
                raise ValueError(
                    f"Unsupported distribution: {dist}. Supported: {supported_distributions}"
                )

        # Divide samples evenly across distributions
        n_per_dist = self.n_samples // len(distributions)
        remaining_samples = self.n_samples % len(distributions)

        all_x1 = []
        all_x2 = []

        for i, dist in enumerate(distributions):
            current_n_samples = n_per_dist + (1 if i < remaining_samples else 0)
            if self.verbose:
                print(
                    f"Generating {current_n_samples} samples from '{dist}' distribution..."
                )

            # Generate base inputs
            x1, x2 = self._generate_base_inputs(current_n_samples, dist)

            # Apply precision limits if specified
            if self.precision_limit is not None:
                x1 = bound_precision(x1, self.precision_limit)
                x2 = bound_precision(x2, self.precision_limit)

            all_x1.append(x1)
            all_x2.append(x2)

        # Concatenate all distributions
        x1_combined = np.concatenate(all_x1)
        x2_combined = np.concatenate(all_x2)

        # Shuffle to mix distributions
        indices = np.random.permutation(len(x1_combined))
        x1_combined = x1_combined[indices]
        x2_combined = x2_combined[indices]

        # Apply operation-specific constraints
        x1_combined, x2_combined = self.spec.input_constraints.apply_constraints(
            x1_combined, x2_combined, self.active_range, strict=self.strict_constraints
        )

        # Compute targets using the operator spec
        y = self.spec.compute_targets(x1_combined, x2_combined)

        # Handle unary operations (set x2 to neutral value)
        if self.spec.arity == 1:
            x2_combined = np.ones_like(x2_combined)  # Neutral value for unary ops

        # Convert to tensors
        X = (
            torch.from_numpy(np.stack([x1_combined, x2_combined], axis=1))
            .double()
            .to(self.device, dtype=self.dtype)
        )
        y_tensor = (
            torch.from_numpy(y).double().unsqueeze(1).to(self.device, dtype=self.dtype)
        )

        # Print stats
        min_seen = min(np.min(x1_combined), np.min(x2_combined))
        max_seen = max(np.max(x1_combined), np.max(x2_combined))

        if self.verbose:
            print(f"Generated {self.n_samples} total samples for {self.spec.name}:")
            print(
                f"  Range: {self.active_range}, actual min: {min_seen:.2e}, max: {max_seen:.2e}"
            )
            print(f"  Distributions used: {distributions}")
            print(f"  Target range: {y.min():.2e} to {y.max():.2e}")

        return X, y_tensor

    def __len__(self) -> int:
        return len(self.data_y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_x[idx], self.data_y[idx]


def create_mathy_dataloaders(
    operator_spec,  # OperatorSpec
    distribution: Literal["uniform", "truncated_normal", "exponential"] = "uniform",
    train_range: Tuple[float, float] = (1, 15),
    test_range: Tuple[float, float] = (25, 200),
    batch_size: int = 128,
    exp_lambda: float = 0.2,
    train_samples: int = 64000,
    test_samples: int = 64000,
    num_workers: int = 0,
    pin_memory: bool = False,
    precision_limit: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42,
    strict_constraints: bool = False,
    **kwargs,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
]:
    """Create DataLoaders for a mathematical operation using OperatorSpec"""

    train_dataset = MathyOperatorDataset(
        operator_spec=operator_spec,
        distribution=distribution,
        n_samples=train_samples,
        exp_lambda=exp_lambda,
        train_range=train_range,
        test_range=test_range,
        precision_limit=precision_limit,
        is_test=False,
        seed=seed,
        device=device,
        dtype=dtype,
        strict_constraints=strict_constraints,
        **kwargs,
    )

    test_dataset = MathyOperatorDataset(
        operator_spec=operator_spec,
        distribution=distribution,
        exp_lambda=exp_lambda,
        n_samples=test_samples,
        train_range=train_range,
        test_range=test_range,
        precision_limit=precision_limit,
        is_test=True,
        device=device,
        dtype=dtype,
        seed=seed,
        strict_constraints=strict_constraints,
        **kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
