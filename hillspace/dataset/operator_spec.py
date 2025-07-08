import numpy as np
from typing import Tuple, List, Callable, Optional
from abc import ABC, abstractmethod


class InputConstraints(ABC):
    """Base class for operation input constraints"""

    @abstractmethod
    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply constraints to generated inputs, returning corrected x1, x2"""
        pass


class OperatorSpec:
    """Specification for a mathematical operator"""

    def __init__(
        self,
        name: str,
        arity: int,
        compute_targets: Callable[[np.ndarray, np.ndarray], np.ndarray],
        input_constraints: Optional[InputConstraints] = None,
        preferred_distributions: Optional[List[str]] = None,
        default_ranges: Optional[dict] = None,
    ):
        self.name = name
        self.arity = arity
        self.compute_targets = compute_targets
        self.input_constraints = input_constraints or NoConstraints()
        self.preferred_distributions = preferred_distributions or [
            "uniform",
            "truncated_normal",
        ]
        self.default_ranges = default_ranges or {
            "uniform": {"train": (1, 5), "test": (10, 20)},
            "truncated_normal": {"train": (-2, 4), "test": (8, 10)},
        }


class NoConstraints(InputConstraints):
    """Default constraint that applies no restrictions"""

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return x1, x2


class PositiveInputs(InputConstraints):
    """Constraint for operations requiring positive inputs like sqrt"""

    def __init__(
        self,
        min_val: float = 1e-8,
        max_val: float = 1e12,
        fallback_range: Tuple[float, float] = (0.1, 100.0),
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.fallback_range = fallback_range

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Check if the active range contains invalid values
        if active_range[0] < 0 or active_range[1] < 0:
            if strict:
                raise ValueError(
                    f"Positive input operation received invalid range {active_range} with negative values. "
                    f"Expected positive range for operations like sqrt. Consider using range like {self.fallback_range}."
                )
            else:
                print(
                    f"  ⚠️  WARNING: Positive input operation received range {active_range} with negative values!"
                )
                print(
                    f"      🔧 Auto-fixing to use fallback range {self.fallback_range}"
                )
                print(
                    f"      💡 Use strict=True to raise an error instead of auto-fixing"
                )
            # Use fallback range for regime sampling instead
            sampling_range = self.fallback_range
        else:
            sampling_range = active_range
        # Make inputs positive and bound them
        x1 = np.abs(x1)
        x2 = np.abs(x2)
        x1 = np.maximum(x1, self.min_val)
        x2 = np.maximum(x2, self.min_val)
        x1 = np.minimum(x1, self.max_val)
        x2 = np.minimum(x2, self.max_val)

        # Add targeted sampling for dual regime testing
        n_samples = len(x1)
        n_small_regime = min(n_samples // 4, 5000)
        n_large_regime = min(n_samples // 4, 5000)
        n_boundary = min(n_samples // 8, 2000)

        # Small regime samples (0 < x < 1): sqrt(x) > x (magnification)
        small_regime_samples = np.random.uniform(self.min_val, 1.0, n_small_regime)

        # Large regime samples (x > 1): sqrt(x) < x (shrinking)
        large_regime_samples = np.random.uniform(
            1.0, max(100.0, sampling_range[1]), n_large_regime
        )

        # Around x=1 boundary samples (critical transition point)
        boundary_samples = np.random.uniform(0.5, 2.0, n_boundary)

        # Replace some samples to ensure good regime coverage
        total_special = n_small_regime + n_large_regime + n_boundary
        if len(x1) > total_special:
            replace_indices = np.random.choice(len(x1), total_special, replace=False)

            idx = 0
            x1[replace_indices[idx : idx + n_small_regime]] = small_regime_samples
            idx += n_small_regime

            x1[replace_indices[idx : idx + n_large_regime]] = large_regime_samples
            idx += n_large_regime

            x1[replace_indices[idx : idx + n_boundary]] = boundary_samples

        print(f"  Sqrt input range: {x1.min():.2e} to {x1.max():.2e}")

        # Report regime distribution
        small_count = np.sum(x1 < 1.0)
        large_count = np.sum(x1 > 1.0)
        boundary_count = np.sum((x1 >= 0.9) & (x1 <= 1.1))
        print(
            f"  Sqrt regime distribution - Small (<1): {small_count}, Large (>1): {large_count}, Boundary (0.9-1.1): {boundary_count}"
        )

        return x1, x2


class NonZeroDivisor(InputConstraints):
    """Constraint for division operation - avoid divide by zero and extreme ratios"""

    def __init__(self, min_divisor_abs: float = 1e-6, max_division_ratio: float = 1e3):
        self.min_divisor_abs = min_divisor_abs
        self.max_division_ratio = max_division_ratio

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # First, ensure no tiny divisors (absolute constraint)
        small_divisor_mask = np.abs(x2) < self.min_divisor_abs
        n_small = np.sum(small_divisor_mask)
        if n_small > 0:
            # Replace with random values in safe range
            safe_min = max(self.min_divisor_abs, active_range[0] * 0.1)
            safe_max = active_range[1]
            x2[small_divisor_mask] = np.random.uniform(
                safe_min, safe_max, size=n_small
            ) * np.random.choice(
                [-1, 1], size=n_small
            )  # Preserve some negative values

        # Second, constrain ratios to prevent pathological results
        ratios = np.abs(x1 / x2)  # Use abs for ratio magnitude

        # Fix ratios that are too large (> max_ratio)
        too_large_mask = ratios > self.max_division_ratio
        n_large = np.sum(too_large_mask)
        if n_large > 0:
            # Randomly choose to scale x1 down or x2 up (50/50 split)
            scale_x1_mask = np.random.random(n_large) < 0.5
            scale_x2_mask = ~scale_x1_mask

            # Scale x1 down: x1_new = x2 * random_ratio, preserving sign
            large_and_scale_x1 = too_large_mask.copy()
            large_and_scale_x1[too_large_mask] = scale_x1_mask
            if np.any(large_and_scale_x1):
                target_ratios = np.random.uniform(
                    1, self.max_division_ratio, np.sum(large_and_scale_x1)
                )
                signs = np.sign(x1[large_and_scale_x1])
                x1[large_and_scale_x1] = (
                    np.abs(x2[large_and_scale_x1]) * target_ratios * signs
                )

            # Scale x2 up: x2_new = x1 / random_ratio, preserving sign
            large_and_scale_x2 = too_large_mask.copy()
            large_and_scale_x2[too_large_mask] = scale_x2_mask
            if np.any(large_and_scale_x2):
                target_ratios = np.random.uniform(
                    1, self.max_division_ratio, np.sum(large_and_scale_x2)
                )
                signs = np.sign(x2[large_and_scale_x2])
                x2[large_and_scale_x2] = (
                    np.abs(x1[large_and_scale_x2]) / target_ratios * signs
                )

        # Third, fix ratios that are too small (< 1/max_ratio)
        too_small_mask = ratios < (1.0 / self.max_division_ratio)
        n_small_ratio = np.sum(too_small_mask)
        if n_small_ratio > 0:
            scale_x1_mask = np.random.random(n_small_ratio) < 0.5
            scale_x2_mask = ~scale_x1_mask

            # Scale x1 up: x1_new = x2 * random_ratio
            small_and_scale_x1 = too_small_mask.copy()
            small_and_scale_x1[too_small_mask] = scale_x1_mask
            if np.any(small_and_scale_x1):
                target_ratios = np.random.uniform(
                    1.0 / self.max_division_ratio, 1, np.sum(small_and_scale_x1)
                )
                signs = np.sign(x1[small_and_scale_x1])
                x1[small_and_scale_x1] = (
                    np.abs(x2[small_and_scale_x1]) * target_ratios * signs
                )

            # Scale x2 down: x2_new = x1 / random_ratio
            small_and_scale_x2 = too_small_mask.copy()
            small_and_scale_x2[too_small_mask] = scale_x2_mask
            if np.any(small_and_scale_x2):
                target_ratios = np.random.uniform(
                    1.0 / self.max_division_ratio, 1, np.sum(small_and_scale_x2)
                )
                signs = np.sign(x2[small_and_scale_x2])
                x2[small_and_scale_x2] = (
                    np.abs(x1[small_and_scale_x2]) / target_ratios * signs
                )

        return x1, x2


class ZeroCrossingConstraints(InputConstraints):
    """Constraint for sign/abs operations - ensure good coverage around zero"""

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(x1)

        # Add targeted sampling around x=0 (the critical discontinuity point)
        n_around_zero = min(n_samples // 4, 5000)
        around_zero_samples = np.random.uniform(-1.0, 1.0, n_around_zero)

        # Ensure good coverage of negative regime
        n_negative = min(n_samples // 3, 8000)
        negative_samples = np.random.uniform(
            min(-100.0, active_range[0]), -0.1, n_negative
        )

        # Ensure good coverage of positive regime
        n_positive = min(n_samples // 3, 8000)
        positive_samples = np.random.uniform(
            0.1, max(100.0, active_range[1]), n_positive
        )

        # Replace some samples to ensure regime coverage
        total_special = n_around_zero + n_negative + n_positive
        if len(x1) > total_special:
            replace_indices = np.random.choice(len(x1), total_special, replace=False)

            idx = 0
            x1[replace_indices[idx : idx + n_around_zero]] = around_zero_samples
            idx += n_around_zero

            x1[replace_indices[idx : idx + n_negative]] = negative_samples
            idx += n_negative

            x1[replace_indices[idx : idx + n_positive]] = positive_samples

        print(f"  Zero-crossing input range: {x1.min():.2e} to {x1.max():.2e}")

        # Report regime distribution
        negative_count = np.sum(x1 < 0.0)
        positive_count = np.sum(x1 > 0.0)
        zero_region_count = np.sum(np.abs(x1) < 0.1)
        print(
            f"  Regime distribution - Negative: {negative_count}, Positive: {positive_count}, Near-zero: {zero_region_count}"
        )

        return x1, x2


class BoundedConstraints(InputConstraints):
    """Constraint for operations that need bounded inputs to avoid numerical issues"""

    def __init__(self, min_val: float = -1e9, max_val: float = 1e9):
        self.min_val = min_val
        self.max_val = max_val

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x1 = np.clip(x1, self.min_val, self.max_val)
        x2 = np.clip(x2, self.min_val, self.max_val)

        print(
            f"  Bounded input range: x1=[{x1.min():.2e}, {x1.max():.2e}], x2=[{x2.min():.2e}, {x2.max():.2e}]"
        )
        return x1, x2


class PositivePairConstraints(InputConstraints):
    """Constraint for operations requiring both inputs to be positive (like geometric mean)"""

    def __init__(self, min_val: float = 1e-8, max_val: float = 1e6):
        self.min_val = min_val
        self.max_val = max_val

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Make both inputs positive
        x1 = np.abs(x1)
        x2 = np.abs(x2)
        x1 = np.maximum(x1, self.min_val)
        x2 = np.maximum(x2, self.min_val)

        # Reasonable upper bound to avoid numerical issues
        x1 = np.minimum(x1, self.max_val)
        x2 = np.minimum(x2, self.max_val)

        print(
            f"  Positive pair input ranges: x1=[{x1.min():.2e}, {x1.max():.2e}], x2=[{x2.min():.2e}, {x2.max():.2e}]"
        )
        return x1, x2


class TangentConstraints(InputConstraints):
    """Constraint for tangent function - avoid asymptotes at π/2 + nπ"""

    def __init__(self, asymptote_buffer: float = 0.1):
        self.asymptote_buffer = asymptote_buffer

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Find inputs too close to asymptotes (π/2 + nπ)
        # Asymptotes are at: ..., -3π/2, -π/2, π/2, 3π/2, ...

        # Calculate distance to nearest asymptote
        asymptote_positions = (np.round(x1 / np.pi - 0.5) + 0.5) * np.pi
        distances_to_asymptotes = np.abs(x1 - asymptote_positions)

        # Find points too close to asymptotes
        too_close_mask = distances_to_asymptotes < self.asymptote_buffer
        n_too_close = np.sum(too_close_mask)

        if n_too_close > 0:
            # Replace with safe values in the active range, avoiding asymptotes
            safe_candidates = []
            attempts = 0
            while len(safe_candidates) < n_too_close and attempts < 1000:
                candidates = np.random.uniform(
                    active_range[0], active_range[1], n_too_close * 2
                )

                # Check distance to asymptotes for candidates
                candidate_asymptotes = (
                    np.round(candidates / np.pi - 0.5) + 0.5
                ) * np.pi
                candidate_distances = np.abs(candidates - candidate_asymptotes)

                # Keep candidates that are safe
                safe_indices = candidate_distances >= self.asymptote_buffer
                safe_candidates.extend(candidates[safe_indices])
                attempts += 1

            # Use the safe candidates we found
            replacement_values = np.array(safe_candidates[:n_too_close])
            x1[too_close_mask] = replacement_values

        # Add targeted sampling around interesting regions
        n_samples = len(x1)

        # Sample around zero (where tan(x) ≈ x)
        n_around_zero = min(n_samples // 4, 3000)
        around_zero_samples = np.random.uniform(-0.5, 0.5, n_around_zero)

        # Sample in "safe" regions away from asymptotes
        n_safe_regions = min(n_samples // 3, 5000)
        safe_regions = [
            (-np.pi / 3, -np.pi / 6),  # Left safe region
            (np.pi / 6, np.pi / 3),  # Right safe region
        ]

        safe_samples = []
        for region in safe_regions:
            region_samples = np.random.uniform(
                region[0], region[1], n_safe_regions // len(safe_regions)
            )
            safe_samples.extend(region_samples)

        safe_samples = np.array(safe_samples)

        # Replace some samples for better coverage
        total_special = n_around_zero + len(safe_samples)
        if len(x1) > total_special:
            replace_indices = np.random.choice(len(x1), total_special, replace=False)

            idx = 0
            x1[replace_indices[idx : idx + n_around_zero]] = around_zero_samples
            idx += n_around_zero

            remaining = len(safe_samples)
            x1[replace_indices[idx : idx + remaining]] = safe_samples

        # Verify no values are too close to asymptotes
        final_asymptotes = (np.round(x1 / np.pi - 0.5) + 0.5) * np.pi
        final_distances = np.abs(x1 - final_asymptotes)
        min_distance = np.min(final_distances)

        print(f"  Tangent input range: {x1.min():.3f} to {x1.max():.3f}")
        print(
            f"  Minimum distance to asymptotes: {min_distance:.3f} (buffer: {self.asymptote_buffer})"
        )

        # Report some sample values and their tangents for verification
        sample_indices = np.random.choice(len(x1), min(5, len(x1)), replace=False)
        for i in sample_indices:
            print(f"    tan({x1[i]:.3f}) = {np.tan(x1[i]):.3f}")

        return x1, x2


class InverseTrigConstraints(InputConstraints):
    """Constraint for inverse trig functions - input must be in [-1, 1] domain"""

    def __init__(
        self, min_val: float = -0.999, max_val: float = 0.999, buffer: float = 0.001
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.buffer = buffer  # Small buffer to avoid numerical issues at ±1

    def apply_constraints(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        active_range: Tuple[float, float],
        strict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Clip inputs to valid domain [-1, 1] with small buffer
        x1 = np.clip(x1, self.min_val, self.max_val)
        x2 = np.clip(x2, self.min_val, self.max_val)

        # Add targeted sampling for critical regions
        n_samples = len(x1)

        # Sample around ±1 (where derivatives blow up)
        n_boundary = min(n_samples // 4, 3000)
        boundary_samples = np.concatenate(
            [
                np.random.uniform(-1.0, -0.8, n_boundary // 2),  # Near -1
                np.random.uniform(0.8, 1.0, n_boundary // 2),  # Near +1
            ]
        )

        # Sample around 0 (where functions are most linear)
        n_around_zero = min(n_samples // 4, 3000)
        around_zero_samples = np.random.uniform(-0.3, 0.3, n_around_zero)

        # Sample intermediate regions
        n_intermediate = min(n_samples // 3, 4000)
        intermediate_samples = np.concatenate(
            [
                np.random.uniform(-0.8, -0.3, n_intermediate // 2),
                np.random.uniform(0.3, 0.8, n_intermediate // 2),
            ]
        )

        # Replace some samples for better domain coverage
        total_special = n_boundary + n_around_zero + n_intermediate
        if len(x1) > total_special:
            replace_indices = np.random.choice(len(x1), total_special, replace=False)

            idx = 0
            x1[replace_indices[idx : idx + n_boundary]] = boundary_samples
            idx += n_boundary

            x1[replace_indices[idx : idx + n_around_zero]] = around_zero_samples
            idx += n_around_zero

            x1[replace_indices[idx : idx + n_intermediate]] = intermediate_samples

        print(f"  Inverse trig input range: {x1.min():.3f} to {x1.max():.3f}")

        # Report domain distribution
        boundary_count = np.sum(np.abs(x1) > 0.8)
        zero_region_count = np.sum(np.abs(x1) < 0.3)
        intermediate_count = np.sum((np.abs(x1) >= 0.3) & (np.abs(x1) <= 0.8))
        print(
            f"  Domain distribution - Boundary (|x|>0.8): {boundary_count}, Zero region (|x|<0.3): {zero_region_count}, Intermediate: {intermediate_count}"
        )

        return x1, x2
