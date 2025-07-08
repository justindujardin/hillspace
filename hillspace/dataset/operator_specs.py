from decimal import Decimal

import numpy as np

from .operator_spec import (
    BoundedConstraints,
    InverseTrigConstraints,
    NoConstraints,
    NonZeroDivisor,
    OperatorSpec,
    PositiveInputs,
    PositivePairConstraints,
    TangentConstraints,
    ZeroCrossingConstraints,
)

ADD_SPEC = OperatorSpec(
    name="add",
    arity=2,
    compute_targets=lambda x1, x2: np.array(
        [
            float(Decimal(str(float(a))) + Decimal(str(float(b))))
            for a, b in zip(x1, x2)
        ],
        dtype=np.float64,
    ),
    input_constraints=NoConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (1, 5), "test": (10, 20)},
        "truncated_normal": {"train": (-2, 4), "test": (8, 10)},
    },
)

SUBTRACT_SPEC = OperatorSpec(
    name="subtract",
    arity=2,
    compute_targets=lambda x1, x2: np.array(
        [
            float(Decimal(str(float(a))) - Decimal(str(float(b))))
            for a, b in zip(x1, x2)
        ],
        dtype=np.float64,
    ),
    input_constraints=NoConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (1, 5), "test": (10, 20)},
        "truncated_normal": {"train": (-2, 4), "test": (8, 10)},
    },
)

MULTIPLY_SPEC = OperatorSpec(
    name="multiply",
    arity=2,
    compute_targets=lambda x1, x2: np.array(
        [
            float(Decimal(str(float(a))) * Decimal(str(float(b))))
            for a, b in zip(x1, x2)
        ],
        dtype=np.float64,
    ),  # Use high-precision Decimal computation
    input_constraints=NoConstraints(),
    preferred_distributions=["uniform"],
    default_ranges={
        "uniform": {"train": (1, 5), "test": (10, 20)},
        "truncated_normal": {"train": (-2, 4), "test": (8, 10)},
    },
)

DIVIDE_SPEC = OperatorSpec(
    name="divide",
    arity=2,
    compute_targets=lambda x1, x2: np.array(
        [
            float(Decimal(str(float(a))) / Decimal(str(float(b))))
            for a, b in zip(x1, x2)
        ],
        dtype=np.float64,
    ),
    input_constraints=NonZeroDivisor(min_divisor_abs=1e-6, max_division_ratio=1e3),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (1, 5), "test": (10, 20)},
        "truncated_normal": {"train": (-2, 4), "test": (8, 10)},
    },
)

SQRT_SPEC = OperatorSpec(
    name="sqrt",
    arity=1,
    compute_targets=lambda x1, x2: np.sqrt(x1),
    input_constraints=PositiveInputs(min_val=1e-8, max_val=1e12),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (10, 100)},
        "truncated_normal": {"train": (0, 4), "test": (8, 20)},
    },
)

SIGN_SPEC = OperatorSpec(
    name="sign",
    arity=1,
    compute_targets=lambda x1, x2: np.sign(x1),
    input_constraints=ZeroCrossingConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

ABS_SPEC = OperatorSpec(
    name="abs",
    arity=1,
    compute_targets=lambda x1, x2: np.abs(x1),
    input_constraints=ZeroCrossingConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

GEOMEAN_SPEC = OperatorSpec(
    name="geomean",
    arity=2,
    compute_targets=lambda x1, x2: np.sqrt(x1 * x2),
    input_constraints=PositivePairConstraints(min_val=1e-8, max_val=1e6),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (10, 100)},
        "truncated_normal": {"train": (0, 4), "test": (8, 20)},
    },
)

CUBE_SPEC = OperatorSpec(
    name="cbrt",
    arity=1,
    compute_targets=lambda x1, x2: np.cbrt(x1),
    input_constraints=BoundedConstraints(min_val=-1e9, max_val=1e9),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

FOURTH_ROOT_SPEC = OperatorSpec(
    name="fourth_root",
    arity=1,
    compute_targets=lambda x1, x2: np.power(x1, 0.25),
    input_constraints=PositiveInputs(min_val=1e-8, max_val=1e8),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

THREE_FOURTHS_POWER_SPEC = OperatorSpec(
    name="three_fourths_power",
    arity=1,
    compute_targets=lambda x1, x2: np.power(x1, 0.75),
    input_constraints=PositiveInputs(min_val=1e-8, max_val=1e8),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

COS_SPEC = OperatorSpec(
    name="cos",
    arity=1,
    compute_targets=lambda x1, x2: np.cos(x1),
    input_constraints=NoConstraints(),  # Cosine works for any real input
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi, np.pi), "test": (-4 * np.pi, 4 * np.pi)},
        "truncated_normal": {
            "train": (-2 * np.pi, 2 * np.pi),
            "test": (-6 * np.pi, 6 * np.pi),
        },
    },
)

COS_NEGATIVE_SPEC = OperatorSpec(
    name="cos_negate",
    arity=1,
    compute_targets=lambda x1, x2: -np.cos(x1),  # Negated cosine
    input_constraints=NoConstraints(),  # Cosine works for any real input
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi, np.pi), "test": (-4 * np.pi, 4 * np.pi)},
        "truncated_normal": {
            "train": (-2 * np.pi, 2 * np.pi),
            "test": (-6 * np.pi, 6 * np.pi),
        },
    },
)

SIN_SPEC = OperatorSpec(
    name="sin",
    arity=1,
    compute_targets=lambda x1, x2: np.sin(x1),
    input_constraints=NoConstraints(),  # Sine works for any real input
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi, np.pi), "test": (-4 * np.pi, 4 * np.pi)},
        "truncated_normal": {
            "train": (-2 * np.pi, 2 * np.pi),
            "test": (-6 * np.pi, 6 * np.pi),
        },
    },
)

TAN_SPEC = OperatorSpec(
    name="tan",
    arity=1,
    compute_targets=lambda x1, x2: np.tan(x1),
    input_constraints=TangentConstraints(),  # Need to avoid discontinuities
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {
            "train": (-np.pi / 3, np.pi / 3),
            "test": (-np.pi / 2.5, np.pi / 2.5),
        },
        "truncated_normal": {
            "train": (-np.pi / 4, np.pi / 4),
            "test": (-np.pi / 3, np.pi / 3),
        },
    },
)

NEGATION_SPEC = OperatorSpec(
    name="negation",
    arity=1,
    compute_targets=lambda x1, x2: -x1,  # Simply negate the first input
    input_constraints=NoConstraints(),  # Works for any real input
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

IDENTITY_SPEC = OperatorSpec(
    name="identity",
    arity=1,
    compute_targets=lambda x1, x2: x1,  # Return first input unchanged
    input_constraints=NoConstraints(),  # Works for any real input
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-5, 5), "test": (-20, 20)},
        "truncated_normal": {"train": (-3, 3), "test": (-10, 10)},
    },
)

RECIPROCAL_SPEC = OperatorSpec(
    name="reciprocal",
    arity=1,
    compute_targets=lambda x1, x2: 1.0 / x1,  # Return 1/x
    input_constraints=NonZeroDivisor(
        min_divisor_abs=1e-6, max_division_ratio=1e3
    ),  # Avoid divide by zero
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (10, 100)},  # Positive to avoid issues
        "truncated_normal": {"train": (0.5, 3), "test": (5, 20)},
    },
)

BINARY_RECIPROCAL_SPEC = OperatorSpec(
    name="binary_reciprocal",
    arity=2,
    compute_targets=lambda x1, x2: 1.0 / (x1 / x2),  # Return 1/(x₁ / x₂)
    input_constraints=NonZeroDivisor(
        min_divisor_abs=1e-6, max_division_ratio=1e3
    ),  # Avoid divide by zero
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (10, 100)},  # Positive to avoid issues
        "truncated_normal": {"train": (0.5, 3), "test": (5, 20)},
    },
)


NEG_COS_ADD_SPEC = OperatorSpec(
    name="neg_cos_add",
    arity=2,  # Two angles as input
    compute_targets=lambda x1, x2: -np.cos(x1 + x2),  # -cos(θ₁ + θ₂)
    input_constraints=NoConstraints(),  # Cosine works for any real angles
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)

NEG_COS_SUB_SPEC = OperatorSpec(
    name="neg_cos_sub",
    arity=2,  # Two angles as input
    compute_targets=lambda x1, x2: -np.cos(x1 - x2),  # -cos(θ₁ - θ₂)
    input_constraints=NoConstraints(),  # Cosine works for any real angles
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)

NEG_SIN_ADD_SPEC = OperatorSpec(
    name="neg_sin_add",
    arity=2,  # Two angles as input
    compute_targets=lambda x1, x2: -np.sin(x1 + x2),  # -sin(θ₁ + θ₂)
    input_constraints=NoConstraints(),  # Sine works for any real angles
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)

COS_SUB_SPEC = OperatorSpec(
    name="cos_sub",
    arity=2,  # Two angles as input
    compute_targets=lambda x1, x2: np.cos(x1 - x2),
    input_constraints=NoConstraints(),  # Cosine works for any real angles
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)
COS_ADD_SPEC = OperatorSpec(
    name="cos_add",
    arity=2,  # Two angles as input
    compute_targets=lambda x1, x2: np.cos(x1 + x2),
    input_constraints=NoConstraints(),  # Cosine works for any real angles
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)


SIN_ADD_SPEC = OperatorSpec(
    name="sin_add",
    arity=2,
    compute_targets=lambda x1, x2: np.sin(x1 + x2),  # sin(θ₁ + θ₂)
    input_constraints=NoConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)

SIN_SUB_SPEC = OperatorSpec(
    name="sin_sub",
    arity=2,
    compute_targets=lambda x1, x2: np.sin(x1 - x2),  # sin(θ₁ - θ₂)
    input_constraints=NoConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-np.pi / 2, np.pi / 2), "test": (-2 * np.pi, 2 * np.pi)},
        "truncated_normal": {"train": (-np.pi / 3, np.pi / 3), "test": (-np.pi, np.pi)},
    },
)


# Inverse trigonometric function specs
ARCSIN_SPEC = OperatorSpec(
    name="arc_sin",
    arity=1,
    compute_targets=lambda x1, x2: np.arcsin(
        x1
    ),  # Input: ratio ∈ [-1,1], Output: angle ∈ [-π/2, π/2]
    input_constraints=InverseTrigConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-0.9, 0.9), "test": (-0.99, 0.99)},
        "truncated_normal": {"train": (-0.7, 0.7), "test": (-0.95, 0.95)},
    },
)

ARCCOS_SPEC = OperatorSpec(
    name="arc_cos",
    arity=1,
    compute_targets=lambda x1, x2: np.arccos(
        x1
    ),  # Input: ratio ∈ [-1,1], Output: angle ∈ [0, π]
    input_constraints=InverseTrigConstraints(),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (-0.9, 0.9), "test": (-0.99, 0.99)},
        "truncated_normal": {"train": (-0.7, 0.7), "test": (-0.95, 0.95)},
    },
)

ARCTAN_SPEC = OperatorSpec(
    name="arc_tan",
    arity=1,
    compute_targets=lambda x1, x2: np.arctan(
        x1
    ),  # Input: any real, Output: angle ∈ (-π/2, π/2)
    input_constraints=NoConstraints(),  # arctan works for any real input
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {
            "train": (-10, 10),
            "test": (-100, 100),
        },  # Can handle larger ranges
        "truncated_normal": {"train": (-5, 5), "test": (-50, 50)},
    },
)

# Ratios
HARMONIC_MEAN_SPEC = OperatorSpec(
    name="harmonic_mean",
    arity=2,
    compute_targets=lambda x1, x2: 2 * x1 * x2 / (x1 + x2),  # 2ab/(a+b)
    input_constraints=PositivePairConstraints(
        min_val=1e-6, max_val=1e6
    ),  # Avoid division issues
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 10), "test": (0.01, 100)},
        "truncated_normal": {"train": (0.5, 5), "test": (0.1, 50)},
    },
)

PROPORTION_SPEC = OperatorSpec(
    name="proportion",
    arity=2,
    compute_targets=lambda x1, x2: x1
    / (x1 + x2),  # a/(a+b) - proportion of first to total
    input_constraints=PositivePairConstraints(min_val=1e-8, max_val=1e8),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (0.01, 50)},
        "truncated_normal": {"train": (0.5, 3), "test": (0.1, 20)},
    },
)

RELATIVE_DIFF_SPEC = OperatorSpec(
    name="relative_diff",
    arity=2,
    compute_targets=lambda x1, x2: (x1 - x2)
    / (x1 + x2),  # (a-b)/(a+b) - normalized difference
    input_constraints=PositivePairConstraints(
        min_val=1e-6, max_val=1e6
    ),  # Keep positive to avoid zero denominator
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (0.01, 20)},
        "truncated_normal": {"train": (0.5, 3), "test": (0.1, 15)},
    },
)

ASPECT_RATIO_SPEC = OperatorSpec(
    name="aspect_ratio",
    arity=2,
    compute_targets=lambda x1, x2: np.maximum(x1, x2)
    / np.maximum(np.minimum(x1, x2), 1e-8),  # max/min ratio
    input_constraints=PositivePairConstraints(min_val=1e-6, max_val=1e6),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 10), "test": (0.01, 100)},
        "truncated_normal": {"train": (0.5, 5), "test": (0.1, 50)},
    },
)

GEOMETRIC_RATIO_SPEC = OperatorSpec(
    name="geometric_ratio",
    arity=2,
    compute_targets=lambda x1, x2: np.sqrt(x1 * x2)
    / (x1 + x2),  # sqrt(ab)/(a+b) - geometric mean over arithmetic mean
    input_constraints=PositivePairConstraints(min_val=1e-6, max_val=1e6),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (0.01, 20)},
        "truncated_normal": {"train": (0.5, 3), "test": (0.1, 15)},
    },
)

SIMPLE_RATIO_SPEC = OperatorSpec(
    name="simple_ratio",
    arity=2,
    compute_targets=lambda x1, x2: x1 / x2,  # Just a/b - basic ratio
    input_constraints=NonZeroDivisor(min_divisor_abs=1e-6, max_division_ratio=1e3),
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (0.1, 5), "test": (0.01, 20)},
        "truncated_normal": {"train": (0.5, 3), "test": (0.1, 15)},
    },
)

CONTRAST_RATIO_SPEC = OperatorSpec(
    name="contrast_ratio",
    arity=2,
    compute_targets=lambda x1, x2: (x1 + x2)
    / (x1 - x2),  # (a+b)/(a-b) - complement of your primitive
    input_constraints=PositivePairConstraints(
        min_val=1e-6, max_val=1e6
    ),  # Ensure x1 > x2 for stability
    preferred_distributions=["uniform", "truncated_normal"],
    default_ranges={
        "uniform": {"train": (1, 10), "test": (0.5, 50)},  # Keep x1 > x2 generally
        "truncated_normal": {"train": (2, 8), "test": (1, 30)},
    },
)


OPERATION_REGISTRY = {
    "add": ADD_SPEC,
    "subtract": SUBTRACT_SPEC,
    "multiply": MULTIPLY_SPEC,
    "divide": DIVIDE_SPEC,
    "sqrt": SQRT_SPEC,
    "sign": SIGN_SPEC,
    "abs": ABS_SPEC,
    "geomean": GEOMEAN_SPEC,
    "cbrt": CUBE_SPEC,
    "fourth_root": FOURTH_ROOT_SPEC,
    "three_fourths_power": THREE_FOURTHS_POWER_SPEC,
    "cos": COS_SPEC,
    "cos_add": COS_ADD_SPEC,
    "cos_sub": COS_SUB_SPEC,
    "sin": SIN_SPEC,
    "tan": TAN_SPEC,
    "negation": NEGATION_SPEC,
    "identity": IDENTITY_SPEC,
    "reciprocal": RECIPROCAL_SPEC,
    "neg_cos_add": NEG_COS_ADD_SPEC,
    "neg_sin_add": NEG_SIN_ADD_SPEC,
    "neg_cos_sub": NEG_COS_SUB_SPEC,
    "sin_add": SIN_ADD_SPEC,
    "sin_sub": SIN_SUB_SPEC,
    "cos_negate": COS_NEGATIVE_SPEC,
    "arc_sin": ARCSIN_SPEC,
    "arc_cos": ARCCOS_SPEC,
    "arc_tan": ARCTAN_SPEC,
    "harmonic_mean": HARMONIC_MEAN_SPEC,
    "proportion": PROPORTION_SPEC,
    "relative_diff": RELATIVE_DIFF_SPEC,
    "aspect_ratio": ASPECT_RATIO_SPEC,
    "geometric_ratio": GEOMETRIC_RATIO_SPEC,
    "simple_ratio": SIMPLE_RATIO_SPEC,
    "contrast_ratio": CONTRAST_RATIO_SPEC,
    "binary_reciprocal": BINARY_RECIPROCAL_SPEC,
}


# Operation symbols for pretty printing
OPERATION_SYMBOLS = {
    "add": "+",
    "subtract": "-",
    "multiply": "x",
    "divide": "÷",
    "sqrt": "√",
    "sign": "sgn",
    "abs": "| |",
    "geomean": "√(ab)",
    "cube": "∛",
    "cos": "cos",
    "sin": "sin",
    "tan": "tan",
    "negation": "-",
    "identity": "id",
    "reciprocal": "1/x",
    "harmonicmean": "2ab/(a+b)",
    "proportion": "a/(a+b)",
    "relativediff": "(a-b)/(a+b)",
    "aspectratio": "max/min",
    "geometricratio": "√(ab)/(a+b)",
    "simpleratio": "a/b",
    "contrastratio": "(a+b)/(a-b)",
}
