import json
import math
import os
from typing import Literal

import torch

from safetensors.torch import load_file, save_file


MathySpace = Literal["hill", "soft", "log", "quad", "cube", "hill_snap"]


def snapping_tanh(x, precision_threshold=1e-6):
    raw_tanh = torch.tanh(x)

    # If we're within precision_threshold of saturation, snap to exact values
    # tanh(15.0) ≈ 0.999999642, so we need threshold > 3.6e-7
    upper_snap_mask = raw_tanh > (1.0 - precision_threshold)
    lower_snap_mask = raw_tanh < (-1.0 + precision_threshold)

    result = raw_tanh.clone()
    result[upper_snap_mask] = 1.0  # Exact unity!
    result[lower_snap_mask] = -1.0  # Exact negative unity!

    return result


def snapping_sigmoid(x, precision_threshold=1e-6):
    raw_sigmoid = torch.sigmoid(x)

    # Snap to exact values near saturation
    # sigmoid(15.0) ≈ 0.999999694, so we need threshold > 3.1e-7
    upper_snap_mask = raw_sigmoid > (1.0 - precision_threshold)
    lower_snap_mask = raw_sigmoid < precision_threshold

    result = raw_sigmoid.clone()
    result[upper_snap_mask] = 1.0  # Exact unity!
    result[lower_snap_mask] = 0.0  # Exact zero!

    return result


class MathyUnit(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 1,
        dtype: torch.dtype = torch.float64,
        space: MathySpace = "hill",
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.space = space
        self.init_scale = init_scale

        # 2 parameter transformations for Additive/Exponential/Trigonometric operations
        #
        #
        # Combining operations (addition vs multiplication)
        self.W_combine_hat = self._new_parameter()
        self.M_combine_hat = self._new_parameter()
        # Combining operations (addition vs multiplication)
        self.W_add_hat = self._new_parameter()
        self.M_add_hat = self._new_parameter()
        # Separating operations (difference vs ratio)
        self.W_separate_hat = self._new_parameter()
        self.M_separate_hat = self._new_parameter()
        # Positive directional (additive/exponential identities, cos)
        self.W_positive_hat = self._new_parameter()
        self.M_positive_hat = self._new_parameter()
        # Negative directional primitive (negation, reciprocal, sin)
        self.W_negative_hat = self._new_parameter()
        self.M_negative_hat = self._new_parameter()
        # Left side negative (sin(θ))
        self.W_lnegative_hat = self._new_parameter()
        self.M_lnegative_hat = self._new_parameter()

        # Trigonometric product operations
        self.W_cos_sub_hat = self._new_parameter()  # cos(θ₁-θ₂)
        self.M_cos_sub_hat = self._new_parameter()
        self.W_sin_sub_hat = self._new_parameter()  # sin(θ₁-θ₂)
        self.M_sin_sub_hat = self._new_parameter()
        self.W_cos_add_hat = self._new_parameter()  # cos(θ₁+θ₂)
        self.M_cos_add_hat = self._new_parameter()
        self.W_sin_add_hat = self._new_parameter()  # sin(θ₁+θ₂)
        self.M_sin_add_hat = self._new_parameter()

        # Set config for saving/loading
        self.config: dict[str, str | int | float] = {
            "input_size": input_size,
            "output_size": output_size,
            "dtype": str(dtype),
            "space": space,
            "init_scale": init_scale,
        }
        print(
            f"MathyUnit initialized with {sum(p.numel() for p in self.parameters())} parameters"
        )

    def _new_parameter(self, in_size: int | None = None) -> torch.nn.Parameter:
        if in_size is None:
            in_size = self.input_size
        return torch.nn.Parameter(
            torch.randn(in_size, self.output_size, dtype=self.dtype) * self.init_scale,
        )

    def get_arithmetic_parameters(self) -> torch.nn.ParameterList:
        return torch.nn.ParameterList([p for p in self.parameters() if p.requires_grad])

    def calculate_weight(
        self,
        W_hat: torch.Tensor,
        M_hat: torch.Tensor,
        tau: float = 1.0,  # used by "soft"
        eps: float = 1e-9,
        beta_quad: float = 4.0,  # slope for "quad"
    ) -> torch.Tensor:
        """
        Map unbounded (Ŵ, M̂) to constrained weight W.

        Supported spaces:
        hill  - NALU tanh x sigmoid
        soft  - soft-mix, smooth (-1,1)
        log   - exponent in [0.08, 1.65]
        quad  - four plateaus 0, 1/3, 2/3, 1
        cube  - same plateaus via 2-gate blend
        """
        if self.space == "hill":
            return torch.tanh(W_hat) * torch.sigmoid(M_hat)

        elif self.space == "hill_snap":
            W = snapping_tanh(W_hat) * snapping_sigmoid(M_hat)
            return W
        elif self.space == "soft":
            a = torch.exp(W_hat / tau)
            b = torch.exp(M_hat / tau)
            return (a - b) / (1.0 + a + b + eps)

        elif self.space == "log":
            W = torch.tanh(W_hat) * torch.sigmoid(M_hat)
            log_exp = W * 1.5 - 1.0  # [-2.5, 0.5]
            return torch.exp(log_exp)  # [0.08, 1.65]

        elif self.space == "quad":
            # three independent gates → 4 attractor plateaus
            g1 = torch.sigmoid(beta_quad * W_hat)  # vertical gate
            g2 = torch.sigmoid(beta_quad * M_hat)  # horizontal gate
            g3 = torch.sigmoid(beta_quad * 0.5 * (W_hat + M_hat))  # diagonal gate
            return (g1 + g2 + g3) / 3.0

        elif self.space == "cube":
            s = torch.sigmoid(W_hat)
            t = torch.sigmoid(M_hat)
            return (2.0 / 3.0) * s + (1.0 / 3.0) * t

        else:
            raise ValueError(
                f"Unknown space: {self.space}. "
                "Choose from 'hill', 'soft', 'log', 'quad', 'cube'."
            )

    def _exponential_primitive(
        self, x: torch.Tensor, W_hat: torch.Tensor, M_hat: torch.Tensor
    ) -> torch.Tensor:
        """Universal exponential operation: x1^w1 x x2^w2"""
        W = self.calculate_weight(W_hat, M_hat)
        # Convert to complex to handle negative bases with fractional exponents
        x_complex = x.to(torch.complex128)
        # Compute x^w using complex arithmetic: [batch, input_size, output_size]
        powered = torch.pow(x_complex.unsqueeze(-1), W.unsqueeze(0))
        # Take product across input dimensions: [batch, output_size]
        result_complex = torch.prod(powered, dim=1)
        # Convert back to real - for valid arithmetic operations, imaginary part should be ~0
        return result_complex.real

    def _trigonometric_product_primitive(
        self, x: torch.Tensor, W_hat: torch.Tensor, M_hat: torch.Tensor
    ) -> torch.Tensor:
        W = self.calculate_weight(W_hat, M_hat)

        # Compute trigonometric components
        cos1, sin1 = torch.cos(x[:, 0:1]), torch.sin(x[:, 0:1])
        cos2, sin2 = torch.cos(x[:, 1:2]), torch.sin(x[:, 1:2])

        # Four fundamental trigonometric products
        cos_diff = cos1 * cos2 + sin1 * sin2  # cos(θ₁ - θ₂)
        cos_sum = cos1 * cos2 - sin1 * sin2  # cos(θ₁ + θ₂)
        sin_diff = sin1 * cos2 - cos1 * sin2  # sin(θ₁ - θ₂)
        sin_sum = sin1 * cos2 + cos1 * sin2  # sin(θ₁ + θ₂)

        # W[0]: cos vs sin selection, W[1]: sum vs diff selection
        result = W[0] * (W[1] * cos_diff + (1 - W[1]) * cos_sum) + (1 - W[0]) * (
            W[1] * sin_diff + (1 - W[1]) * sin_sum
        )
        return result

    def _trigonometric_primitive(
        self,
        x: torch.Tensor,
        W_hat: torch.Tensor,
        M_hat: torch.Tensor,
    ):
        """Trigonometric primitive: learns unary trigonometric transformations"""
        W = self.calculate_weight(W_hat, M_hat)
        # TODO: assert is unary (but we pad so shape won't tell us)
        theta = x[:, 0:1]

        # Use same weights for trigonometric selection
        cos_sin_selector = W[0, :]  # [-1,1]: -1=sin, +1=cos, 0=mix
        phase_shift = W[1, :] * math.pi  # Phase shift

        # Apply phase shift
        shifted_theta = theta + phase_shift

        # Compute both components of unit circle
        cos_component = torch.cos(shifted_theta)
        sin_component = torch.sin(shifted_theta)

        # Select which component based on weight
        # When cos_sin_selector = 1: pure cos
        # When cos_sin_selector = -1: pure sin
        # When cos_sin_selector = 0: equal mix
        result = (
            cos_component * (1 + cos_sin_selector)
            + sin_component * (1 - cos_sin_selector)
        ) / 2

        return result

    def _additive_primitive(
        self, x: torch.Tensor, W_hat: torch.Tensor, M_hat: torch.Tensor
    ) -> torch.Tensor:
        """Linear combination: matrix multiplication in Hill Space"""
        W = self.calculate_weight(W_hat, M_hat)
        result = torch.matmul(x, W)
        return result

    def inspect_weights(self, operation: str) -> list[float]:
        """Debug util for easy weight access based on operation names."""
        if operation == "add":
            W_hat = self.W_add_hat
            M_hat = self.M_add_hat
        elif operation == "subtract":
            W_hat = self.W_separate_hat
            M_hat = self.M_separate_hat
        elif operation == "multiply":
            W_hat = self.W_combine_hat
            M_hat = self.M_combine_hat
        elif operation == "divide":
            W_hat = self.W_separate_hat
            M_hat = self.M_separate_hat
        elif operation == "negation":
            W_hat = self.W_negative_hat
            M_hat = self.M_negative_hat
        elif operation == "identity":
            W_hat = self.W_positive_hat
            M_hat = self.M_positive_hat
        elif operation == "reciprocal":
            W_hat = self.W_negative_hat
            M_hat = self.M_negative_hat
        elif operation == "cos":
            W_hat = self.W_positive_hat
            M_hat = self.M_positive_hat
        elif operation == "sin":
            W_hat = self.W_lnegative_hat
            M_hat = self.M_lnegative_hat
        elif operation == "cos_add":
            W_hat = self.W_cos_add_hat
            M_hat = self.M_cos_add_hat
        elif operation == "sin_add":
            W_hat = self.W_sin_add_hat
            M_hat = self.M_sin_add_hat
        elif operation == "cos_sub":
            W_hat = self.W_cos_sub_hat
            M_hat = self.M_cos_sub_hat
        elif operation == "sin_sub":
            W_hat = self.W_sin_sub_hat
            M_hat = self.M_sin_sub_hat
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return self.calculate_weight(W_hat, M_hat).detach().flatten().tolist()

    def forward(self, x: torch.Tensor, operation: str) -> torch.Tensor:
        """Perform arithmetic operation on input tensor x with Hill Space.

        Universal transformation patterns:
        - combine [1,1]: addition, multiplication
        - separate [1,-1]: subtraction, division
        - positive [1,0]: identity, identity, cos(θ)
        - negative [0,-1]: negation, reciprocal
        - lnegative: [-1, 0]: sin(θ)
        """
        additive_ops = {
            "add": [self.W_combine_hat, self.M_combine_hat],
            "subtract": [self.W_separate_hat, self.M_separate_hat],
            "negation": [self.W_negative_hat, self.M_negative_hat],
            # "identity": [self.W_positive_hat, self.M_positive_hat],
        }
        exponential_ops = {
            "multiply": [self.W_combine_hat, self.M_combine_hat],
            "divide": [self.W_separate_hat, self.M_separate_hat],
            "identity": [self.W_positive_hat, self.M_positive_hat],
            "reciprocal": [self.W_negative_hat, self.M_negative_hat],
            # "binary_reciprocal": [self.W_lnegative_hat, self.M_lnegative_hat],
        }
        trig_ops = {
            "cos": [self.W_positive_hat, self.M_positive_hat],
            "sin": [self.W_lnegative_hat, self.M_lnegative_hat],
        }
        trig_product_ops = {
            "cos_add": [self.W_cos_add_hat, self.M_cos_add_hat],
            "sin_add": [self.W_sin_add_hat, self.M_sin_add_hat],
            "cos_sub": [self.W_cos_sub_hat, self.M_cos_sub_hat],
            "sin_sub": [self.W_sin_sub_hat, self.M_sin_sub_hat],
        }
        if operation in additive_ops:
            final_result = self._additive_primitive(x, *additive_ops[operation])
        elif operation in exponential_ops:
            final_result = self._exponential_primitive(x, *exponential_ops[operation])
        elif operation in trig_ops:
            final_result = self._trigonometric_primitive(x, *trig_ops[operation])
        elif operation in trig_product_ops:
            final_result = self._trigonometric_product_primitive(
                x, *trig_product_ops[operation]
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")
        return final_result

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        weight_keys = set(
            [
                n.replace("W_", "").replace("M_", "").replace("_hat", "")
                for n, p in self.named_parameters()
                if p.requires_grad
            ]
        )

        get_flat = lambda tensor: tensor.cpu().flatten().detach().numpy().tolist()
        metadata = {
            "model_type": "MathyUnit",
            "version": "0.1.0",
            "total_parameters": int(sum(p.numel() for p in self.parameters())),
            "parameters": {
                op: [
                    get_flat(getattr(self, f"W_{op}_hat")),
                    get_flat(getattr(self, f"M_{op}_hat")),
                ]
                for op in weight_keys
            },
        }

        metadata_path = os.path.join(save_directory, "weights.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Model saved to {metadata_path} with {len(weight_keys)} operations.")

    @classmethod
    def from_pretrained(cls, model_directory: str, device: str = "cpu"):
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        dtype_str = config["dtype"]
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str[len("torch.") :]
        config["dtype"] = getattr(torch, dtype_str)
        model = cls(**config)
        model_path = os.path.join(model_directory, "model.safetensors")
        state_dict = load_file(model_path, device=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model
