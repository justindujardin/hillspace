import numpy as np
import matplotlib.pyplot as plt


def plot_hill_space_cross_sections(figsize=(12, 5), dpi=300):
    """
    Side-by-side cross-section plots showing how Hill Space constraint
    W = tanh(W_hat) x sigmoid(M_hat) varies with each parameter.
    """

    # Parameter ranges - extended to show saturation
    w_range = np.linspace(-15, 15, 300)
    m_range = np.linspace(-15, 15, 300)

    # Cross-section values: outliers at ±15, reasonable spacing in normal ranges
    tanh_cross_section_vals = [-15, -2, 0, 2, 15]
    sigmoid_cross_section_vals = [-15, -1, 0, 1, 15]

    # Color palette - using viridis-like colors for consistency
    colors = ["#440154", "#31688e", "#35b779", "#fde725", "#ff6b6b"]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Left plot: Cross-sections at different M_hat values
    # Shows how tanh(W_hat) is scaled by sigmoid(M_hat)
    for i, m_val in enumerate(tanh_cross_section_vals):
        sigmoid_m = 1 / (1 + np.exp(-m_val))
        weights = np.tanh(w_range) * sigmoid_m
        ax1.plot(
            w_range,
            weights,
            color=colors[i],
            linewidth=2.5,
            label=rf"$\hat{{M}}$ = {m_val}",
            alpha=0.8,
        )
    ax1.set_xlabel(r"$\hat{W}$  (signed-value knob)", fontsize=11)
    ax1.set_ylabel(r"Constrained weight $W$", fontsize=12)
    ax1.set_title(r"$\hat W$ Chooses Sign & Extent", fontsize=13, pad=14)
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.legend(frameon=True, fontsize=9, loc="upper left")
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-1.1, 1.1)

    # Add reference lines for key values
    ax1.axhline(y=1, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(y=-1, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.text(13, 1.05, "+1", fontsize=10, color="red", alpha=0.7)
    ax1.text(13, -1.05, "-1", fontsize=10, color="red", alpha=0.7)
    ax1.text(13, 0.05, "0", fontsize=10, color="gray", alpha=0.7)

    # Right plot: Cross-sections at different W_hat values
    # Shows how sigmoid(M_hat) selects the magnitude of tanh(W_hat)
    for i, w_val in enumerate(sigmoid_cross_section_vals):
        tanh_w = np.tanh(w_val)
        weights = tanh_w * (1 / (1 + np.exp(-m_range)))
        ax2.plot(
            m_range,
            weights,
            color=colors[i],
            linewidth=2.5,
            label=rf"$\hat{{W}}$ = {w_val}",
            alpha=0.8,
        )

    ax2.set_xlabel(r"$\hat{M}$  (gate strength)", fontsize=11)
    ax2.set_ylabel(r"Constrained weight $W$", fontsize=12)
    ax2.set_title(r"$\hat M$ Gates Magnitude", fontsize=13, pad=14)
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.legend(frameon=True, fontsize=9, loc="upper left")
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-1.1, 1.1)

    # Add reference lines
    ax2.axhline(y=1, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=-1, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(13, 1.05, "+1", fontsize=10, color="red", alpha=0.7)
    ax2.text(13, -1.05, "-1", fontsize=10, color="red", alpha=0.7)
    ax2.text(13, 0.05, "0", fontsize=10, color="gray", alpha=0.7)

    # Clean up the layout
    plt.tight_layout()

    # Add a subtle background color
    fig.patch.set_facecolor("white")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#fafafa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

    return fig


if __name__ == "__main__":
    filename = "hill_space_cross_sections.svg"
    fig = plot_hill_space_cross_sections()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved → {filename}")

