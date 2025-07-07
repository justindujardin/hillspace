import numpy as np
from matplotlib import cm, pyplot as plt, colors 
from typing import Sequence, Callable, Optional, Tuple
import seaborn as sns  # noqa: F401

   


def _sample(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(xs, ys)
    return X, Y, np.tanh(X) * (1 / (1 + np.exp(-Y)))


def plot_landscape(
    xs: Optional[np.ndarray] = None,
    ys: Optional[np.ndarray] = None,
    out: str = "landscapes.png",
    elev: float = 28,  # isometric-ish
    azim: float = 135,  #   view angle
    dpi: int = 500,
    figsize_per_plot: Tuple[float, float] = (4.5, 4),
):
    if xs is None or ys is None:
        xs = ys = np.linspace(-15, 15, 5000)

    fig = plt.figure(
        figsize=(figsize_per_plot[0], figsize_per_plot[1]), dpi=dpi
    )
    X, Y, Z = _sample(xs, ys)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=False,
        alpha=1.0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.7))
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0, 0, 0, 0))  # transparent panes

    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

if __name__ == "__main__":
    plot_landscape(out="hill_space.png")
