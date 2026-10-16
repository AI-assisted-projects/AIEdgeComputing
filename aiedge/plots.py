"""Plotting helpers for the notebooks (matplotlib)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config as C

# Categorical colours in a fixed order (colour-blind-checked palette).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})


def plot_gesture(window: np.ndarray, title: str = "", axes=None):
    """One gesture: accelerometer (top) and gyroscope (bottom) over time."""
    t = np.arange(len(window)) / C.SAMPLE_RATE_HZ
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    for k, name in enumerate("xyz"):
        axes[0].plot(t, window[:, k], color=PALETTE[k], ls=LINESTYLES[k], label=f"a{name}")
        axes[1].plot(t, window[:, 3 + k], color=PALETTE[k], ls=LINESTYLES[k], label=f"g{name}")
    axes[0].set_ylabel("acceleration [g]")
    axes[1].set_ylabel("rotation [deg/s]")
    axes[1].set_xlabel("time [s]")
    axes[0].set_title(title)
    return axes


def plot_examples(raw: np.ndarray, y: np.ndarray, per_class: int = 1, seed: int = 0):
    """A grid with ``per_class`` random examples of every gesture (one column per gesture)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2 * per_class, C.N_CLASSES, figsize=(3.2 * C.N_CLASSES, 3.6 * per_class),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for c, name in enumerate(C.GESTURES):
        idx = rng.choice(np.flatnonzero(y == c), per_class, replace=False)
        for r, i in enumerate(idx):
            plot_gesture(raw[i], name if r == 0 else "", axes=axes[2 * r:2 * r + 2, c])
            if c > 0:
                axes[2 * r, c].set_ylabel("")
                axes[2 * r + 1, c].set_ylabel("")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_history(history: dict, title: str = "Training"):
    """Loss (left) and accuracy (right) per epoch."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    ep = np.arange(1, len(history["train_loss"]) + 1)
    a1.plot(ep, history["train_loss"], color=PALETTE[0])
    a1.set_xlabel("epoch"); a1.set_ylabel("training loss"); a1.set_title(f"{title}: loss")
    a2.plot(ep, history["train_acc"], color=PALETTE[0], label="training set")
    a2.plot(ep, history["val_acc"], color=PALETTE[1], ls="--", label="validation set")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title(f"{title}: accuracy")
    a2.legend()
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, title: str = "Confusion matrix", ax=None):
    """Rows: true gesture, columns: predicted gesture. The diagonal holds the correct answers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(C.N_CLASSES), C.GESTURES, rotation=45)
    ax.set_yticks(range(C.N_CLASSES), C.GESTURES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=9)
    return ax


def plot_curves(x, curves: dict[str, list[float]], xlabel: str, ylabel: str = "test accuracy",
                title: str = "", ax=None, xscale: str = "linear"):
    """Several lines on one axis, each with its own colour, line style and marker."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.8))
    for k, (name, ys) in enumerate(curves.items()):
        ax.plot(x, ys, color=PALETTE[k], ls=LINESTYLES[k % len(LINESTYLES)],
                marker=MARKERS[k], markersize=6, label=name)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend()
    return ax
