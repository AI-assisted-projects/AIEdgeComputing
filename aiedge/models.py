"""The neural network used throughout the course: a small multi-layer perceptron.

We deliberately use only ``Linear`` layers and ``ReLU`` activations: every
operation maps onto a few lines of plain C++ on the Arduino (see
``arduino/gesture_classifier/edge_nn.h``), so nothing is hidden by a framework.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class MLP(nn.Module):
    """``in_features -> hidden[0] -> hidden[1] -> ... -> n_classes``, ReLU in between.

    The network outputs *logits*: one raw score per gesture. Softmax turns them
    into probabilities; the largest logit is the prediction.

    ``dropout`` randomly switches off that fraction of hidden neurons during
    training only (a regularizer for big models); it has no effect on inference.
    """

    def __init__(self, hidden=(64, 32), in_features: int = C.N_FEATURES,
                 n_classes: int = C.N_CLASSES, dropout: float = 0.0):
        super().__init__()
        self.hidden = tuple(int(h) for h in hidden)
        sizes = [in_features, *self.hidden, n_classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:      # no ReLU after the last layer
                x = self.dropout(torch.relu(x))
        return x

    @property
    def sizes(self) -> list[int]:
        return [self.layers[0].in_features] + [l.out_features for l in self.layers]

    def __repr__(self) -> str:
        return f"MLP({' -> '.join(map(str, self.sizes))})"


def count_params(model: nn.Module) -> int:
    """Number of learnable numbers (weights + biases)."""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_params(model: nn.Module) -> int:
    return sum(int((p != 0).sum()) for p in model.parameters())


def count_macs(model: MLP) -> int:
    """Multiply-accumulate operations for ONE prediction (= number of weights)."""
    return sum(l.in_features * l.out_features for l in model.layers)


class ConvNet(nn.Module):
    """A 1-D convolutional network -- used only as a *teacher* for distillation.

    Convolutions slide the same small pattern detector along the time axis, so
    the network recognises a gesture no matter whether it happens a bit earlier
    or later in the window. It is more accurate than an MLP on this task, but
    also much more expensive to run -- exactly the kind of model you would like
    to *distill* into something small.
    """

    def __init__(self, channels=(32, 64), kernel_size: int = 5, dropout: float = 0.3,
                 n_classes: int = C.N_CLASSES):
        super().__init__()
        c1, c2 = channels
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(C.N_CHANNELS, c1, kernel_size, padding=pad), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size, padding=pad), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c2, kernel_size, padding=pad), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * c2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, 300) -> (N, 50 time steps, 6 channels) -> (N, 6, 50) for Conv1d
        x = x.view(-1, C.WINDOW_SAMPLES, C.N_CHANNELS).transpose(1, 2)
        x = self.features(x)
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)   # pool over time
        return self.head(self.dropout(x))


def count_macs_conv(model: ConvNet) -> int:
    """MACs of the ConvNet teacher for one prediction (for comparison tables)."""
    macs, length = 0, C.WINDOW_SAMPLES
    for m in model.features:
        if isinstance(m, nn.Conv1d):
            macs += m.in_channels * m.out_channels * m.kernel_size[0] * length
        elif isinstance(m, nn.MaxPool1d):
            length //= 2
    return macs + model.head.in_features * model.head.out_features
