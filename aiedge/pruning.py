"""Pruning: removing parts of a trained network.

Two flavours, named after what they cut in the network *graph* (neurons are
the nodes, weights are the edges between them):

* **Edge pruning** (a.k.a. *unstructured* or *weight* pruning) sets individual
  weights to zero. The matrices keep their shape; we just get many zeros.
* **Node pruning** (a.k.a. *structured* or *neuron* pruning) removes whole
  neurons. The weight matrices really become smaller, so a plain dense C
  implementation immediately becomes smaller and faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import MLP


# ---------------------------------------------------------------------------
# Edge (weight) pruning
# ---------------------------------------------------------------------------

def magnitude_masks(model: nn.Module, sparsity: float, scope: str = "global") -> dict[str, torch.Tensor]:
    """Masks (1 = keep, 0 = prune) that remove the ``sparsity`` fraction of weights
    with the smallest absolute value.

    ``scope="global"``  -- one threshold for all layers together (layers with many
                           unimportant weights lose more of them)
    ``scope="layer"``   -- every layer loses the same fraction
    Biases are never pruned: there are few of them and they are cheap.
    """
    weights = {n: p.detach() for n, p in model.named_parameters() if n.endswith("weight")}
    masks = {}
    if scope == "global":
        all_abs = torch.cat([w.abs().flatten() for w in weights.values()])
        k = int(sparsity * all_abs.numel())
        threshold = all_abs.kthvalue(k).values if k > 0 else torch.tensor(-1.0)
        for n, w in weights.items():
            masks[n] = (w.abs() > threshold).float()
    elif scope == "layer":
        for n, w in weights.items():
            k = int(sparsity * w.numel())
            threshold = w.abs().flatten().kthvalue(k).values if k > 0 else torch.tensor(-1.0)
            masks[n] = (w.abs() > threshold).float()
    else:
        raise ValueError("scope must be 'global' or 'layer'")
    return masks


def sparsity_of(model: nn.Module) -> float:
    """Fraction of weights (not biases) that are exactly zero."""
    ws = [p for n, p in model.named_parameters() if n.endswith("weight")]
    zeros = sum(int((w == 0).sum()) for w in ws)
    return zeros / sum(w.numel() for w in ws)


def sparse_storage_bytes(model: nn.Module, bytes_per_value: int = 1, bytes_per_index: int = 2) -> int:
    """Bytes needed to store the weights in a simple sparse format (CSR):
    each non-zero weight costs a value plus a column index, each row a pointer."""
    total = 0
    for n, p in model.named_parameters():
        if n.endswith("weight"):
            nnz = int((p != 0).sum())
            total += nnz * (bytes_per_value + bytes_per_index) + (p.shape[0] + 1) * bytes_per_index
        else:
            total += p.numel() * 4   # biases stay dense, 32-bit
    return total


# ---------------------------------------------------------------------------
# Node (neuron) pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def neuron_importance(model: MLP, X: torch.Tensor) -> list[torch.Tensor]:
    """Importance score of every hidden neuron, one tensor per hidden layer.

    importance = (average |output| of the neuron on real data)
               x (size of the weights that carry its output to the next layer)

    A neuron that is almost always 0, or whose output is multiplied by tiny
    weights, hardly influences the prediction -- it is a good pruning candidate.
    """
    model.eval()
    scores, x = [], X
    for i, layer in enumerate(model.layers[:-1]):
        x = torch.relu(layer(x))
        mean_act = x.abs().mean(dim=0)                          # (n_neurons,)
        out_w = model.layers[i + 1].weight.norm(dim=0)          # (n_neurons,)
        scores.append(mean_act * out_w)
    return scores


@torch.no_grad()
def remove_neurons(model: MLP, keep: list[torch.Tensor]) -> MLP:
    """Network surgery: build a smaller MLP keeping only the listed neurons.

    ``keep[l]`` holds the indices of the neurons to keep in hidden layer ``l``.
    For each removed neuron we delete its *row* in the layer before it (its
    incoming weights + bias) and its *column* in the layer after it (its
    outgoing weights).
    """
    keep = [torch.as_tensor(k).sort().values for k in keep]
    small = MLP(hidden=[len(k) for k in keep], in_features=model.sizes[0], n_classes=model.sizes[-1])
    prev_keep = torch.arange(model.sizes[0])                    # all inputs are kept
    for l, (old, new) in enumerate(zip(model.layers, small.layers)):
        rows = keep[l] if l < len(keep) else torch.arange(old.out_features)
        new.weight.copy_(old.weight[rows][:, prev_keep])
        new.bias.copy_(old.bias[rows])
        prev_keep = rows
    return small


def prune_neurons(model: MLP, X: torch.Tensor, new_hidden) -> MLP:
    """Keep the ``new_hidden[l]`` most important neurons of every hidden layer."""
    scores = neuron_importance(model, X)
    keep = [s.argsort(descending=True)[:n] for s, n in zip(scores, new_hidden)]
    return remove_neurons(model, keep)

