"""Quantization: storing and computing with 8-bit integers instead of 32-bit floats.

The scheme (the same one TensorFlow Lite Micro and CMSIS-NN use)
-------------------------------------------------------------------
A real number ``r`` is represented by an integer ``q`` via

        r  ~=  scale * (q - zero_point)

* **Weights**: symmetric (zero_point = 0), one scale per output neuron
  ("per-channel"), integers in [-127, 127].
* **Activations** (the numbers flowing between layers): asymmetric, one
  (scale, zero_point) pair per layer, integers in [-128, 127].
* **Biases**: int32 with scale = input_scale * weight_scale, so they can be
  added straight into the integer accumulator.

int8 inference of one layer (what the Arduino executes): integer MACs, one float
rescale per neuron:

    acc[j]  = bias_q[j] + sum_i  W_q[j, i] * (x_q[i] - x_zero_point)       (int32)
    y_q[j]  = clamp( round(acc[j] * M[j]) + y_zero_point, y_zero_point, 127 )
              with  M[j] = x_scale * w_scale[j] / y_scale

The lower clamp at ``y_zero_point`` *is* the ReLU (it is where real 0 lies).
For the last layer we skip re-quantization and output float logits
``acc[j] * x_scale * w_scale[j]``.

``int8_forward`` below is written to give *bit-identical* results to the C++
code in ``arduino/gesture_classifier/edge_nn.h``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import MLP


# ---------------------------------------------------------------------------
# The basic building blocks
# ---------------------------------------------------------------------------

def symmetric_scale(max_abs: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Scale that maps [-max_abs, max_abs] onto [-(2^(b-1)-1), 2^(b-1)-1]."""
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(max_abs, min=1e-8) / qmax


def asymmetric_params(lo: float, hi: float, bits: int = 8) -> tuple[float, int]:
    """(scale, zero_point) mapping [lo, hi] onto [-2^(b-1), 2^(b-1)-1].

    The range is first widened to include 0, so that real zero is exactly
    representable (important: ReLU produces lots of exact zeros).
    """
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = max(hi - lo, 1e-8) / (qmax - qmin)
    zero_point = int(round(qmin - lo / scale))
    return scale, int(np.clip(zero_point, qmin, qmax))


def quantize(x: torch.Tensor, scale, zero_point, qmin: int, qmax: int) -> torch.Tensor:
    """real -> integer"""
    return torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)


def dequantize(q: torch.Tensor, scale, zero_point) -> torch.Tensor:
    """integer -> real (approximately the original number)"""
    return (q - zero_point) * scale


def fake_quantize(x: torch.Tensor, scale, zero_point, qmin: int, qmax: int) -> torch.Tensor:
    """Quantize and immediately dequantize, so the result is a float that can
    only take the values an integer model could represent.

    Rounding has zero gradient almost everywhere, which would stop training.
    The *straight-through estimator* (STE) pretends rounding is the identity
    during the backward pass: ``x + (fq(x) - x).detach()`` has the value of
    ``fq(x)`` but the gradient of ``x``. Values outside the representable range
    are clipped and get zero gradient.
    """
    lo, hi = (qmin - zero_point) * scale, (qmax - zero_point) * scale
    x_c = torch.maximum(torch.minimum(x, torch.as_tensor(hi)), torch.as_tensor(lo))
    fq = dequantize(quantize(x_c, scale, zero_point, qmin, qmax), scale, zero_point)
    return x_c + (fq - x_c).detach()


# ---------------------------------------------------------------------------
# A quantization-aware wrapper around an MLP
# ---------------------------------------------------------------------------

class QuantMLP(nn.Module):
    """Wraps a trained ``MLP`` and simulates integer inference with fake quantization.

    * ``calibrate(X)`` measures the range of every activation on real data
      (this is all *post-training quantization* needs).
    * Training this module (``training.train(qmodel, ...)``) is *quantization-aware
      training* (QAT): the forward pass sees the rounding errors, the backward
      pass (via the STE) lets the float weights adapt to them.
    * ``to_int8()`` exports the integer model for ``int8_forward`` and the Arduino.

    ``weight_bits`` / ``act_bits`` can be lowered (e.g. 4) to explore the
    accuracy-vs-bits trade-off; only 8 bits is exported to the board.
    """

    def __init__(self, mlp: MLP, weight_bits: int = 8, act_bits: int = 8,
                 per_channel: bool = True, ema_momentum: float = 0.02):
        super().__init__()
        self.mlp = mlp
        self.weight_bits, self.act_bits = weight_bits, act_bits
        self.per_channel = per_channel
        self.ema_momentum = ema_momentum
        n_points = len(mlp.layers)          # the input + every hidden layer output
        self.register_buffer("act_lo", torch.zeros(n_points))
        self.register_buffer("act_hi", torch.zeros(n_points))
        self.calibrated = False
        self.enabled = True                 # False -> behaves exactly like the float MLP

    # -- ranges -------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, X: torch.Tensor) -> None:
        """Record min/max of the input and of every hidden activation on ``X``."""
        self.mlp.eval()
        x = X
        self.act_lo[0], self.act_hi[0] = x.min(), x.max()
        for i, layer in enumerate(self.mlp.layers[:-1]):
            x = torch.relu(layer(x))
            self.act_lo[i + 1], self.act_hi[i + 1] = 0.0, x.max()
        self.calibrated = True

    def act_params(self, k: int) -> tuple[float, int]:
        return asymmetric_params(self.act_lo[k].item(), self.act_hi[k].item(), self.act_bits)

    def weight_scale(self, layer: nn.Linear) -> torch.Tensor:
        w = layer.weight
        max_abs = w.abs().amax(dim=1) if self.per_channel else w.abs().max().expand(w.shape[0])
        return symmetric_scale(max_abs.detach(), self.weight_bits)

    # -- forward ------------------------------------------------------------
    def _act(self, k: int, x: torch.Tensor) -> torch.Tensor:
        if self.training:                   # QAT: let the ranges follow the data slowly
            with torch.no_grad():
                lo = x.min() if k == 0 else torch.zeros(())
                m = self.ema_momentum
                self.act_lo[k] = (1 - m) * self.act_lo[k] + m * lo
                self.act_hi[k] = (1 - m) * self.act_hi[k] + m * x.max()
        scale, zp = self.act_params(k)
        qmin, qmax = -(2 ** (self.act_bits - 1)), 2 ** (self.act_bits - 1) - 1
        return fake_quantize(x, scale, zp, qmin, qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.mlp(x)
        if not self.calibrated:
            raise RuntimeError("call calibrate(X_train) first")
        x = self._act(0, x)
        n = len(self.mlp.layers)
        qmax_w = 2 ** (self.weight_bits - 1) - 1
        for i, layer in enumerate(self.mlp.layers):
            s_w = self.weight_scale(layer)
            w = fake_quantize(layer.weight, s_w[:, None], 0, -qmax_w, qmax_w)
            s_x, _ = self.act_params(i)
            s_b = s_x * s_w                          # int32 bias, scale = s_x * s_w
            b = fake_quantize(layer.bias, s_b, 0, -(2 ** 31), 2 ** 31 - 1)
            x = F.linear(x, w, b)
            if i < n - 1:
                x = self._act(i + 1, torch.relu(x))
        return x

    # -- export ---------------------------------------------------------------
    @torch.no_grad()
    def to_int8(self) -> dict:
        """Export the integer model: int8 weights, int32 biases, float32 multipliers."""
        assert self.weight_bits <= 8 and self.act_bits == 8, "the board runs 8-bit activations"
        layers = []
        n = len(self.mlp.layers)
        for i, layer in enumerate(self.mlp.layers):
            s_x, z_x = self.act_params(i)
            s_w = self.weight_scale(layer).double()
            qmax_w = 2 ** (self.weight_bits - 1) - 1
            w_q = quantize(layer.weight.double(), s_w[:, None], 0, -qmax_w, qmax_w)
            b_q = torch.round(layer.bias.double() / (s_x * s_w))
            entry = {
                "w_q": w_q.numpy().astype(np.int8),
                "b_q": b_q.numpy().astype(np.int32),
                "in_scale": np.float32(s_x),
                "in_zero_point": int(z_x),
                "w_scale": s_w.numpy().astype(np.float32),
            }
            if i < n - 1:
                s_y, z_y = self.act_params(i + 1)
                entry["out_zero_point"] = int(z_y)
                entry["out_scale"] = np.float32(s_y)
                entry["multiplier"] = (s_x * s_w.numpy() / s_y).astype(np.float32)
            else:
                entry["multiplier"] = (s_x * s_w.numpy()).astype(np.float32)
            layers.append(entry)
        return {"layers": layers,
                "input_scale": np.float32(self.act_params(0)[0]),
                "input_zero_point": int(self.act_params(0)[1])}


# ---------------------------------------------------------------------------
# int8 inference in NumPy (bit-exact twin of the C++ code)
# ---------------------------------------------------------------------------

def _round_half_away(v: np.ndarray) -> np.ndarray:
    """C's ``roundf``: halves are rounded away from zero (2.5 -> 3, -2.5 -> -3)."""
    v = v.astype(np.float64)                  # exact for float32 inputs
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def quantize_input(int8_model: dict, x: np.ndarray) -> np.ndarray:
    """float32 normalized features -> int8, exactly like the C++ code."""
    x = np.asarray(x, dtype=np.float32)
    q = _round_half_away(x / int8_model["input_scale"]) + int8_model["input_zero_point"]
    return np.clip(q, -128, 127).astype(np.int8)


def int8_forward(int8_model: dict, x: np.ndarray) -> np.ndarray:
    """Run the integer model on normalized float input ``(N, 300)`` -> float32 logits ``(N, 5)``."""
    a = quantize_input(int8_model, x).astype(np.int64)
    layers = int8_model["layers"]
    for i, L in enumerate(layers):
        acc = L["b_q"].astype(np.int64) + (a - L["in_zero_point"]) @ L["w_q"].astype(np.int64).T
        assert np.abs(acc).max() < 2 ** 24, "accumulator too large for exact float32 conversion"
        p = acc.astype(np.float32) * L["multiplier"]               # float32 product
        if i < len(layers) - 1:
            q = _round_half_away(p) + L["out_zero_point"]
            a = np.clip(q, L["out_zero_point"], 127).astype(np.int64)   # clamp = ReLU
        else:
            return p.astype(np.float32)
    raise AssertionError("unreachable")


def int8_accuracy(int8_model: dict, X: torch.Tensor, y: torch.Tensor) -> float:
    logits = int8_forward(int8_model, X.numpy())
    return float((logits.argmax(1) == y.numpy()).mean())


def int8_model_bytes(int8_model: dict) -> int:
    """Flash needed by the integer model: int8 weights + int32 biases + float32 multipliers."""
    return sum(L["w_q"].size + 4 * L["b_q"].size + 4 * L["multiplier"].size
               for L in int8_model["layers"])
