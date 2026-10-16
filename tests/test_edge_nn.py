"""The C++ inference code must compute exactly what the Python code computes.

Builds a small quantized model, writes a model_data.h into a temporary
folder, compiles arduino/gesture_classifier/edge_nn.h for this computer with
g++, runs it on the whole test set and compares the logits with
``aiedge.quantization.int8_forward`` (bit-identical) and with the PyTorch
float model (~1e-4).

Run with:  uv run pytest
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiedge.data import load_gesture_data  # noqa: E402
from aiedge.export import check_with_host_compiler, device_features, write_model_header  # noqa: E402
from aiedge.models import MLP  # noqa: E402
from aiedge.quantization import QuantMLP  # noqa: E402
from aiedge.training import set_seed, train  # noqa: E402


def test_device_normalization_matches_training():
    data = load_gesture_data(n_per_class=50)
    feats = np.stack([device_features(w, data.channel_mean, data.channel_std) for w in data.raw_test])
    np.testing.assert_array_equal(feats, data.X_test.numpy())


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_cpp_matches_python(tmp_path):
    data = load_gesture_data(n_per_class=200)
    set_seed(0)
    mlp = MLP((24, 12))
    train(mlp, data, epochs=15, verbose=False)
    qmodel = QuantMLP(mlp)
    qmodel.calibrate(data.X_train)
    header = write_model_header(qmodel, data, tmp_path / "model_data.h")
    assert check_with_host_compiler(qmodel, data, header)
