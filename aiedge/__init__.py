"""aiedge -- helper package for the "AI on the Edge" notebooks.

Modules
-------
config        constants shared by Python and the Arduino code (sample rate, gestures, ...)
data          synthetic gesture generator + loader for your own recordings
models        the MLP network and size/cost counters
training      training loop (plain, distillation, pruning masks, QAT) and metrics
pruning       edge (weight) pruning and node (neuron) pruning
quantization  int8 quantization: fake-quant for QAT and int8 inference
export        cost model for the microcontroller + C header generation
board         talk to the Arduino over USB serial (record gestures, live predictions)
"""

import os as _os

import torch as _torch

# The networks in this course are tiny. Many CPU threads only add overhead
# (and can make training 100x slower on some machines), so cap them.
_torch.set_num_threads(min(4, _os.cpu_count() or 1))
