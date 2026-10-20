# AI on the Edge: gesture recognition on an Arduino Nano 33 BLE Sense

A hands-on course in six Jupyter notebooks. Students train a neural network
that recognises hand gestures from motion-sensor data, shrink it with the
three core model-compression techniques, and run it on an **Arduino Nano 33 BLE
Sense** (nRF52840, Arm Cortex-M4F @ 64 MHz, 1 MB flash, 256 KB RAM).

* **Quantization**: float32 → int8, post-training and quantization-aware training
* **Pruning**: edge (weight) pruning and node (neuron) pruning
* **Knowledge distillation**: a convolutional teacher trains a small MLP student

Students only need a very basic AI/ML background: every concept is explained
in the notebooks, and the important code (quantization formulas, pruning
surgery, the distillation loss, the integer-only inference engine) is written
out rather than hidden in a framework.

## The notebooks

| # | Notebook | Content | Board needed |
|---|---|---|---|
| 00 | `start_here` | Edge AI, the board, the project, set-up check | no |
| 01 | `data_and_first_model` | IMU data, neural-network basics, training the baseline, MCU cost; optional: record your own gestures | optional |
| 02 | `quantization` | Number formats, $r = s(q-z)$, per-channel weights, calibration, integer-only inference, QAT and the straight-through estimator; other flavours: dynamic range, float16, 16x8, partial quantization, integer inputs/outputs | no |
| 03 | `pruning` | Magnitude (edge) pruning, masks and fine-tuning, gradual pruning schedules, layer sensitivity and selective pruning, sparse storage, zip compression, measured sparse speed, node pruning and network surgery, comparison with training from scratch, pruning + int8 | no |
| 04 | `distillation` | Teacher/student, temperature, KD loss, when distillation helps (plenty vs little data) | no |
| 05 | `deploy_to_arduino` | Distill → prune → quantize pipeline, C++ export, host-side bit-exact check, upload, measurements on the board | **yes** |

The course goal ("challenge"): a model of **≤ 10 KB** that is at most 1 point
less accurate than the 84 KB float32 baseline. The reference pipeline reaches
about 96% test accuracy at 7.7 KB (≈ 11x smaller).

Each notebook runs on its own (it regenerates the data and re-trains what it
needs), on a laptop CPU in well under two minutes.

## Getting started

```bash
uv sync                  # Python 3.10+, CPU-only PyTorch, Jupyter
uv run jupyter lab       # then open notebooks/00_start_here.ipynb
```

For the board: Arduino IDE 2 with the *Arduino Mbed OS Nano Boards* package
and the IMU library for your board revision: `Arduino_LSM9DS1` (Rev1) or
`Arduino_BMI270_BMM150` (Rev2). Notebook 00 has step-by-step instructions.

## Data

Out of the box, all notebooks use a **physics-based gesture simulator**
(`aiedge/data.py`: hand trajectories, gravity, grip tilt, arm sway, sensor
noise and saturation), so no board is needed for notebooks 00-04. Students can
record their own gestures with `arduino/gesture_capture` (notebook 01, section
8) and train/deploy on those by setting `DATA_SOURCE = "recorded"` in notebook 05.

## Repository layout

```
notebooks/                 the course (00 ... 05)
solutions/                 worked solutions to all exercises (for instructors)
aiedge/                    last-resort helpers; the notebooks show all other code inline
  config.py                constants shared with the Arduino code (rate, window, gestures, trigger)
  data.py                  gesture simulator, loader for recordings, splits + normalization
  models.py                MLP (student/baseline) and ConvNet (teacher), size/MAC counters
  training.py              training loop: plain, distillation, pruning masks, QAT; metrics
  pruning.py               edge pruning (masks) and node pruning (importance + surgery)
  quantization.py          fake quantization, QuantMLP (PTQ/QAT), int8 export, integer inference
  export.py                MCU cost model, model_data.h generator, host-compiler check
  board.py                 serial-port helpers (record gestures, show predictions)
arduino/
  gesture_capture/         sketch: record gesture windows as CSV over USB
  gesture_classifier/      sketch: on-device inference (edge_nn.h + generated model_data.h)
data/recorded/             your own recordings (<gesture>.csv)
tests/                     checks that the C++ inference matches Python bit for bit
```

## For instructors

* `solutions/exercise_solutions.ipynb` contains detailed, worked solutions to
  all 27 exercises, saved with outputs (re-running takes ~6 minutes). It lives
  outside `notebooks/` so it is easy to leave out when handing out the course.
* `arduino/gesture_classifier/model_data.h` ships pre-generated (synthetic
  data), so the classifier sketch can be uploaded before running any notebook.
  Notebook 05 overwrites it.
* `uv run --group dev pytest` compiles `edge_nn.h` with the host `g++` and
  checks it against the Python integer reference (requires `g++`).
* Both sketches compile for Rev1 and Rev2 with `arduino-cli compile -b
  arduino:mbed_nano:nano33ble` (set `BOARD_REVISION` at the top of each sketch).
* Results on synthetic data vary by about ±1 percentage point between seeds;
  the notebooks discuss this explicitly and average over seeds where it matters.
