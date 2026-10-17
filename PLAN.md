# Plan: AIEdgeComputing

## Goal
A hands-on edge-AI course: six Jupyter notebooks (`notebooks/00`–`05`) in which students train a
gesture-recognition network on IMU data, compress it with quantization, pruning and knowledge
distillation, and deploy it on an Arduino Nano 33 BLE Sense. Supporting code: the `aiedge/` helper
package, two Arduino sketches (`arduino/`), worked solutions for instructors
(`solutions/exercise_solutions.ipynb`) and host-side tests (`tests/`). See README.md.

There is no new roadmap. Work arrives as accepted owner suggestions from Telegram; each one becomes a
milestone here (see Milestones).

## Constraints
- Python 3.10+, dependencies in `pyproject.toml`/`uv.lock`, managed with uv (CPU-only PyTorch).
- Python environments: `.venv` belongs to the owner on the host. Never create, rebuild or modify
  `.venv`. In the container use `.venv-container/` (gitignored):
  `UV_PROJECT_ENVIRONMENT=.venv-container uv sync --group dev`.
- Audience: students with a very basic AI/ML background. Keep explanations short and concrete, and
  keep the important code written out rather than hidden in a framework.
- Each notebook keeps running on its own on a laptop CPU in well under two minutes, on the
  synthetic data by default (no board needed for notebooks 00–04).
- Notebooks and solutions stay in sync: an exercise added, changed or removed in `notebooks/` is
  reflected in `solutions/exercise_solutions.ipynb`. Student notebooks never contain solutions.
- Constants shared with the Arduino code live in `aiedge/config.py`; keep them consistent with the
  sketches. The C++ inference (`edge_nn.h`) must stay bit-exact with the Python integer reference.
- Verification:
  - `uv run --group dev pytest` (in `.venv-container`) passes;
  - every changed notebook runs top to bottom without errors, e.g.
    `jupyter nbconvert --to notebook --execute <nb> --output /tmp/<nb>` (notebook 05 up to the
    board steps; the upload and on-board measurements can't run in the container);
  - Arduino sketches can't be compiled or uploaded in the container (no `arduino-cli`, no board):
    say so in the PR and list what the owner should check.
- Commit student notebooks with their outputs as they were, and the solutions notebook with
  regenerated outputs (it ships with outputs); don't commit large stray outputs.
- Never run destructive file operations (`git clean`, `git reset --hard`, `git checkout .`,
  `rm -rf`, mass deletes) in this repository, and never delete data, recordings, model weights,
  outputs or build artifacts to "clean up". Only remove what a task strictly needs; when in doubt,
  ask.
- Never print, log or commit secrets.

## Out of scope
- Anything that needs the physical board (uploading, serial capture, on-device measurements).
- GPU/CUDA support.

## Ask the owner about
- Changing the course structure (adding, removing or reordering notebooks or exercises).
- Changing the course challenge (≤ 10 KB, ≤ 1 point below the baseline) or the target board.
- New dependencies.

## Allowed external access
- Downloading Python packages.
- Nothing else.

## Milestones
None planned. Accepted suggestions add milestones here, marked `(💡 accepted suggestion S<n>)`.

### M0 — Minimal plan
- Deliverable: this plan, so suggestions can be added to the project.
- Verification: review of the plan files.

## Delegation notes
Qwen may draft explanation text, extra exercise ideas and test data. Not the exercise solutions,
the quantization/pruning/distillation core code or `edge_nn.h`.
