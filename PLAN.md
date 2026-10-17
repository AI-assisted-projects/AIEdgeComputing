# Plan: AIEdgeComputing

## Goal
Keep the existing hands-on edge-AI course (gesture recognition on an Arduino Nano 33 BLE
Sense: quantization, pruning, distillation, deployment) working as it is, and make its
notebooks self-contained and readable for students with little or no AI/ML background
(💡 accepted suggestion S7). Nothing else is in scope.

"Done" means:
- Each notebook's important code is inline, short and plain PyTorch/NumPy, with text cells that
  explain loops, quirks and jargon.
- Code is never duplicated across notebooks.
- The few remaining `aiedge` imports are last-resort helpers, each with a clear inline comment.
- Results match the pre-refactor notebooks, and the tests pass.

## Constraints
- Python >=3.10, `uv`, PyTorch (CPU), NumPy, Matplotlib, Jupyter. No new dependencies.
- Owner rule (S7): inline first, no duplication, external packages only as a last resort and
  documented very clearly with inline comments.
- Keep the course goal and numbers valid: <=10 KB model, within about 1 point of the float32
  baseline. Results on synthetic data vary by about ±1 point between seeds.
- Each notebook must still run on its own, top to bottom in a fresh kernel, on a laptop CPU.
- The Arduino sketches and `edge_nn.h` keep their behaviour. The C++ output must still match
  the Python integer reference bit for bit.
- Do not hand-edit notebook JSON. Edit with nbformat or a script and keep the notebooks valid.

## Out of scope
- New notebooks, exercises, techniques, boards or datasets.
- Changes to the Arduino sketches' behaviour.
- CI setup and other new tooling.
- Rewriting the course content beyond what S7 needs.

## Ask the owner about
- Removing or renaming a notebook, or changing the course goal or its numbers.
- Anything that needs hardware, paid services or network access.
Everything else: use judgment and log it in DECISIONS.md.

## Allowed external access
None, except installing the dependencies already in `pyproject.toml` with `uv sync`.
The board is not available to autonomous sessions. Board-only cells are verified by reading
them and by the host-side check.

## Milestones

### M1 — Baseline and safety net
- Deliverable: a reproducible record of today's behaviour so the refactor can be compared against it:
  - `tools/run_notebook.py` executes any notebook headless in a fresh kernel (nbclient or
    `jupyter nbconvert`) and fails on any error.
  - `tools/baseline.md` lists, per notebook, the key printed numbers (accuracies, sizes, sparsity)
    and the runtime.
  - The notebooks and solutions are not changed.
- Acceptance criteria:
  - Every notebook runs without errors. Board-only cells are skipped or guarded, and the runner documents how.
  - The existing tests pass.
  - The baseline file records numbers from a fixed seed.
- Verification: `uv run python tools/run_notebook.py notebooks/<each notebook>`, then `uv run --group dev pytest`.

### M2 — Self-contained beginner notebooks (💡 accepted suggestion S7)
- Deliverable: the S7 refactor (see the tasks in PROGRESS.md).
- Acceptance criteria:
  - Every `aiedge` function a notebook used is inline or documented as a last-resort external.
  - Shared code lives in exactly one place. No code is duplicated across notebooks.
  - Every text cell is beginner-readable and jargon is defined on first use.
  - Board serial code and the C header export stay external, each with an inline comment block
    (purpose, inputs, outputs, why not inline).
  - Results match the M1 baseline within seed noise (about ±1 point).
  - `aiedge` is trimmed to the documented helpers plus what the tests need.
  - `README.md` and `pyproject.toml` match the new layout.
- Verification:
  - Each notebook runs top to bottom through the M1 runner in a fresh kernel.
  - `grep` for `aiedge` imports in the notebooks shows only the documented helpers.
  - A duplicate-code check by eye and by grep over the notebook code cells.
  - `uv run --group dev pytest` passes.

## Delegation notes
- Qwen may draft the explanatory text cells and the baseline summary table. Always review the output.
- Qwen must not decide what is inlined or shared, and must not edit the notebooks directly.
  The audit and the numerical comparisons are done by Claude.
