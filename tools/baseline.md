# Baseline: behaviour before the S7 refactor (M1)

Recorded 2026-10-17 on a laptop-class CPU (Linux, Python 3.12, PyTorch 2.14.1+cpu, NumPy 2.5.3).
Data: synthetic generator, seed 0 (the notebooks' own seeds, unchanged). Notebooks were not modified.

Reproduce: `uv run python tools/run_notebook.py --print-output notebooks/<name>.ipynb`
(`--save-dir DIR` keeps the executed copies). Compare the M2 results against the numbers below
within seed noise (about ±1 point on accuracy).

## Runner and board-only cells
- `tools/run_notebook.py` executes in a fresh `python3` kernel with the notebook folder as cwd and stops on the first error.
- Board-only cells are already guarded inside the notebooks: `RECORD = False` (notebook 01) and `WATCH = False` (notebook 05). The runner leaves them off, so no serial port is opened.
- The C++ host check in notebook 05 prints "g++ not found -- skipping" when there is no compiler. That is not an error. `tests/test_edge_nn.py` skips its C++ test for the same reason.
- Side effect: notebook 05 rewrites `arduino/gesture_classifier/model_data.h` (timestamp line plus regenerated numbers). Run `git checkout arduino/gesture_classifier/model_data.h` afterwards. Notebooks 01 and 04 write `models/*.pt` (not tracked). Notebook 02 and 05 read them, so run the notebooks in order 01 → 04 → 05 on a fresh clone.
- Environment note: the repo's `.venv` may be unusable in a sandbox (the interpreter symlink points into another user's home). Use `UV_PROJECT_ENVIRONMENT=/some/dir uv sync --group dev` instead.

## Results
| Notebook | Run time | Key numbers |
|---|---|---|
| 00 start here | 2 s | environment check passes |
| 01 data and first model | 12 s | 2800/600/600 train/val/test; MLP 300-64-32-5, 21,509 params, 21,408 MACs, 84.0 KB; untrained 19.0%; **baseline test acc 97.2%** (17 mistakes of 600) |
| 02 quantization | 14 s | float32 97.2%; int8 PTQ 97.0%; integer-only engine matches simulation 100%; memory 84.0 KB → 21.7 KB; QAT 2-bit 95.8% vs PTQ 69.7%. (The very first run showed float32 96.2% / PTQ 96.2%; three later reruns gave 97.2% / 97.0% — within seed noise.) |
| 03 pruning | 29 s | 50% sparsity one-shot 97.5%; 90% sparsity one-shot 73.0% → fine-tuned 97.5%; 98% fine-tuned 76.5%; neuron pruning to (24,12): one-shot 52.0%, fine-tuned 96.5%, from scratch 96.2%; (8,4) fine-tuned 94.5% |
| 04 distillation | 94 s | ConvNet teacher 98.0% (32,485 params); student (64,32) 95.8% → 97.1% distilled; (32,16) 96.3% → 96.1%; (8,) 92.9% → 93.7%; with 20 examples/gesture gain +3.1 points, 40: +0.8, 80: +0.2 |
| 05 deploy | 17 s | baseline 3-seed mean 95.6%; distilled 96.8%; pruned 24-12 96.2%; **final int8 96.0%, 7,876 B weights (7.7 KB), 10.9× smaller than float32**; challenge (≤10 KB, ≥ baseline −1) MET; 7,548 MACs, ~0.47 ms estimated |
| solutions | 376 s | baseline 97.2%; 3-seed mean 95.8% ± 1.0%; hidden (8,) 95.0%, (256,128) 96.3%; 4-bit per-tensor with outlier 20.0% vs per-channel 96.0%; with masks 90% sparsity 97.5%; T=4 best distillation at 20/class (88.4%); final table: all three techniques 96.3% at 7,876 B (10.9×); only distillation 97.2% at 86,036 B |

All six notebooks and the solutions notebook run without errors.

## Tests
`uv run --group dev pytest` → 1 passed, 1 skipped (the C++ bit-exactness test skips because `g++` is not installed in this sandbox, so it is not verified here).
