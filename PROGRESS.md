# Progress: AIEdgeComputing

Status: active
<!-- The runner reads this line: planning | active | blocked | done -->
Blocked reason: -
Current milestone: M2

## Tasks

### M1 — Baseline and safety net
- [x] Write `tools/run_notebook.py` (headless fresh-kernel run, fails on error, guards board-only cells). Check that `jupyter`/`nbclient` is available in the dev environment.
- [x] Run every notebook and the solutions notebook through it. Fix only environment problems, not content. Record key numbers and runtimes in `tools/baseline.md`.
- [x] Run `uv run --group dev pytest` and note the result in the baseline. Open the M1 PR.

### M2 — Self-contained beginner notebooks (💡 accepted suggestion S7)
- [x] Audit: list every aiedge function each notebook and the solutions import. Mark each as inline, share or last-resort external. For code used in several notebooks, decide the single place it lives (inline in the first notebook that needs it, or one shared helper). Write the audit as a short table in DECISIONS.md, including the owner rule: inline first, no duplication, externals only as a last resort with clear inline comments.
- [x] Data-and-first-model notebook: inline training, accuracy, seed and model definitions as short plain PyTorch cells. Add text cells explaining the loop and any quirks. If the data generator stays external, add an inline comment block above the import saying what it does, why it is external and how to read it. Run it end to end.
- [x] Quantization notebook: inline scale, zero-point and int8 forward code in simple steps with a short explanation before each. Reuse code from earlier notebooks via the shared place decided in the audit, not copies. Run it and compare results with the M1 baseline.
- [x] Pruning notebook: inline magnitude and neuron pruning code with plain-language text cells. Run it and compare results.
- [x] Distillation notebook: inline the distillation loss and training. Add a text cell explaining temperature. Run it and compare results.
- [x] Deploy notebook (05): inline `cost_table`, the confusion plot and constants; comment blocks for the shared imports and the external last-resort helpers (`write_model_header`, `check_with_host_compiler`, `show_predictions`). Check that `tests/test_edge_nn.py` still passes.
- [x] Solutions notebook (`solutions/exercise_solutions.ipynb`, 66 cells): replace `plots.*` and `config as C` with inline code, comment the remaining shared imports, run it and compare with the baseline.
- [x] Start-here notebook: remove the `sys.path` hack and the unneeded `aiedge` imports if the audit allows. Otherwise comment them clearly.
- [ ] Trim the `aiedge` package to only what the notebooks still import as documented last-resort helpers, plus what the tests need. Remove unused modules. Update README.md and pyproject.toml. Run the tests.
- [ ] Beginner read-through of all notebooks: jargon defined on first use, every cell runs top to bottom in a fresh kernel, every remaining aiedge import has a clear inline comment explaining why it is external, and no code is duplicated across notebooks. Open the M2 PR.

## Session log
<!-- Newest first: YYYY-MM-DD — task — result — how verified — Qwen used? (y/n) -->
2026-10-17 — M2 start-here notebook — sys.path hack and aiedge import removed; the 4 constants it prints are defined inline with a comment — ran via tools/run_notebook.py (2 s), OK — Qwen used? n
2026-10-17 — M2 solutions notebook — plots.* and config as C replaced by inline constants/palette and a plain matplotlib curve plot; remaining aiedge imports have a comment block naming the notebook where each is shown — ran via tools/run_notebook.py (349 s): baseline 97.2%, 4-bit per-tensor 20.0% vs per-channel 96.0%, final table 96.3% at 7,876 B (10.9x), only distillation 97.2% at 86,036 B, all = baseline — Qwen used? n
2026-10-17 — M2 notebook 05 inlined (task split; solutions notebook is next) — cost_table and confusion plot are inline, import cell has a comment block for shared helpers, write_model_header/check_with_host_compiler/show_predictions have their own external-helper comments — ran via tools/run_notebook.py: baseline mean 95.6%, final int8 96.0%, 7.7 KB, 10.9x smaller, challenge MET (= baseline); pytest 1 passed/1 skipped; model_data.h restored — Qwen used? n
2026-10-17 — M2 notebook 04 inlined — ConvNet, count_macs_conv, random_time_shift, a teacher-aware train loop and distillation_loss are now inline with explanations (temperature note added); plots are plain matplotlib; only data/MLP/counters/accuracy/set_seed stay imported (commented). Note: the notebook's `train` is a compact copy of 01's loop plus the teacher option, because the shared one calls aiedge's own loss — ran via tools/run_notebook.py: all numbers identical to baseline (teacher 98.0%, (64,32) 95.8→97.1, 20/class gain +3.1) — Qwen used? n
2026-10-17 — M2 notebook 03 inlined — sparsity_of, count_nonzero_params, sparse_storage_bytes, neuron_importance, remove_neurons, prune_neurons, macs and the plots are now inline with explanations; only data/MLP/train/accuracy/set_seed stay imported (commented). magnitude_masks dropped from the notebook (exercise 2 reworded). aiedge/pruning.py kept for 05/solutions — ran via tools/run_notebook.py (fresh env in /tmp because the project .venv symlink points into /root): all numbers identical to baseline (90% sparsity 73.0% → 97.5%, (24,12) 52.0/96.5/96.2) — Qwen used? n
2026-10-17 — M2 notebook 02 inlined — asymmetric_params, quantize/dequantize/fake_quantize, QuantMLP, export_int8, int8_forward/accuracy/bytes and the plot are now inline with explanations; MLP/train/accuracy/set_seed/data stay imported (shared, commented). aiedge/quantization.py kept for notebook 05/solutions — ran via tools/run_notebook.py: float 97.2%, PTQ int8 97.0%, integer engine 100% match, 21.7 KB vs 84.0 KB (= baseline); QAT 2-bit 97.3% vs PTQ 60.2% (baseline run: 95.8% / 69.7%, both collapse vs recover, seed noise) — Qwen used? n
2026-10-17 — M2 notebook 01 inlined — set_seed, MLP, counters, accuracy, train (+augmentation), confusion matrix, history/example plots, mcu cost now inline with explanations; only the data generator stays external (comment block added) — ran via tools/run_notebook.py: 97.2% test acc, 17 mistakes, 21,509 params, identical to baseline — Qwen used? n
2026-10-17 — M2 audit — table of aiedge usage and single home per function written as D3 — listed imports programmatically from all 7 notebooks — Qwen used? n
2026-10-17 — M1 all tasks — runner + baseline.md written; all 6 notebooks and solutions run clean; pytest 1 passed/1 skipped (no g++) — ran each notebook via tools/run_notebook.py, reran 01/02 for stability, pytest — Qwen used? n
