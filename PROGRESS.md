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
- [ ] Pruning notebook: inline magnitude and neuron pruning code with plain-language text cells. Run it and compare results.
- [ ] Distillation notebook: inline the distillation loss and training. Add a text cell explaining temperature. Run it and compare results.
- [ ] Deploy notebook and the solutions notebook: inline what is feasible. Keep board serial code and the C header export as external last-resort helpers, each with a clear inline comment block (purpose, inputs, outputs, why not inline). Check that `tests/test_edge_nn.py` still passes.
- [ ] Start-here notebook: remove the `sys.path` hack and the unneeded `aiedge` imports if the audit allows. Otherwise comment them clearly.
- [ ] Trim the `aiedge` package to only what the notebooks still import as documented last-resort helpers, plus what the tests need. Remove unused modules. Update README.md and pyproject.toml. Run the tests.
- [ ] Beginner read-through of all notebooks: jargon defined on first use, every cell runs top to bottom in a fresh kernel, every remaining aiedge import has a clear inline comment explaining why it is external, and no code is duplicated across notebooks. Open the M2 PR.

## Session log
<!-- Newest first: YYYY-MM-DD — task — result — how verified — Qwen used? (y/n) -->
2026-10-17 — M2 notebook 02 inlined — asymmetric_params, quantize/dequantize/fake_quantize, QuantMLP, export_int8, int8_forward/accuracy/bytes and the plot are now inline with explanations; MLP/train/accuracy/set_seed/data stay imported (shared, commented). aiedge/quantization.py kept for notebook 05/solutions — ran via tools/run_notebook.py: float 97.2%, PTQ int8 97.0%, integer engine 100% match, 21.7 KB vs 84.0 KB (= baseline); QAT 2-bit 97.3% vs PTQ 60.2% (baseline run: 95.8% / 69.7%, both collapse vs recover, seed noise) — Qwen used? n
2026-10-17 — M2 notebook 01 inlined — set_seed, MLP, counters, accuracy, train (+augmentation), confusion matrix, history/example plots, mcu cost now inline with explanations; only the data generator stays external (comment block added) — ran via tools/run_notebook.py: 97.2% test acc, 17 mistakes, 21,509 params, identical to baseline — Qwen used? n
2026-10-17 — M2 audit — table of aiedge usage and single home per function written as D3 — listed imports programmatically from all 7 notebooks — Qwen used? n
2026-10-17 — M1 all tasks — runner + baseline.md written; all 6 notebooks and solutions run clean; pytest 1 passed/1 skipped (no g++) — ran each notebook via tools/run_notebook.py, reran 01/02 for stability, pytest — Qwen used? n
