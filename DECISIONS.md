# Decisions: AIEdgeComputing

<!--
One entry per decision, newest first: add new entries at the top, numbered one above the
highest D<n> on main.

## D<n> — <short title> (YYYY-MM-DD)
- Context:
- Decision / default taken:
- Asked owner: no | yes (question id, answer)
- Status: decided | provisional (awaiting answer) | superseded by D<m>
-->

## D3 — M2 audit: where each aiedge function lives (2026-10-17)
- Context: M2 (S7) rule from the owner: inline first, no duplication across notebooks, externals only as a last resort with clear inline comments. Imports of `aiedge` per notebook were listed from the notebooks and the solutions notebook (`tools`-free audit, `from aiedge… import`).
- Decision / default taken (single home per function):

| Function(s) | Used in | Home | Class |
|---|---|---|---|
| `config` (`C`: sizes, class names, `CYCLES_PER_MAC`) | 00, 01, 04, sol | constants inline in the first notebook that needs them (01); later notebooks re-define only the 1–2 they use, as plain variables | inline |
| `set_seed`, `accuracy`, `predict_logits`, `train`, `confusion_matrix` | 01–05, sol | inline in 01 (training loop explained); 02–05 and sol import them from the shared helper (see Shared-helper rule) | inline (01) + share |
| `MLP`, `ConvNet`, `count_params`, `count_macs`, `count_macs_conv`, `count_nonzero_params` | 01–05, sol | `MLP`, `count_*` inline in 01; `ConvNet` inline in 04 (only teacher there; 05/sol reuse via shared helper) | inline / share |
| `load_gesture_data`, `limit_training_data`, `GestureData` (generator + splits) | 01–05, sol | external last resort (long synthetic generator, ~300 lines, physics-like motion code that is not the learning goal). Keep as `aiedge.data` with an inline comment block above the import | last-resort external |
| `magnitude_masks`, `sparsity_of`, `sparse_storage_bytes`, `neuron_importance`, `remove_neurons`, `prune_neurons` | 03, 05, sol | inline in 03; 05 and sol reuse `prune_neurons` via the shared helper | inline (03) + share |
| `QuantMLP`, `asymmetric_params`, `int8_forward`, `int8_accuracy`, `int8_model_bytes` | 02, 05, sol | inline in 02 (steps explained); 05 and sol reuse via the shared helper | inline (02) + share |
| `distillation_loss` | 04 | inline in 04 | inline |
| `mcu_cost`, `cost_table`, `CYCLES_PER_MAC` | 01, 04, 05 | inline in 01 (`mcu_cost`); `cost_table` inline in 05 | inline |
| `plots.*` | 01–05, sol | short matplotlib calls inline where used; keep no plot helpers | inline |
| `write_model_header`, `check_with_host_compiler` | 05 | external last resort (C header export, host compiler check) | last-resort external |
| `board.record_gestures`, `list_ports`, `show_predictions` | 01, 05 | external last resort (serial port code) | last-resort external |

- Shared-helper rule: notebooks cannot import each other without duplication or a package, so code needed in 2+ notebooks (`ConvNet`, `prune_neurons`, `QuantMLP`/int8 functions, `train`) stays in ONE small module `aiedge/` (trimmed in the last M2 task), imported with an inline comment saying what it does and which notebook shows the code in full. The first notebook that needs it shows the full code inline; later notebooks import it. This is the only place "duplication" is avoided by import rather than inline. Revisit per task if a notebook can simply load `models/*.pt`.
- Asked owner: no
- Status: decided

## D2 — M1 is a baseline and safety net before the refactor (2026-10-17)
- Context: S7 changes every notebook, and the repo has no way to run notebooks headless or to compare results.
- Decision / default taken: M1 adds a notebook runner and a recorded baseline without touching content. S7 becomes M2.
- Asked owner: no
- Status: decided

## D1 — Scope is the existing content plus S7 only (2026-10-17)
- Context: the repo is a finished course. The owner was asked about other work.
- Decision / default taken: the plan contains only the existing content and S7. The plan does not state a notebook count.
- Asked owner: yes (planning interview, owner answer: "Only include the current content and S7")
- Status: decided
