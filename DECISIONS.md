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

## D6 — Simplify notebook 02 (2026-10-27)
- Context: the owner found notebook 02 too complex for students and allowed removing content.
- Decision / default taken: sections 1–6 keep their code unchanged (same numbers; `QuantMLP`, `int8_forward`, QAT and the five exercises are used by 03, 05 and the solutions), but their texts are shorter: section 1 without the bit layout and energy table, a short `QuantMLP` walk-through (QAT-only parts explained in section 6), section 5 rewritten as four steps (products, sum, back to int8 with $M$, ReLU), and the export/inference code cell split in two. The memory table at the end of section 6 became one sentence. Section 7 is now a text-only overview (one table with a short explanation per row, and three bullets); the code for dynamic range, float16, 16x8 (`QuantMLP16x8`) and per-layer sensitivity is removed. This replaces the notebook-02 part of D4 and the 16x8 part of D5: the slide topics are still named, but no longer simulated.
- Asked owner: yes ("The notebook is way too complex, try to make it simpler and easier to understand. Removing content is allowed.")
- Status: decided

## D5 — M3 review: say what our int8 engine does, and simulate 16x8 by its own rules (2026-10-20)
- Context: the Copilot review of PR #4 found three problems in the new section 7 of notebook 02. It called our engine integer-only, but it rescales each layer with a float multiply. Its overflow example was wrong (300 int16 × int8 products fit in int32). And `act_bits=16` is not LiteRT's 16x8, which uses symmetric int16 activations and int64 biases.
- Decision / default taken: "integer-only" now only means fully integer kernels (toolkits, hardware without an FPU). Our engine is called "int8 inference" everywhere: notebooks 02 and 05, README, and the `aiedge` docstrings. This breaks the M3 rule "existing cells unchanged" for wording only (cells 14, 20, 22–24 of 02 and two labels in 05), because the owner prefers clearer notebooks over untouched cells. No code or numbers changed; `int8_forward` and `edge_nn.h` behave as before, so Python and C++ stay bit-identical. Section 7 refers back to the float $M_j$ of section 5. 7e adds integer rescaling ($M \approx m \cdot 2^{-n}$) as a third need for integer-only hardware. The overflow text now explains in terms of about 512 worst-case products and the larger bias. 16x8 runs through a small `QuantMLP16x8` subclass (symmetric scale, zero-point 0), and a printed check shows that no bias comes near the int32 limit, so int64 biases would give the same result here.
- Asked owner: yes (owner asked to fix the review comments, and agreed to change older cells when that is clearer for students, 2026-10-20)
- Status: decided

## D4 — M3: cover the slide decks' quantization and pruning subtopics (2026-10-20)
- Context: the course started as slide decks (`slides/`). A comparison showed that notebooks 02 and 03 miss some quantization variants and pruning details from decks 3 and 4. PLAN.md had new techniques out of scope.
- Decision / default taken: the owner asked to cover all of it, not necessarily with code. Added as new cells after the existing ones, so existing results stay the same. TensorFlow-specific APIs are named in text only (no new dependency). Things the board cannot run (float16, 16x8, dynamic range, partial QAT) are simulated in PyTorch or explained. Sparse timing uses PyTorch CSR on the laptop and only claims the pattern, not the numbers. No new exercises.
- Asked owner: yes ("Try to cover all of it. Not everything necessarily needs examples in code.")
- Status: decided

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
