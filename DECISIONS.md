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
