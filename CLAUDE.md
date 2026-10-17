# Conventions
- Run everything through `uv run` (`uv sync` first). Tests: `uv run --group dev pytest`.
- Notebooks are the product. Students have little or no AI/ML background. Keep code plain and add explanatory text cells.
- S7 rule: inline first, no duplication across notebooks, external packages only as a last resort with clear inline comments.
- Edit notebooks programmatically (nbformat) and keep them valid. Do not hand-edit JSON.
- Do not change Arduino sketch behaviour. C++ and Python integer inference must stay bit-identical.
- Never push to `main`. One branch and PR per milestone: `feat/m<n>-<slug>`.
