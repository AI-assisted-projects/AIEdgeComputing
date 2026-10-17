"""Run a notebook headless in a fresh kernel and fail on any error.

Usage:
    uv run python tools/run_notebook.py notebooks/02_quantization.ipynb [more.ipynb ...]
    uv run python tools/run_notebook.py --print-output notebooks/01_data_and_first_model.ipynb

The notebook file itself is never modified; the executed copy lives in memory.
Use --save-dir DIR to also write the executed notebooks (with outputs) to DIR.

Board-only cells: the notebooks already guard them with flags that default to
off (RECORD = False in notebook 01, WATCH = False in notebook 05). The runner
leaves those flags alone, so no serial port is ever opened. If a notebook ever
tries to open a port, the run fails like any other error.
"""
import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run(path: Path, timeout: int, print_output: bool, save_dir: Path | None) -> float:
    nb = nbformat.read(path, as_version=4)
    # cwd = the notebook's folder, as when a student opens it in Jupyter
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent.resolve())}})
    start = time.time()
    client.execute()  # raises CellExecutionError on the first failing cell
    elapsed = time.time() - start
    if print_output:
        for cell in nb.cells:
            for out in cell.get("outputs", []):
                if out.output_type == "stream":
                    print(out.text, end="")
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, save_dir / path.name)
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds allowed per cell")
    ap.add_argument("--print-output", action="store_true", help="print the text output of all cells")
    ap.add_argument("--save-dir", type=Path, help="write executed copies here")
    args = ap.parse_args()

    failed = False
    for path in args.notebooks:
        try:
            secs = run(path, args.timeout, args.print_output, args.save_dir)
            print(f"OK    {path}  ({secs:.0f} s)")
        except Exception as exc:  # noqa: BLE001 - report and continue with the next notebook
            failed = True
            print(f"FAIL  {path}: {type(exc).__name__}: {str(exc)[-1500:]}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
