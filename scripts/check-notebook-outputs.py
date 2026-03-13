#!/usr/bin/env python3
"""Check that Jupyter notebooks have no cell outputs or execution counts.

Usage:
    python scripts/check-notebook-outputs.py [files...]

If no files are given, checks all .ipynb files under docs/.
Exit code 1 if any notebook has outputs.
"""

import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> list[str]:
    """Return list of problems found in the notebook."""
    with open(path) as f:
        nb = json.load(f)

    problems = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("outputs"):
            problems.append(f"  cell {i}: has outputs")
        if cell.get("execution_count") is not None:
            problems.append(f"  cell {i}: has execution_count")
    return problems


def main():
    files = sys.argv[1:]
    if not files:
        files = [str(p) for p in Path("docs").rglob("*.ipynb")]

    failed = False
    for f in files:
        path = Path(f)
        if not path.exists() or not path.suffix == ".ipynb":
            continue
        problems = check_notebook(path)
        if problems:
            print(f"{path}:")
            for p in problems:
                print(p)
            failed = True

    if failed:
        print("\nNotebooks must not contain outputs. Strip them with:")
        print('  jupyter nbconvert --clear-output --inplace <notebook>')
        sys.exit(1)


if __name__ == "__main__":
    main()
