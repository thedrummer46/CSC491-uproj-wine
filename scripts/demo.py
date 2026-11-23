"""Run one treatment's step demo, the same way on Windows, Mac and Linux.

    python scripts/demo.py 7 1          # T7, step 1
    python scripts/demo.py 5 2 --repeat 1

Puts lib/ (for `import ezr`) and the step's folder (for `import stepK`) on
the import path, points ezr's default data file at data/Wine_quality.csv,
then runs stepN.py as a script, from the repo root.
"""
import argparse, os, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

a = argparse.ArgumentParser()
a.add_argument("t", type=int)
a.add_argument("step", type=int)
a.add_argument("--repeat", type=int, default=1)
x, rest = a.parse_known_args()

folder = ROOT / "treatments" / f"t{x.t}"
if x.t != 7:
    folder = folder / f"r{x.repeat:02d}"
f = folder / f"step{x.step}.py"
if not f.exists():
    sys.exit(f"no such step: {f.relative_to(ROOT)}")

os.chdir(ROOT)
sys.path[:0] = [str(folder), str(ROOT / "lib")]
import ezr  # noqa: E402
# LLM steps fall back on ezr's default file (the lecturer's auto93); point it at ours.
ezr.the.File = ezr.the._defaults.File = "data/Wine_quality.csv"
sys.argv = [str(f)] + rest
runpy.run_path(str(f), run_name="__main__")
