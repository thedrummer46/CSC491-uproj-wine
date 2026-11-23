"""New code per step, per treatment: the data for the Results histogram.

    python scripts/newcode.py > results/newcode.csv

"cloc per tag": for every tag stepN, read each treatment's stepN.py exactly
as it was at that tag and count its lines (non-blank, the same count
run_trial.py logs). Averaged over repeats r01..r05 (T7 has one copy).

  loc      lines in stepN.py at tag stepN
  pct_new  100 * loc / lines in step1..stepN.py at that tag: the share of the
           treatment's code that this step had to add (the histogram's y-axis)

A tag only needs to sit on a commit holding every treatment's stepN.py;
later steps being there too is fine, since only step1..stepN are read.
"""
import re, subprocess, sys
from collections import defaultdict
from statistics import mean

FILE = re.compile(r"treatments/(t\d+)/(?:(r\d\d)/)?step(\d+)\.py$")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def loc(ref, path):
    return sum(1 for l in git("show", f"{ref}:{path}").splitlines() if l.strip())


def at(ref, n):
    "{treatment: [(loc, pct_new) per repeat]} for step N at REF."
    files = defaultdict(dict)  # (treatment, repeat) -> {step: path}
    for path in git("ls-tree", "-r", "--name-only", ref, "--", "treatments").splitlines():
        m = FILE.match(path)
        if m and int(m[3]) <= n:
            files[(m[1], m[2] or "-")][int(m[3])] = path
    out = defaultdict(list)
    for (t, _), steps in sorted(files.items()):
        if n in steps:
            new = loc(ref, steps[n])
            total = sum(loc(ref, p) for p in steps.values())
            out[t].append((new, 100 * new / total if total else 0))
    return out


def main(refs):
    "REFS: [(ref, n)], by default every stepN tag."
    print("treatment,step,tag,repeats,loc,pct_new")
    for ref, n in refs:
        for t, vals in sorted(at(ref, n).items(), key=lambda kv: int(kv[0][1:])):
            print(f"{t},{n},{ref},{len(vals)},"
                  f"{mean(v[0] for v in vals):.1f},{mean(v[1] for v in vals):.1f}")


if __name__ == "__main__":
    # testing aid: `--at REF N` measures step N at any commit, tag or not
    if sys.argv[1:2] == ["--at"]:
        main([(sys.argv[2], int(sys.argv[3]))])
    else:
        tags = [t for t in git("tag", "-l", "step*").split() if re.fullmatch(r"step\d+", t)]
        main(sorted(((t, int(t[4:])) for t in tags), key=lambda x: x[1]))
