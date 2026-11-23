"""Score every treatment's stepN.py on the one test suite (data/splits.json).

    python -m harness.score 1        # score step 1 for every treatment found

Per docs/assessment.md: LLM repeat r (treatments/tK/rNN/) is scored on split r;
T7 (treatments/t7/) is written once and scored on all 5 splits. A step that
crashes or breaks the interface scores blank and keeps its error message --
broken stays broken. Writes results/scores/stepN.csv and prints it.
Harness code: used to SCORE treatments, never shown to them.
"""
import csv, importlib, json, random, re, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))  # treatments `import ezr`, so we do too
import ezr  # noqa: E402

DATA = ROOT / "data" / "Wine_quality.csv"
SPLITS = ROOT / "data" / "splits.json"
OUT = ROOT / "results" / "scores"


#-- data ----------------------------------------------------
def wine():
    "[header] + data rows, cells coerced to numbers."
    return list(ezr.csv(str(DATA)))


def splits():
    "[(train_rows, test_rows), ...], each with the header first."
    rows = wine()
    names, data = rows[0], rows[1:]
    return [([names] + [data[i] for i in s["train"]],
             [names] + [data[i] for i in s["test"]])
            for s in json.loads(SPLITS.read_text())["splits"]]


#-- finding and loading treatments ---------------------------
def trials(step):
    "[(treatment, repeat, folder)] for every folder holding stepN.py."
    out = []
    for t in sorted((ROOT / "treatments").glob("t*"), key=lambda p: int(p.name[1:])):
        if (t / f"step{step}.py").exists():          # T7: one copy, no repeats
            out.append((t.name, 0, t))
        for r in sorted(t.glob("r[0-9][0-9]")):
            if (r / f"step{step}.py").exists():
                out.append((t.name, int(r.name[1:]), r))
    return out


def load(folder, step):
    """Import FOLDER/stepN.py fresh. Every treatment has its own step1,
    step2, ..., so forget any previously loaded ones first."""
    for k in [k for k in sys.modules if re.fullmatch(r"step\d+", k)]:
        del sys.modules[k]
    sys.path.insert(0, str(folder))
    try:
        return importlib.import_module(f"step{step}")
    finally:
        sys.path.remove(str(folder))


#-- scorers: one per step, (module, train, test) -> (score, note) ----
def step1(m, train, test):
    """Mean `wins` of every test batch remember() returns (at most 5%).
    Ask 2 (summary) is checked, not scored: note says if it passed."""
    got = m.remember(test)
    cap = int(0.05 * (len(test) - 1))
    if not got or len(got) > cap:
        raise ValueError(f"remember returned {len(got)} rows; want 1..{cap}")
    win = ezr.wins(ezr.Tbl(test))
    score = sum(win(r) for r in got) / len(got)
    s = m.summary(test)
    tbl = ezr.Tbl(test)
    ok = (set(s) == set(test[0]) and
          all(abs(float(s[c.txt]) - c.mu) <= 0.1 * (c.sd or 1)
              for c in tbl.cols.values()))
    return score, "summary ok" if ok else "summary WRONG"


def blind(names, row, goal="Quality+"):
    "ROW with its GOAL cell hidden as '?': the lab sheet alone."
    at = names.index(goal)
    return row[:at] + ["?"] + row[at + 1:]


def step2(m, train, test):
    """Mean absolute error of guess() on test batches (lower is better),
    each passed with its Quality+ hidden. Ask 4 (kinds) is checked, not
    scored: note says if it returns (rows, number) pairs covering train."""
    names, at = test[0], test[0].index("Quality+")
    score = sum(abs(float(m.guess(train, blind(names, r))) - r[at])
                for r in test[1:]) / (len(test) - 1)
    k = m.kinds(train)
    ok = (isinstance(k, list) and 1 < len(k) < len(train) - 1 and
          all(len(g) == 2 and isinstance(g[1], (int, float)) for g in k) and
          sum(len(g[0]) for g in k) == len(train) - 1)
    return score, "kinds ok" if ok else "kinds WRONG"


def step3(m, train, test, pairs=500):
    """Of PAIRS random test pairs (Quality+ hidden), the share certify() gets
    right among those it calls a real difference ('A' or 'B'); calling a
    winner between two batches with equal true Quality+ counts as wrong.
    Ask 6 (accuracy) is checked, not scored: note gives its value, which
    must be a share in 0..1."""
    names, at = test[0], test[0].index("Quality+")
    rng, right, called = random.Random(1), 0, 0
    for _ in range(pairs):
        a, b = rng.sample(test[1:], 2)
        c = m.certify(train, blind(names, a), blind(names, b))
        if c not in ("A", "B", "same"):
            raise ValueError(f"certify returned {c!r}")
        if c != "same":
            called += 1
            right += (c == "A" and a[at] > b[at]) or (c == "B" and b[at] > a[at])
    if not called:
        raise ValueError("certify called every pair 'same'")
    acc = float(m.accuracy(train, test))
    ok = 0 <= acc <= 1
    return (100 * right / called,
            f"called {called}/{pairs}; accuracy {acc:.2f}" + ("" if ok else " WRONG"))


SCORERS = {1: step1, 2: step2, 3: step3}


#-- main -----------------------------------------------------
def score(step):
    "Score every treatment's stepN; return CSV rows."
    rows, suite = [], splits()
    for t, rep, folder in trials(step):
        which = range(len(suite)) if rep == 0 else [rep - 1]
        try:
            m, err = load(folder, step), None
        except Exception:
            m, err = None, traceback.format_exc(limit=1).strip().splitlines()[-1]
        for i in which:
            if i >= len(suite):
                continue  # more repeats than splits
            val, note = "", err or ""
            if m:
                random.seed(1)
                try:
                    val, note = SCORERS[step](m, *suite[i])
                    val = round(val, 2)
                except Exception as e:
                    note = f"{type(e).__name__}: {e}"
            rows.append(dict(treatment=t, repeat=rep, split=i + 1,
                             score=val, note=note))
    return rows


def main(step):
    rows = score(step)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"step{step}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["treatment", "repeat", "split", "score", "note"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['treatment']:>4} r{r['repeat']:02d} split{r['split']}  "
              f"{r['score']!s:>7}  {r['note']}")
    print(f"-> {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
