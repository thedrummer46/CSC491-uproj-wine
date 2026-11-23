"""Score every treatment's stepN.py on the one test suite (data/splits.json).

    python -m harness.score 1        # score step 1 for every treatment found
    python -m harness.score 3 600    # same, with a 600 s limit instead of 90

Per docs/assessment.md: LLM repeat r (treatments/tK/rNN/) is scored on split r;
T7 (treatments/t7/) is written once and scored on all 5 splits. A step that
crashes, breaks the interface, or takes over 90 s on one split scores blank
and keeps its reason -- broken stays broken. Writes results/scores/stepN.csv and prints it.
Harness code: used to SCORE treatments, never shown to them.
"""
import csv, importlib, json, random, re, sys
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
    """% of the batches remember() returns (at most 5%) that really are in
    the test pool's true top 5% by `wins`. Doesn't depend on how many are
    returned or in what order, so returning fewer can't inflate it.
    Ask 2 (summary) is checked, not scored: note says if it passed."""
    got = m.remember(test)
    cap = int(0.05 * (len(test) - 1))
    if not got or len(got) > cap:
        raise ValueError(f"remember returned {len(got)} rows; want 1..{cap}")
    win = ezr.wins(ezr.Tbl(test))
    bar = sorted((win(r) for r in test[1:]), reverse=True)[cap - 1]
    score = 100 * sum(win(r) >= bar for r in got) / len(got)
    s = m.summary(test)
    tbl = ezr.Tbl(test)
    def near_middle(c):  # mean or median both count as "typical"
        xs = sorted(r[c.at] for r in tbl.rows)
        return min(abs(float(s[c.txt]) - mid) for mid in (c.mu, xs[len(xs) // 2])) <= 0.1 * (c.sd or 1)
    ok = set(s) == set(test[0]) and all(near_middle(c) for c in tbl.cols.values())
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


def step4(m, train, test, every=100):
    """Feed train batches to learn() one at a time; after every EVERY,
    take predict()'s mean absolute error on the test batches (Quality+
    hidden). Score = mean error over that learning curve (lower is better),
    so learning fast counts, not just ending well. Note gives the first
    and last points of the curve, and checks worth() agrees with
    'predicted Quality+ of 6 or more' on the final model."""
    names, at = test[0], test[0].index("Quality+")
    blinds = [blind(names, r) for r in test[1:]]
    model, curve = m.start(names), []
    for i, row in enumerate(train[1:], 1):
        model = m.learn(model, row)
        if i % every == 0:
            curve.append(sum(abs(float(m.predict(model, b)) - r[at])
                             for b, r in zip(blinds, test[1:])) / len(blinds))
    ok = all(bool(m.worth(model, b)) == (float(m.predict(model, b)) >= 6)
             for b in blinds[:50])
    return (sum(curve) / len(curve),
            f"curve {curve[0]:.2f} -> {curve[-1]:.2f}" + ("" if ok else "; worth WRONG"))


def hide_goals(rows):
    """[header] + rows with every goal cell '?', plus label(): the hidden
    row -> its real row, counting each call as one tasting."""
    names = rows[0]
    goals = [i for i, n in enumerate(names) if n[-1] in "+-"]
    hidden = [["?" if i in goals else v for i, v in enumerate(r)] for r in rows[1:]]
    real = {id(h): r for h, r in zip(hidden, rows[1:])}
    calls = [0]
    def label(h):
        calls[0] += 1
        return list(real[id(h)])
    return [names] + hidden, label, calls


def step5(m, train, test, budget=20):
    """`wins` (0 = average batch, 100 = the best) of the batch choose()
    picks from the train batches, goals hidden until label()ed, at most
    BUDGET tastings; scored on the true goals over the whole train pool.
    Ask 13 (taste) is checked: note says if it stayed in budget."""
    win = ezr.wins(ezr.Tbl(train))
    rows, label, calls = hide_goals(train)
    pick = m.choose(rows, label, budget)
    if calls[0] > budget:
        raise ValueError(f"choose tasted {calls[0]} batches; budget {budget}")
    if "?" in pick:
        raise ValueError("choose returned a batch it never tasted")
    rows, label, calls = hide_goals(train)
    t = m.taste(rows, label, budget)
    ok = calls[0] <= budget and 0 < len(t) <= budget and all("?" not in r for r in t)
    return win(pick), f"tasted {len(t)}" + ("" if ok else " WRONG")


def refit(cols, train, test):
    """A plain predictor that may only use COLS: an ezr tree on Quality+
    alone, every other column skipped. Its mean absolute error on test."""
    names = [n if n in cols or n == "Quality+" else n.rstrip("+-") + "X"
             for n in train[0]]
    tbl, at = ezr.Tbl([names] + train[1:]), names.index("Quality+")
    node = ezr.tree(tbl, tbl.rows)
    return sum(abs(sum(r[at] for r in ezr.leaf(node, x).rows)
                   / len(ezr.leaf(node, x).rows) - x[at])
               for x in test[1:]) / (len(test) - 1)


def step6(m, train, test, k=3):
    """Mean absolute error on test of a plain predictor refit on ONLY the K
    readings blame() names (lower is better): good blame keeps most of the
    predictive power. Ask 9 (dragging) is checked: note says if it returns
    (lab reading, 'high'/'low') pairs."""
    names = train[0]
    readings = {n for n in names if n[-1] not in "+-!X"}
    got = list(m.blame(train, k))
    if len(got) != k or len(set(got)) != k or not set(got) <= readings:
        raise ValueError(f"blame returned {got}; want {k} distinct lab readings")
    d = m.dragging(train)
    ok = all(len(p) == 2 and p[0] in readings and p[1] in ("high", "low") for p in d)
    return refit(got, train, test), (", ".join(got) + ("" if ok else "; dragging WRONG"))


SCORERS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6}


#-- main -----------------------------------------------------
LIMIT = 90     # CPU seconds for everything on one split (stated in every prompt)
WALL = 600     # wall-clock backstop, for code that hangs without using CPU
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


def cpu(proc):
    "CPU seconds used so far by PROC and anything it started."
    import psutil
    try:
        p = psutil.Process(proc.pid)
        return sum(sum(q.cpu_times()[:2]) for q in [p] + p.children(recursive=True))
    except psutil.Error:
        return 0.0


def job(step, folder, i, conn):
    """Child process: load FOLDER/stepN.py fresh, score it on split I,
    send back (score, note, CPU seconds). Runs alone, so a slow step can be
    stopped."""
    import time
    try:
        m = load(Path(folder), step)
        random.seed(1)
        val, note = SCORERS[step](m, *splits()[i])
        conn.send((round(val, 2), note, round(time.process_time(), 1)))
    except Exception as e:
        conn.send(("", f"{type(e).__name__}: {e}", round(time.process_time(), 1)))
    conn.close()


def score(step, limit=LIMIT, workers=None):
    """Score every treatment's stepN, each (treatment, repeat, split) in its
    own single-threaded process, several at once. The limit is on each job's
    own CPU time, so jobs running side by side can't push each other over it.
    Any job past LIMIT CPU seconds (or WALL seconds of clock) is stopped and
    scores blank as 'too slow'. Returns CSV rows."""
    import multiprocessing as mp, os, time
    for k in THREADS:            # children inherit: one thread each, so
        os.environ[k] = "1"      # numpy/scikit-learn don't fight for cores
    n = len(json.loads(SPLITS.read_text())["splits"])
    todo = [(t, rep, folder, i) for t, rep, folder in trials(step)
            for i in (range(n) if rep == 0 else [rep - 1]) if i < n]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    running, done = {}, {}
    while todo or running:
        while todo and len(running) < workers:
            key = todo.pop(0)
            parent, child = mp.Pipe(False)
            proc = mp.Process(target=job, args=(step, str(key[2]), key[3], child))
            proc.start()
            child.close()
            running[key] = (proc, parent, time.time(), [0.0])
        for key, (proc, parent, t0, used) in list(running.items()):
            used[0] = max(used[0], cpu(proc))      # last reading before exit
            secs = round(used[0], 1)
            if parent.poll():
                done[key] = parent.recv()
            elif not proc.is_alive():
                done[key] = ("", "crashed without an error message", secs)
            elif used[0] > limit or time.time() - t0 > WALL:
                proc.terminate()
                done[key] = ("", f"too slow (>{limit} CPU s)", secs)
            else:
                continue
            proc.join()
            del running[key]
        time.sleep(0.2)
    return [dict(treatment=t, repeat=rep, split=i + 1, score=done[(t, rep, f, i)][0],
                 note=done[(t, rep, f, i)][1], seconds=done[(t, rep, f, i)][2])
            for t, rep, f, i in sorted(done, key=lambda k: (int(k[0][1:]), k[1], k[3]))]


def main(step, limit=LIMIT):
    rows = score(step, limit)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"step{step}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["treatment", "repeat", "split", "score", "note", "seconds"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['treatment']:>4} r{r['repeat']:02d} split{r['split']}  "
              f"{r['score']!s:>7}  {r['seconds']:>5}s  {r['note']}")
    print(f"-> {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]), *(int(a) for a in sys.argv[2:3]))
