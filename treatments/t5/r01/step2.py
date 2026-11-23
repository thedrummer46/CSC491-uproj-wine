#!/usr/bin/env python3 -B
"""
step2.py: two more answers for the winemaker, built on ezr.

  guess(train, row) -> the panel score to expect for one batch
  kinds(train)      -> a few kinds of batch, each with its score

Rows arrive as in step1: rows[0] is the header (strings) and the
rest are numbers.  The header marks say what each column is for:
`+` maximize, `-` minimize, everything else is a lab reading.

The score is Quality+.  The lab sheet is all you have before the
panel sits, so the predictors are the lab readings.  One other
goal (Alcohol-) is also measured in the lab, so if a batch comes
in with that cell filled we read it too; if it comes in blank we
fall back to a readings-only model.  The score itself is never a
predictor, so nothing here can leak it.

How: ezr's tree already hunts for the split whose two sides have
the tightest goals, so we reuse it.  We rewrite the header so
Quality+ is the only goal -- then "tightest ys" means "tightest
quality" -- and grow a small forest of those trees, each on a
bootstrap sample and a random two thirds of the columns.  A leaf
predicts its own mean quality, shrunk a little towards its
parent's, which keeps small leaves honest.  `guess` averages the
forest; one forest is built on the first call and reused for all
the later ones.

`kinds` grows one deliberately shallow tree instead: its leaves
are the kinds, their rows are the batches of that kind, and each
leaf's mean quality is the score to expect from it.
"""
import sys, time, random
import ezr

TREES  = 8       # guess: trees in the forest
CUTS   = 8       # guess: splits to try, per column
MM     = 2.0     # guess: shrink a leaf by MM rows of its parent
BUDGET = 20.0    # guess: seconds to spend growing one forest
CAP    = 20000   # guess: most rows shown to any one tree
GROUPS = 8       # kinds: at most this many kinds
FINE   = 16      # kinds: splits to try (few nodes, so look hard)

#-- what to predict ----------------------------------------
def goalAt(names: list[str]) -> int:
  """Which column is the panel's score?  The goal named Quality,
  else the last goal we are asked to maximize."""
  ups = [at for at, s in enumerate(names) if s[-1] in "+!"]
  all_ = [at for at, s in enumerate(names) if s[-1] in "+-!"]
  for at in ups + all_:
    if "quality" in names[at][:-1].lower(): return at
  if ups:  return ups[-1]
  if all_: return all_[-1]
  raise ValueError("step2: no goal column in the header")

def xnames(names: list[str], at: int,
           extra: tuple = ()) -> list[str]:
  """A header where AT is the only goal.  Other goals become
  plain columns if they are in EXTRA, else they are skipped (X)."""
  out = []
  for i, s in enumerate(names):
    if   i == at:          out += [s]            # keep the mark
    elif s[-1] in "+-!":   out += [s[:-1] if i in extra
                                   else s[:-1] + "X"]
    else:                  out += [s]
  return out

#-- the model ----------------------------------------------
def model(train: ezr.ROWS) -> ezr.TBL:
  "Everything guess needs.  Built once, per train set."
  names = train[0]
  at    = goalAt(names)
  data  = [r for r in train[1:] if r[at] != "?"]
  if not train[1:]: raise ValueError("step2: no training rows")
  if not data:      raise ValueError("step2: no scores in train")
  ys = [r[at] for r in data]
  ok = {i for i, s in enumerate(names)
        if i != at and s[-1] in "+-!"
        and sum(1 for r in data if r[i] != "?") >= .9*len(data)}
  return ezr.o(names=names, at=at, data=data, ok=ok,
               mu=sum(ys)/len(ys), lo=min(ys), hi=max(ys),
               forests={})

def leafsize(n: int) -> int:
  "Rows per leaf: enough to average, few enough to be specific."
  return max(6, min(40, n // 80))

def fit(node: ezr.NODE, at: int, prior: float) -> ezr.NODE:
  "Give every node a mean score, shrunk towards its parent's."
  ys = [r[at] for r in node.rows if r[at] != "?"]
  node.q = ((sum(ys) + MM*prior) / (len(ys) + MM) if ys
            else prior)
  for kid in node.kids: fit(kid, at, node.q)
  return node

def grow(tbl: ezr.TBL, data: ezr.ROWS, leaf: int,
         budget: float = BUDGET) -> list[ezr.NODE]:
  "A forest: bootstrap the rows, sample the columns, split."
  t0, xs, out = time.time(), tbl.x, []
  k   = max(2, (2*len(xs) + 2) // 3)
  n   = min(len(data), CAP)
  old = {key: ezr.the[key] for key in ("Leaf", "Cuts")}
  try:
    ezr.the.Leaf, ezr.the.Cuts = leaf, CUTS
    for _ in range(max(1, TREES)):
      tbl.x = random.sample(xs, min(k, len(xs)))
      out  += [ezr.tree(tbl, [random.choice(data)
                              for _ in range(n)])]
      if time.time() - t0 > budget: break
  finally:
    tbl.x = xs                  # leave the table as we found it
    ezr.the.update(old)
  return out

def forest(m: ezr.TBL, extra: tuple = ()) -> list[ezr.NODE]:
  "The forest for this set of EXTRA columns, built once."
  if extra not in m.forests:
    tbl = ezr.Tbl([xnames(m.names, m.at, extra)] + m.data)
    m.forests[extra] = [fit(t, m.at, m.mu) for t in
                        grow(tbl, tbl.rows, leafsize(len(m.data)))]
  return m.forests[extra]

#-- guess --------------------------------------------------
_CACHE: dict = {}

def _model(train: ezr.ROWS) -> ezr.TBL:
  "The model for TRAIN; the same train rows reuse the same model."
  key = (id(train), len(train), id(train[0]))
  if key not in _CACHE:
    _CACHE.clear()                     # one train set at a time
    _CACHE[key] = model(train)
  return _CACHE[key]

def guess(train: ezr.ROWS, row: ezr.ROW) -> float:
  """The score to expect for ROW, from its lab sheet alone.
  Learned from TRAIN, which it only ever reads."""
  m  = _model(train)
  ex = tuple(sorted(a for a in m.ok if row[a] != "?"))
  if ex not in m.forests and len(m.forests) >= 2: ex = ()
  ts = forest(m, ex)
  y  = sum(ezr.leaf(t, row).q for t in ts) / len(ts)
  return float(min(m.hi, max(m.lo, y)))    # stay in known range

#-- kinds --------------------------------------------------
def kinds(train: ezr.ROWS,
          most: int = GROUPS) -> list[tuple[ezr.ROWS, float]]:
  """A few kinds of batch, best first.  Each kind is the rows that
  fell into one leaf of one shallow tree, paired with the score to
  expect from that kind (its mean Quality+)."""
  m    = _model(train)
  ex   = tuple(sorted(m.ok))      # past batches: read it all
  tbl  = ezr.Tbl([xnames(m.names, m.at, ex)] + m.data)
  old  = {k: ezr.the[k] for k in ("Leaf", "Cuts")}
  try:
    ezr.the.Leaf = max(ezr.the.Leaf, len(m.data) // max(1, most))
    ezr.the.Cuts = FINE
    node = fit(ezr.tree(tbl, tbl.rows), m.at, m.mu)
  finally:
    ezr.the.update(old)
  groups = {id(lf): (lf, list(lf.rows))
            for lf in ezr.leafs(node)}
  for row in train[1:]:              # unscored batches join in
    if row[m.at] == "?":
      groups[id(ezr.leaf(node, row))][1].append(row)
  out = []
  for lf, rows in groups.values():
    ys = [r[m.at] for r in rows if r[m.at] != "?"]
    out += [(rows, float(sum(ys)/len(ys)) if ys else float(m.mu))]
  out.sort(key=lambda z: -z[1])
  return out

#-- demo ---------------------------------------------------
def main(args: list[str]) -> None:
  "step2.py [file.csv] -- hold out a fifth, then guess it."
  rows = list(ezr.csv(args[0] if args else ezr.the.File))
  random.seed(ezr.the.Seed)
  data = rows[1:]
  random.shuffle(data)
  n     = max(1, len(data) // 5)
  test  = data[:n]
  train = [rows[0]] + data[n:]
  at    = goalAt(rows[0])
  m     = model(train)
  t0    = time.time()
  err   = [abs(guess(train, r) - r[at])
           for r in test if r[at] != "?"]
  secs  = time.time() - t0
  dumb  = [abs(m.mu - r[at]) for r in test if r[at] != "?"]
  print(f"# guess: {len(err)} batches in {secs:.1f}s")
  print(f"  mae           {ezr.say(sum(err)/len(err),3)}")
  print(f"  mae (as if mean) {ezr.say(sum(dumb)/len(dumb),3)}")
  print(f"# kinds, of {len(train)-1} batches")
  try:    import step1
  except Exception: step1 = None
  for group, score in kinds(train):
    print(f"  expect {ezr.say(score,2):>5}  from {len(group):>5}"
          " batches")
    if step1:
      mids = step1.summary([rows[0]] + group, p=2)
      print("      ", ezr.say({k: v for k, v in mids.items()
                               if k[-1] not in "+-!"}, 2))

if __name__ == "__main__": main(sys.argv[1:])
