#!/usr/bin/env python3 -B
"""
step6.py: a winemaker's assistant, built on ezr.py, step1..step4.

Two questions about the readings themselves:

  "Which two or three readings matter most for whether a batch
   scores well?"
      -> blame(train, k=3)   -> ["Volatileacidity", ...]

  "Our low-scoring batches: which lab readings are dragging
   them down?"
      -> dragging(train)     -> [("Volatileacidity","high"), ...]

TRAIN is [header, *data], as in step1..step5: the header names
the columns (trailing "+" = maximize, "-" = minimize, the rest
are lab readings) and every later row holds numbers.  Here the
scores are already in hand -- this is a look back over the
season, not a tasting plan -- so nothing is blinded and nothing
is labelled.

Both answers range over *lab readings only*: the cells the lab
hands us, which the panel does not own.  So "Quality+" is the
thing being explained and "Alcohol-", a goal the panel scores
even though a meter reads it, stays out of both lists.  To keep
the credit honest, the goal columns are dropped from the inputs
too: a model that can see Alcohol- soaks up its signal there and
then has nothing left to say about the readings we are allowed
to name.

How does `blame` rank them?  step2 already grows a small forest
that predicts the panel's score, so we reuse it on the readings
alone and then scramble one reading at a time: deal its column's
values out at random among the rows and re-measure the error.  A
reading the forest leans on cannot be scrambled without the
error growing; a reading it ignores costs nothing.  That growth,
averaged over REPEATS shuffles, is the reading's importance.
Permutation beats counting tree splits here because two
correlated readings share the splits between them, which flips
their order run to run.

How does `dragging` work?  Split the cellar at step4's bar
(PANEL = 6): poor batches below it, good batches at or above.
For each reading, compare the two groups.  "Too high" or "too
low" is the direction of the gap between their means, measured
in pooled standard deviations, and "worst first" is that gap's
size.  A gap is only reported if ezr's three tests all call the
two groups different -- cliffs (rank imbalance), ks (biggest gap
between the curves) and cohen (means more than EFFECT pooled
standard deviations apart).  Readings that merely wobble are
left off the list rather than padded on.

Two honest limits.  Importance is measured on the training rows
themselves, so it says what this forest uses, not what a cellar
would do under intervention: dropping sulphates by half is a
different experiment.  And `dragging` compares groups one
reading at a time, so two readings that always move together
will both appear, even if only one is the real culprit.  Read
the two answers together: a reading in both lists is a lever,
a reading in `dragging` alone is a symptom.

Costs: one forest (step2 builds it once and caches it), then
SAMPLE x REPEATS predictions per reading for `blame`, and one
linear pass per reading for `dragging`.  On 1279 rows with
eleven readings that is a couple of seconds, well inside the
budget.
"""
# pylint: disable=invalid-name

import sys
from math import sqrt
from random import Random
from typing import Any

import ezr
from ezr import adds, div, same, o, the, say
import step1
from step1 import names, data
import step2
from step2 import goalAt, model, guess, score
import step3
from step3 import isHeader, EFFECT
import step4
from step4 import PANEL

#-- types --------------------------------------------------
type ROW  = list[Any]
type ROWS = list[ROW]
type PREP = o        # o(ns, at, keep, xs, rat, rows, red)

#-- dials --------------------------------------------------
SAMPLE  = 512    # rows weighed when scrambling a reading
REPEATS = 3      # shuffles averaged per reading

#-- the header, re-read ------------------------------------
def Prep(train: ROWS) -> PREP | None:
  """The score, the readings, and the rows that carry both.

  `red` is TRAIN with every goal column but the score removed,
  so step2's forest sees readings and nothing else.
  """
  train = train or []
  if len(train) < 2: return None
  ns = names(train)
  if not ns: return None
  at = goalAt(ns)
  keep = [i for i, s in enumerate(ns)
          if i == at or s[-1:] not in "+-!X"]
  rows = [r for r in data(train)
          if not isHeader(r) and score(r, at) is not None]
  if len(keep) < 2 or not rows: return None
  red = [[ns[i] for i in keep]] + \
        [[(r[i] if i < len(r) else "?") for i in keep]
         for r in rows]
  return o(ns=ns, at=at, keep=keep, rat=keep.index(at),
           xs=[i for i in keep if i != at], rows=rows, red=red)

_cache: dict[int, tuple[ROWS, PREP | None]] = {}

def prep(train: ROWS) -> PREP | None:
  "Prep for TRAIN, built once: both answers want the same one."
  hit = _cache.get(id(train))
  if hit and hit[0] is train: return hit[1]
  out = Prep(train)
  if len(_cache) > 2: _cache.clear()   # one split at a time
  _cache[id(train)] = (train, out)     # keep TRAIN alive, so
  return out                           # its id stays its own

#-- request 1: which readings matter? ----------------------
def errs(p: PREP, rows: ROWS) -> float:
  "Mean absolute error of the forest over ROWS (reduced rows)."
  e, n = 0.0, 0
  for r in rows:
    y = score(r, p.rat)
    if y is None: continue
    n += 1
    e += abs(y - guess(p.red, r))
  return e / n if n else 0.0

def shuffled(rows: ROWS, at: int, rnd: Random) -> ROWS:
  "ROWS again, with column AT's values dealt out at random."
  vs = [r[at] for r in rows]
  rnd.shuffle(vs)
  out = []
  for r, v in zip(rows, vs):
    r2 = list(r)
    r2[at] = v
    out += [r2]
  return out

def importance(p: PREP) -> list[tuple[str, float]]:
  "Each reading, with the error its scrambling costs.  Big first."
  if p is None or model(p.red) is None: return []
  rnd  = Random(the.Seed)
  rows = p.red[1:]
  if len(rows) > SAMPLE: rows = rnd.sample(rows, SAMPLE)
  base = errs(p, rows)
  out  = []
  for j in range(len(p.keep)):
    if j == p.rat: continue            # the score is not a reading
    hurt = sum(errs(p, shuffled(rows, j, rnd))
               for _ in range(REPEATS)) / REPEATS
    out += [(p.ns[p.keep[j]], max(0.0, hurt - base))]
  out.sort(key=lambda z: -z[1])
  return out

def blame(train: ROWS, k: int = 3) -> list[str]:
  """The K readings that matter most for the panel's score.

  Most important first, and never a goal column: these are
  readings the lab hands us, so they are the ones a winemaker
  can actually chase.  Fewer than K come back if there are
  fewer readings than that.
  """
  p = prep(train)
  if p is None or k < 1: return []
  return [nm for nm, _ in importance(p)[:k]]

#-- request 2: what drags the poor batches down? -----------
def sides(p: PREP) -> tuple[ROWS, ROWS]:
  "Poor batches and good ones, split at the panel's bar."
  poor, good = [], []
  for r in p.rows:
    y = score(r, p.at)
    if y is None: continue
    (poor if y < PANEL else good).append(r)
  return poor, good

def nums(rows: ROWS, at: int) -> list[float]:
  "Column AT of ROWS, keeping only the cells that are numbers."
  out = []
  for r in rows:
    v = r[at] if at < len(r) else "?"
    if not isinstance(v, bool) and isinstance(v, (int, float)):
      out += [float(v)]
  return out

def gap(xs: list[float], ys: list[float]) -> float:
  "How far XS sits above YS, in pooled standard deviations."
  a, b = adds(xs), adds(ys)
  pool = ((a.n-1)*div(a)**2 + (b.n-1)*div(b)**2) / (a.n+b.n-2)
  return 0.0 if pool < 1e-32 else (a.mu - b.mu) / sqrt(pool)

def shifts(p: PREP) -> list[tuple[str, str, float]]:
  "Readings that really differ in poor batches.  Worst first."
  poor, good = sides(p)
  if len(poor) < 2 or len(good) < 2: return []
  out = []
  for at in p.xs:
    xs, ys = nums(poor, at), nums(good, at)
    if len(xs) < 2 or len(ys) < 2: continue
    if same(xs, ys, eps=EFFECT): continue   # just a wobble
    if (d := gap(xs, ys)) == 0: continue
    out += [(p.ns[at], "high" if d > 0 else "low", abs(d))]
  out.sort(key=lambda z: -z[2])
  return out

def dragging(train: ROWS) -> list[tuple[str, str]]:
  """The readings that are off in the poor batches, worst first.

  Each pair is (reading, "high") or (reading, "low"): which way
  that reading leans in batches the panel scored below 6,
  compared with the batches it scored 6 or better.  Only shifts
  that survive cliffs, ks and cohen are listed, so an empty list
  means the poor batches look like the good ones on the bench.
  """
  p = prep(train)
  if p is None: return []
  return [(nm, way) for nm, way, _ in shifts(p)]

#-- show me ------------------------------------------------
def eg_h() -> None:
  "Print help."
  print(__doc__)

def eg_all() -> None:
  "Load ezr's -File, split it in half, and answer both questions."
  rows  = list(ezr.csv(ezr.the.File))
  ns, d = names(rows), data(rows)
  d     = Random(the.Seed).sample(d, len(d))
  n     = len(d) // 2
  train = [ns] + d[:n]
  p     = prep(train)
  if p is None: return print("no scored rows")

  imps = importance(p)
  wide = max((v for _, v in imps), default=0) or 1
  print(f"\n-- what moves {ns[p.at]} ({len(p.rows)} batches) ----")
  print("   reading                 cost of scrambling it")
  for nm, v in imps:
    print(f"  {nm:>18} : {say(v,3):>6} "
          f"{'#' * round(30 * v / wide)}")
  top = blame(train, 3)
  print(f"\n  the big three : {', '.join(top)}")

  poor, good = sides(p)
  bad = shifts(p)
  print(f"\n-- {len(poor)} poor batches (under {say(PANEL,1)}) vs "
        f"{len(good)} good ----")
  for nm, way, d in bad:
    at = ns.index(nm)
    mp = adds(nums(poor, at)).mu
    mg = adds(nums(good, at)).mu
    print(f"  {nm:>18} : too {way:<4} "
          f"(poor {say(mp,3)} vs good {say(mg,3)}, "
          f"{say(d,2)} sd)")
  if not bad: print("  nothing stands out")

  levers = [nm for nm, _ in dragging(train) if nm in top]
  print(f"\n-- where to start ------------------------------")
  print(f"  both lists (fix these) : "
        f"{', '.join(levers) or 'none'}")
  print(f"  off, but not a lever   : "
        f"{', '.join(nm for nm, _ in dragging(train)
                     if nm not in top) or 'none'}")

if __name__ == "__main__":
  ezr.main(sys.argv[1:], globals())
