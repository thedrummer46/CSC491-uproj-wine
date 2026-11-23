#!/usr/bin/env python3 -B
"""
step3.py: two more answers for the winemaker, built on ezr and
step2.

  certify(train, a, b)  -> 'A', 'B' or 'same'
  accuracy(train, test) -> share of guesses that land on the
                           panel's own score

Rows arrive as before: rows[0] is the header (strings) and the
rest are numbers; `+` marks what to maximize, `-` what to
minimize, everything else is a lab reading.  The panel's score
is Quality+ (step2.goalAt finds it), and it is never a
predictor, so nothing here can leak it.

certify.  "A beats B" is a real difference only if it survives
two doubts.

First doubt: the panel.  Scores come off a tasting sheet in
whole points, so two batches less than half a point apart get
written down the same way.  We read that step size out of the
training scores -- the smallest gap between two different
scores, halved -- rather than hard-coding 0.5, so a finer
sheet would get a finer floor.

Second doubt: the model.  step2's forest sends a batch to one
leaf per tree, and every leaf holds real past batches with real
panel scores.  Pooling those scores gives a sample of what the
panel has actually said about batches like this one, tasting
noise and all.  That gives us two samples, and ezr already
knows how to ask whether two samples are distinguishable:
`same`, i.e. Cliff's delta plus Kolmogorov-Smirnov.  If it
cannot tell them apart, we say 'same'.

Both doubts lean cautious -- the within-leaf spread carries the
full tasting noise -- so a verdict of 'A' or 'B' is a claim we
can stand behind.  The direction, when there is one, is
step2.guess's, the same number we quote elsewhere, so our
story never contradicts itself.

accuracy.  Guess each unseen batch from its lab sheet, write
the guess the way the panel would write it, and count the
batches where that lands on the panel's own score.  Train rows
are only ever read.

Speed.  Growing forests is the only slow part, and step2 grows
each at most once per train set; 500 certify calls and 320
accuracy calls then cost a few tree walks apiece.  Two things
keep that true: we hold on to the first train list we are
given and hand step2 that same object every time (so its cache
keeps hitting even if the caller rebuilds the list), and we cap
how long any one forest may spend growing.
"""
import sys, time, random, bisect
from math import floor
import ezr, step2

EPS   = 0        # same(): 0 means cliffs and ks, without cohen
SHARE = 12.0     # seconds any one forest may spend growing
KEEP  = 2048     # most batches to cache a pooled sample for

#-- the train set ------------------------------------------
def _key(train: ezr.ROWS) -> tuple:
  "A cheap fingerprint for one train set."
  return (len(train), tuple(str(s) for s in train[0]),
          tuple(train[1]) if len(train) > 1 else (),
          tuple(train[-1]))

_SEEN: dict = {}

def _seen(train: ezr.ROWS) -> ezr.o:
  "All we know about one train set, built once."
  if not train or not train[0]:
    raise ValueError("step3: no train rows given")
  k = _key(train)
  if k not in _SEEN:
    _SEEN.clear()                   # one train set at a time
    _SEEN[k] = ezr.o(train=train, m=None, levels=None, step=0,
                     ints=True, ys={}, samples={})
  return _SEEN[k]

def _model(c: ezr.o) -> ezr.TBL:
  "step2's model for this train set -- the one `guess` uses."
  if c.m is None: c.m = step2._model(c.train)
  return c.m

def _capped(fn, *args):
  "Run FN with a cap on how long a new forest may grow."
  old = step2.BUDGET
  try:
    step2.BUDGET = min(old, SHARE)
    return fn(*args)
  finally:
    step2.BUDGET = old

def _guess(c: ezr.o, row: ezr.ROW) -> float:
  "The score to expect for ROW, from step2."
  return _capped(step2.guess, c.train, row)

#-- the panel's scale --------------------------------------
def _levels(c: ezr.o) -> list:
  "The scores the panel has actually written down, sorted."
  if c.levels is None:
    at   = _model(c).at
    vs   = sorted({r[at] for r in c.m.data})
    gaps = [b - a for a, b in zip(vs, vs[1:])]
    c.levels = vs
    c.step   = min(gaps) if gaps else 0
    c.ints   = all(float(v).is_integer() for v in vs)
  return c.levels

def _floor(c: ezr.o) -> float:
  "Half a step on the sheet: under this, nothing shows up."
  _levels(c)
  return (c.step or 0) / 2

def _snap(c: ezr.o, y: float) -> float:
  "A guess, written the way the panel would write it."
  vs = _levels(c)
  if not vs:   return float(y)
  if c.ints:   return float(floor(y + .5))
  i = bisect.bisect_left(vs, y)
  if i <= 0:         return float(vs[0])
  if i >= len(vs):   return float(vs[-1])
  return float(vs[i] if vs[i] - y < y - vs[i-1] else vs[i-1])

def _hit(c: ezr.o, y: float, want: ezr.ATOM) -> bool:
  "Did one guess land on the panel's score?"
  if _snap(c, y) == want: return True
  return bool(c.step) and abs(y - want) <= c.step/2 + 1e-9

#-- what the panel said about batches like this one --------
def _trees(c: ezr.o, row: ezr.ROW) -> list:
  """The trees `guess` used for ROW.  Mirrors the choice inside
  step2.guess: the forest for whichever extra goals this row
  happens to carry."""
  m  = _model(c)
  ex = tuple(sorted(at for at in m.ok if row[at] != "?"))
  if ex not in m.forests and len(m.forests) >= 2: ex = ()
  return _capped(step2.forest, m, ex)

def _leafys(c: ezr.o, node: ezr.NODE) -> list:
  "The panel scores held inside one leaf, cached."
  k = id(node)                 # forests are kept, so ids hold
  if k not in c.ys:
    at = _model(c).at
    c.ys[k] = [r[at] for r in node.rows if r[at] != "?"]
  return c.ys[k]

def _spread(c: ezr.o, row: ezr.ROW) -> list:
  "ROW's leaf scores, pooled over the forest.  Sorted."
  k = tuple(row)
  if k not in c.samples:
    if len(c.samples) > KEEP: c.samples.clear()
    out = []
    for t in _trees(c, row): out += _leafys(c, ezr.leaf(t, row))
    c.samples[k] = sorted(out)
  return c.samples[k]

#-- certify ------------------------------------------------
def certify(train: ezr.ROWS, a: ezr.ROW, b: ezr.ROW) -> str:
  """Which batch will score better: 'A', 'B', or 'same' when the
  gap is small enough to be tasting noise."""
  c = _seen(train)
  ga, gb = _guess(c, a), _guess(c, b)
  if abs(ga - gb) < _floor(c): return "same"      # the panel
  if ezr.same(_spread(c, a), _spread(c, b), EPS):
    return "same"                                 # the model
  return "A" if ga > gb else "B" if gb > ga else "same"

#-- accuracy -----------------------------------------------
def _data(rows: ezr.ROWS) -> ezr.ROWS:
  "ROWS without their header, if they came with one."
  if not rows: return []
  return (rows[1:] if all(isinstance(v, str) for v in rows[0])
          else rows)

def accuracy(train: ezr.ROWS, test: ezr.ROWS) -> float:
  """How often our estimates are right on batches we have never
  seen: the share of scored TEST batches whose guess, written on
  the panel's scale, is the panel's own score."""
  c  = _seen(train)
  at = _model(c).at
  n  = hits = 0
  for row in _data(test):
    want = row[at] if at < len(row) else "?"
    if isinstance(want, bool): continue
    if not isinstance(want, (int, float)): continue
    n += 1
    if _hit(c, _guess(c, row), want): hits += 1
  return hits / n if n else 0.0

#-- demo ---------------------------------------------------
def main(args: list[str]) -> None:
  "step3.py [file.csv] -- hold out a fifth, then check ourselves."
  rows = list(ezr.csv(args[0] if args else ezr.the.File))
  random.seed(ezr.the.Seed)
  data = rows[1:]
  random.shuffle(data)
  n     = max(1, len(data) // 5)
  test  = data[:n]
  train = [rows[0]] + data[n:]
  at    = step2.goalAt(rows[0])
  t0    = time.time()
  acc   = accuracy(train, test)
  print(f"# accuracy, on {len(test)} unseen batches")
  print(f"  right    {ezr.say(100*acc,1)}%"
        f"   in {time.time()-t0:.1f}s")
  t1    = time.time()
  tally = {"A": 0, "B": 0, "same": 0}
  c, right, calls = _seen(train), 0, 0
  for _ in range(500):
    a, b = random.choice(test), random.choice(test)
    verdict = certify(train, a, b)
    tally[verdict] += 1
    if a[at] != "?" and b[at] != "?":
      calls += 1
      truth = ("same" if abs(a[at] - b[at]) < _floor(c) else
               "A" if a[at] > b[at] else "B")
      right += verdict == truth
  print(f"# certify, 500 pairs in {time.time()-t1:.1f}s")
  print(f"  verdicts {ezr.say(tally)}")
  print(f"  agrees with the panel "
        f"{ezr.say(100*right/max(1,calls),1)}%")

if __name__ == "__main__": main(sys.argv[1:])
