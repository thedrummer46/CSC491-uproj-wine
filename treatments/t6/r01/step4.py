#!/usr/bin/env python3 -B
"""
step4.py: a winemaker's assistant, built on ezr.py, step1..step3.

Two questions from the line, not the cellar:

  "As each new batch comes off the line, tell me right away
   whether it's worth sending to the panel."
      -> worth(model, row)

  "Keep sharpening your sense of what makes a good batch as
   this season's scores come in, without starting over."
      -> start(names), then learn(model, row) per batch

  start(names)        -> an empty model, knowing only the header
  learn(model, row)   -> the model, one scored batch wiser
  predict(model, row) -> the Quality+ we expect from one batch
  worth(model, row)   -> is that 6 or more?

NAMES is the header row of step1/step2/step3: trailing "+" is a
goal to maximize, "-" to minimize, the rest are lab readings.
As in step2, the panel scores one column (Quality+) and every
other cell -- including "Alcohol-" -- is an input, because the
lab hands it to us before the panel ever sits.

Why not a tree?  steps 2 and 3 grow trees, and a tree wants all
its rows at once: one new batch means growing it again.  On the
line, rows arrive one at a time, so we keep an *additive* model
instead:

  * one running ezr summary (Num or Sym) per lab reading;
  * each reading cut into BINS slices by ezr's `norm`, so a
    slice is "low-ish chlorides", not a magic number;
  * one weight per slice, and a running mean of the scores.

The score we expect is that running mean plus the weights of
the slices this batch falls into.  A scored batch teaches the
model by nudging exactly those slices toward its own score
(stochastic gradient descent on one-hot slices) and that is the
whole update: O(columns) per batch, nothing refit, no rows
revisited except a short rehearsal tail.

That tail matters.  One pass of plain SGD wastes its early
rows, so each new batch also re-nudges a few of the last MEMORY
batches.  The step size cools as `seen` grows, so late in the
season the model drifts rather than lurches.

Costs: learn is O(columns) x (1 + REPLAY), predict is
O(columns) and a dict lookup per reading.  A full season
(1279 learns, 4000 predicts) is well under a second.
"""
# pylint: disable=invalid-name

import sys
from random import Random
from typing import Any

import ezr
from ezr import Num, Sym, Col, add, norm, o, the, say
import step1
import step2
from step2 import goalAt, xname, score
import step3
from step3 import isHeader, near

#-- types --------------------------------------------------
type ROW   = list[Any]
type ROWS  = list[ROW]
type MODEL = o      # o(ns, at, n, y, lo, hi, cols, w, seen, buf)

#-- dials --------------------------------------------------
BINS   = 7      # slices per lab reading
RATE   = 0.3    # first step size: how far to chase one batch
COOL   = 150    # batches after which the step size halves-ish
REPLAY = 3      # recent batches rehearsed per new batch
MEMORY = 128    # how many recent batches we keep around
PANEL  = 6.0    # send it to the panel at this score or better

#-- request 0: an empty model ------------------------------
def start(names: list[str]) -> MODEL:
  """An empty model that knows only the header row NAMES.

  It predicts nothing useful yet (0.0, until it has seen one
  score), but it already knows which column the panel owns and
  which columns are readings.
  """
  if names and isinstance(names[0], (list, tuple)):
    names = names[0]                 # a step1-style [header, *d]
  ns = [str(s).strip() for s in (names or [])]
  at = goalAt(ns) if ns else 0
  return o(ns=ns, at=at, n=len(ns),
           y=Num(ns[at] if ns else "Y+", at),
           lo=None, hi=None,
           cols={i: Col(xname(s) or " ", i)
                 for i, s in enumerate(ns) if i != at},
           w={}, seen=0, buf=[], rnd=Random(the.Seed))

#-- slices: where does a reading fall? ---------------------
def slot(col: Any, v: Any) -> Any:
  "Which slice of this reading holds V?  None if we don't know."
  if v == "?" or v is None: return None
  if type(col) is Sym: return v
  if col.n < 2 or col.sd == 0: return BINS // 2
  return min(BINS - 1, int(norm(col, v) * BINS))

def feats(m: MODEL, row: ROW) -> list[tuple[int, Any]]:
  "The slices this batch falls into: one per known reading."
  out = []
  for at, col in m.cols.items():
    b = slot(col, row[at] if at < len(row) else "?")
    if b is not None: out += [(at, b)]
  return out

def raw(m: MODEL, fs: list[tuple[int, Any]]) -> float:
  "Running mean of the scores, plus each slice's weight."
  return m.y.mu + sum(m.w.get(f, 0.0) for f in fs)

#-- request 1: the score we expect -------------------------
def predict(m: MODEL, row: ROW) -> float:
  """Predicted Quality+ for one batch, from its readings alone.

  Clipped to the scores the panel has actually handed out, so a
  half-trained model cannot invent a 9.  With nothing learned
  yet there is nothing to say, so that answer is 0.0.
  """
  if not m or not m.y.n: return 0.0
  return max(m.lo, min(m.hi, raw(m, feats(m, row))))

#-- request 2: is it worth the panel's time? ---------------
def worth(m: MODEL, row: ROW) -> bool:
  "True if this batch looks like a 6 or better."
  return predict(m, row) >= PANEL

#-- request 3: one more batch, learned in place ------------
def fit(m: MODEL, row: ROW, y: float) -> None:
  "One gradient step: share the error over this batch's slices."
  fs = feats(m, row)
  if not fs: return
  step = (RATE / (1 + m.seen / COOL)) * (y - raw(m, fs))
  step /= len(fs)                   # the slices split the blame
  for f in fs: m.w[f] = m.w.get(f, 0.0) + step

def rehearse(m: MODEL) -> None:
  "Re-nudge a few recent batches: SGD settles sooner this way."
  for _ in range(min(REPLAY, len(m.buf))):
    row, y = m.buf[m.rnd.randrange(len(m.buf))]
    fit(m, row, y)

def learn(m: MODEL, row: ROW) -> MODEL:
  """The model after seeing one more scored batch.

  Updates in place and returns itself: the running summaries,
  the slice weights, and the rehearsal tail all move a little.
  Nothing is rebuilt.  A row with no score of its own still
  teaches the shape of the readings, and a stray header row is
  ignored.
  """
  if m is None or not row or isHeader(row): return m
  row = list(row)[:m.n] if m.n else list(row)
  row += ["?"] * max(0, m.n - len(row))
  y = score(row, m.at)
  if y is not None:
    m.lo = y if m.lo is None else min(m.lo, y)
    m.hi = y if m.hi is None else max(m.hi, y)
    add(m.y, y)                   # the intercept rides the mean
    fit(m, row, y)                # then the slices take the rest
  for at, col in m.cols.items(): add(col, row[at])
  if y is not None:
    m.seen += 1
    m.buf += [(row, y)]
    if len(m.buf) > MEMORY: m.buf.pop(0)
    rehearse(m)
  return m

def learns(m: MODEL, rows: ROWS) -> MODEL:
  "Convenience: learn a whole block of batches, in order."
  for row in rows or []: learn(m, row)
  return m

#-- what has it learned? -----------------------------------
def why(m: MODEL, k: int = 6) -> list[tuple[str, float]]:
  "The slices that move the score most, biggest push first."
  top = sorted(m.w.items(), key=lambda z: -abs(z[1]))[:k]
  return [(f"{m.ns[at]}[{b}/{BINS}]", w) for (at, b), w in top]

#-- show me ------------------------------------------------
def eg_h() -> None:
  "Print help."
  print(__doc__)

def report(m: MODEL, test: ROWS, seen: int) -> None:
  "One line: error, baseline, hit rate, and the panel's queue."
  ys = [score(r, m.at) for r in test]
  gs = [predict(m, r) for r in test]
  mae  = sum(abs(y-g) for y, g in zip(ys, gs)) / len(ys)
  base = sum(abs(y-m.y.mu) for y in ys) / len(ys)
  acc  = sum(near(g) == near(y) for y, g in zip(ys, gs))/len(ys)
  sent = [y for y, g in zip(ys, gs) if g >= PANEL]
  good = sum(1 for y in sent if y >= PANEL)
  hit  = f"{say(100*good/len(sent),0)}% of {len(sent)}" if sent \
         else "none sent"
  print(f"  {seen:>5} {say(mae,3):>6} {say(base,3):>6} "
        f"{say(100*acc,1):>6} {hit:>15}")

def eg_all() -> None:
  "Load ezr's -File, stream 80% of it, watch the other 20%."
  rows  = list(ezr.csv(ezr.the.File))
  ns, d = step1.names(rows), step1.data(rows)
  at    = goalAt(ns)
  d     = Random(the.Seed).sample(d, len(d))
  n     = len(d) - len(d)//5
  train = d[:n]
  test  = [r for r in d[n:] if score(r, at) is not None]

  m = start(ns)
  print(f"\n-- learning {ns[at]} as it arrives "
        f"({len(train)} train, {len(test)} test) ----")
  print("   seen    mae   base   acc%     panel queue")
  for i, row in enumerate(train, 1):
    learn(m, row)
    if i % 100 == 0 or i == len(train): report(m, test, i)

  print(f"\n-- what moves the score ------------------------")
  for txt, w in why(m):
    print(f"  {txt:>22} : {say(w,3):>7}")

  a = max(test, key=lambda r: predict(m, r))
  b = min(test, key=lambda r: predict(m, r))
  print(f"\n-- two batches off the line --------------------")
  for r in (a, b):
    print(f"  expect {say(predict(m,r),2):>5} "
          f"(panel said {say(score(r,at),2)}) -> "
          f"{'send it' if worth(m,r) else 'keep it back'}")

if __name__ == "__main__":
  ezr.main(sys.argv[1:], globals())
