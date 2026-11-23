#!/usr/bin/env python3 -B
"""
step4.py: two more answers for the winemaker, built on ezr and
step2.

  start(names)        -> an empty model that knows only the header
  learn(model, row)   -> that model, after one more scored batch
  predict(model, row) -> the Quality+ to expect for one batch
  worth(model, row)   -> True if that batch earns a panel slot

Rows arrive as before: the header names the columns, `+` marks
what to maximize, `-` what to minimize, the rest are lab
readings.  step2.goalAt finds the panel's score (Quality+); it is
never a predictor, so nothing here can leak it.  Alcohol- comes
off the same lab sheet, so we read it as a reading; cells that
arrive blank are filled with the middle of what we have seen.

Why not reuse step2's forest?  Because the winemaker asked for
two things at once: an answer as each batch comes off the line,
and a sense that keeps sharpening "without starting over".  A
forest has to be regrown to learn, and regrowing it 1279 times
is both slow and a restart.  So this layer keeps a different
kind of memory.

What `learn` keeps.  Not the rows -- only their sums: the count,
the running mean and spread of every reading (ezr's Num does
that with Welford), the sum of the scores, and the cross-products
of the features with each other and with the score.  One batch
costs one pass over those sums and nothing else; no past batch is
ever revisited, and memory never grows.  Those sums are exactly
what a ridge fit needs, so `predict` can read the weights off
them whenever it is asked.

The features are each reading and its square, standardized and
centered at fit time (which the sums also support), so the fit
can bend a little and so no reading dominates just by being
measured in bigger units.  Ridge shrinks the weights, hard while
the season is young and gently once it is not; below a few rows
per weight we simply quote the running mean, and every answer is
clamped to the range of scores the panel has actually given.

Speed.  The weights are solved lazily: `learn` only marks them
stale, and the first `predict` after it does the solve.  So the
whole workload -- 1279 learns with 320 predicts after every
hundredth -- is 1279 cheap updates plus a dozen solves of a
small system, well under a second.
"""
import sys, time, random
from math import sqrt
import ezr, step2

WORTH = 6.0   # worth: the score that earns a panel slot
LAM   = 0.02  # ridge: shrink, on standardized features
PER   = 2     # fit only once we hold PER rows per weight
SQS   = True  # keep x*x terms, so the fit can bend

#-- the model ----------------------------------------------
def start(names: list[str]) -> ezr.o:
  """An empty model.  It knows the header: which column the panel
  scores, and which columns it may read."""
  if not names or not all(isinstance(s, str) for s in names):
    raise ValueError("step4: start wants a header row of strings")
  names = list(names)
  at    = step2.goalAt(names)
  xs, cols = [], {}
  for i, s in enumerate(names):
    if i == at or s[-1] == "X": continue   # the score, or skipped
    col = ezr.Col(s, i)
    if type(col) is ezr.Sym: continue      # readings here are nums
    xs += [i]
    cols[i] = col
  d = len(xs) * (2 if SQS else 1)
  return ezr.o(names=names, at=at, xs=xs, cols=cols, d=d,
               n=0, sy=0.0, y=ezr.Num(names[at], at),
               g=[[0.0]*d for _ in range(d)],   # upper triangle
               s=[0.0]*d, b=[0.0]*d,
               w=None, w0=0.0, fresh=False, lo=None, hi=None)

def feats(m: ezr.o, row: ezr.ROW) -> list[float]:
  "One batch as numbers: each reading, then each reading squared."
  out = []
  for i in m.xs:
    v = row[i] if i < len(row) else "?"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      v = m.cols[i].mu          # unknown: the middle we have seen
    out += [float(v)]
  if SQS: out += [v*v for v in list(out)]
  return out

#-- learn --------------------------------------------------
def learn(m: ezr.o, row: ezr.ROW) -> ezr.o:
  """The model after one more scored batch.  Updates sums only:
  no past row is re-read, and nothing is refit here."""
  at = m.at
  y  = row[at] if at < len(row) else "?"
  if isinstance(y, bool) or not isinstance(y, (int, float)):
    return m                  # an unscored batch teaches nothing
  y = float(y)
  f = feats(m, row)           # read the row before it moves the
  g, b, s = m.g, m.b, m.s     # middles it is imputed from
  for j in range(m.d):
    fj = f[j]
    if not fj: continue
    gj = g[j]
    for k in range(j, m.d): gj[k] += fj * f[k]
    b[j] += fj * y
    s[j] += fj
  for i in m.xs:              # Num ignores "?" and non-numbers
    v = row[i] if i < len(row) else "?"
    if not isinstance(v, bool) and isinstance(v, (int, float)):
      ezr.add(m.cols[i], v)
  ezr.add(m.y, y)
  m.n  += 1
  m.sy += y
  m.lo  = y if m.lo is None else min(m.lo, y)
  m.hi  = y if m.hi is None else max(m.hi, y)
  m.fresh = False             # the weights are now stale
  return m

#-- the fit ------------------------------------------------
def _gv(g: list, j: int, k: int) -> float:
  "The Gram matrix, kept as its upper triangle."
  return g[j][k] if j <= k else g[k][j]

def _gauss(a: list, b: list) -> list | None:
  "Solve A x = b, with partial pivoting.  None if singular."
  n = len(b)
  for i in range(n):
    p = max(range(i, n), key=lambda r: abs(a[r][i]))
    if abs(a[p][i]) < 1e-12: return None
    a[i], a[p] = a[p], a[i]
    b[i], b[p] = b[p], b[i]
    piv = a[i][i]
    for r in range(i+1, n):
      f = a[r][i] / piv
      if f:
        ar, ai = a[r], a[i]
        for c in range(i, n): ar[c] -= f * ai[c]
        b[r] -= f * b[i]
  x = [0.0]*n
  for i in range(n-1, -1, -1):
    x[i] = (b[i] - sum(a[i][k]*x[k]
                       for k in range(i+1, n))) / a[i][i]
  return x

def _fit(m: ezr.o) -> None:
  "Read the weights off the sums.  Called at most once per learn."
  m.fresh = True
  m.w, m.w0 = None, m.y.mu if m.n else 0.0
  d, n = m.d, m.n
  if d == 0 or n < PER*d + 4: return   # too soon to trust a fit
  g, b = m.g, m.b
  ybar  = m.sy / n
  mu    = [v/n for v in m.s]
  sd    = [sqrt(max(0.0, g[j][j] - n*mu[j]*mu[j]) / n)
           for j in range(d)]
  use   = [j for j in range(d)         # drop flat features
           if sd[j] > 1e-9 * (1 + abs(mu[j]))]
  if not use: return
  lam = LAM + d/n                      # shrink hard while young
  k   = len(use)
  a   = [[0.0]*k for _ in range(k)]
  rhs = [0.0]*k
  for ja, j in enumerate(use):         # the correlation matrix
    for kb, kk in enumerate(use):
      a[ja][kb] = ((_gv(g, j, kk) - n*mu[j]*mu[kk])
                   / (n * sd[j] * sd[kk]))
    a[ja][ja] += lam
    rhs[ja] = (b[j] - n*mu[j]*ybar) / (n * sd[j])
  x = _gauss(a, rhs)
  if x is None: return                 # fall back to the mean
  w = [0.0]*d
  for ja, j in enumerate(use): w[j] = x[ja] / sd[j]
  m.w  = w
  m.w0 = ybar - sum(w[j]*mu[j] for j in range(d))

#-- predict ------------------------------------------------
def predict(m: ezr.o, row: ezr.ROW) -> float:
  """The Quality+ to expect for ROW, from its lab sheet and from
  every scored batch learned so far."""
  if not m.fresh: _fit(m)
  if not m.n: return 0.0               # we know nothing yet
  y = m.w0
  if m.w:
    y += sum(wj*v for wj, v in zip(m.w, feats(m, row)) if wj)
  return float(min(m.hi, max(m.lo, y)))   # stay in known range

def worth(m: ezr.o, row: ezr.ROW) -> bool:
  "Is this batch worth a panel slot?  Yes at WORTH or better."
  return predict(m, row) >= WORTH - 1e-9

#-- demo ---------------------------------------------------
def main(args: list[str]) -> None:
  "step4.py [file.csv] -- learn a season one batch at a time."
  rows = list(ezr.csv(args[0] if args else ezr.the.File))
  random.seed(ezr.the.Seed)
  data = rows[1:]
  random.shuffle(data)
  n     = max(1, len(data) // 5)
  test  = [r for r in data[:n]]
  train = data[n:]
  at    = step2.goalAt(rows[0])
  got   = [r for r in test if r[at] != "?"]
  m, t0, marks = start(rows[0]), time.time(), []
  for i, row in enumerate(train, 1):
    learn(m, row)
    if i % 100 == 0:
      err = [abs(predict(m, r) - r[at]) for r in got]
      marks += [(i, sum(err)/len(err))]
  secs = time.time() - t0
  print(f"# {len(train)} learns, {len(got)} predicts per 100"
        f"   in {secs:.1f}s")
  for i, mae in marks:
    print(f"  after {i:>5} batches   mae {ezr.say(mae,3)}")
  dumb = [abs(m.y.mu - r[at]) for r in got]
  print(f"  (as if mean)           mae "
        f"{ezr.say(sum(dumb)/len(dumb),3)}")
  tp = fp = fn = tn = 0
  for r in got:
    good, sent = r[at] >= WORTH, worth(m, r)
    tp, fp = tp + (good and sent),     fp + (not good and sent)
    fn, tn = fn + (good and not sent), tn + (not good
                                             and not sent)
  print(f"# worth: sending at {ezr.say(WORTH,1)} or better")
  print(f"  {ezr.say(dict(sent_good=tp, sent_weak=fp,
                          held_good=fn, held_weak=tn))}")
  if tp + fp: print("  of those sent, good "
                   f"{ezr.say(100*tp/(tp+fp),1)}%")
  if tp + fn: print("  of the good, sent   "
                   f"{ezr.say(100*tp/(tp+fn),1)}%")
  print(f"  right               "
        f"{ezr.say(100*(tp+tn)/max(1,len(got)),1)}%")

if __name__ == "__main__": main(sys.argv[1:])
