#!/usr/bin/env python3 -B
"""
step3.py: answers the winemaker's last two questions.

  5. "When you say batch A will beat batch B, tell me
     whether that's a real difference or just tasting
     noise."
         -> certify(train, a, b)

  6. "How often are your score estimates right on batches
     they have never seen before?"
         -> accuracy(train, test)

Rows arrive as in step1 and step2: row[0] is the header of
strings, the rest are numbers.  A name ending in `+` is a
goal to maximize, `-` a goal to minimize, and the rest are
lab readings.

step2.guess already turns one lab sheet into one expected
score, by dropping the batch into a tree of kinds and
reporting that kind's mean score.  Here we ask what that
number is worth.

A guess is a mean over a whole kind of batch, so two
batches that land in kinds of nearly equal score are not
really being told apart: the gap is inside the spread of
the batches that made those two means.  So `certify` does
not compare the two numbers; it compares the two bags of
real scores behind them, using ezr's `same` (cliff's delta,
Kolmogorov-Smirnov, and -- since the winemaker cares about
tasting-sized differences -- Cohen's d).  Indistinguishable
bags mean "same", however the two means happen to fall.

`accuracy` is the plain question the winemaker asked: on
batches held out of training, how often does the guess,
rounded to the nearest whole point, hit the panel's
Quality+ exactly?
"""
import ezr
import step2


# How wide a difference in panel score still counts as
# tasting noise: `same`'s third test (Cohen) fires with this
# as its effect size, in pooled standard deviations.
NOISE = 0.35


def _scores(train: list[list], row: list) -> list[float]:
  "The real panel scores of the kind of batch ROW belongs to."
  fit = step2._fit(train)
  at  = fit.tbl.at
  kin = ezr.leaf(fit.node, row).rows
  return [r[at] for r in kin if step2._known(r, at)]


def certify(train: list[list], a: list, b: list) -> str:
  """Which of two batches scores better -- if either does.

  Each batch picks out a kind of past batch (its leaf in
  TRAIN's tree), and each kind brings a bag of real panel
  scores.  If those two bags are indistinguishable we say
  "same": the winemaker would be reading noise.  Otherwise
  the batch whose bag has the higher mean wins.

  Two batches of the same kind are always "same" -- we have
  nothing that tells them apart.  A batch whose kind holds
  no scored batch at all is also "same": no evidence, so no
  claim.
  """
  xs, ys = _scores(train, a), _scores(train, b)
  if not xs or not ys: return "same"
  if ezr.same(xs, ys, eps=NOISE): return "same"
  return "A" if sum(xs)/len(xs) > sum(ys)/len(ys) else "B"


def accuracy(train: list[list], test: list[list]) -> float:
  """Share of unseen batches whose guess is right on the dot.

  TEST is a list of lists like TRAIN -- its first row may be
  a header, which we skip.  A guess counts as right when it
  rounds to the batch's true Quality+.  Batches with no
  recorded score cannot be marked, so they are left out of
  both the hits and the total.  No markable batch scores 0.
  """
  at   = step2._at(train[0])
  hits = n = 0
  for row in test:
    if row and isinstance(row[0], str): continue   # a header
    if not step2._known(row, at): continue         # unmarkable
    n += 1
    hits += round(step2.guess(train, row)) == round(row[at])
  return hits / n if n else 0.0


def main() -> None:
  "Read -File: hold back half, then certify pairs and score."
  import random
  src   = list(ezr.csv(ezr.the.File))
  head, at = src[0], step2._at(src[0])
  rows  = random.sample(src[1:], len(src) - 1)
  n     = len(rows) // 2
  train = [head] + rows[:n]
  test  = rows[n:]
  print("# A vs B")
  print(f"{'A':>8} {'B':>8}   verdict")
  pool = [r for r in test if step2._known(r, at)]
  for a, b in zip(pool[:10], pool[10:20]):
    print(f"{a[at]:8.2f} {b[at]:8.2f}   "
          f"{certify(train, a, b)}")
  print(f"\n# accuracy\n{100*accuracy(train, test):.1f}% of "
        f"{len(pool)} held-back batches guessed exactly")


if __name__ == "__main__":
  ezr.run(main)
