#!/usr/bin/env python3 -B
"""
step6.py: two more answers for the winemaker, built on ezr and
step2.

  blame(train, k=3) -> the K lab readings that matter most for
                       Quality+, most important first
  dragging(train)   -> the readings sitting too high or too low
                       in the poor batches, worst first

Rows arrive as before: rows[0] is the header (strings) and the
rest are numbers; `+` marks what to maximize, `-` what to
minimize, everything else is a lab reading.  step2.goalAt finds
the panel's score (Quality+).

Neither answer may ever name a goal column, so we do not filter
names by hand.  We rewrite the header the way step2 does --
Quality+ the only goal, every other goal (Alcohol-) marked X and
so skipped -- and then the table's own x columns *are* the lab
readings, by construction.  The score is never a predictor, so
nothing here can leak it.

blame.  "Matters most" is asked of a model, not of a column, so
we grow a small forest of ezr trees (bootstrap rows, a random
two thirds of the readings per tree) and ask it two different
questions.  Two, because either one alone can mislead.

  gain: walking one tree, every split on a reading bought some
        drop in the spread of panel scores.  Summing those drops
        says how much the tree leaned on that reading.  Cheap and
        stable, but it only sees the rows the tree was built on.

  shuffle: take the rows a tree never saw, scramble one reading
        among them, and see how much worse the tree's guesses
        get.  That is the honest test -- out of sample -- but it
        is noisier, so we repeat it.

Each reading is scored only by the trees that were offered it
(otherwise a reading loses for not being dealt in), each view is
scaled against its own leader, and the two are averaged.  Where
readings are correlated they split the credit between them, so
read the result as "these three carry the signal", not "the
others are irrelevant"; `dragging` below looks at the readings
one at a time and is a useful second opinion.

dragging.  Here "poor" is the winemaker's own line: a batch
scoring under 6.  For every reading we compare the poor batches
against the rest, and keep only the readings where those two
groups are really distinguishable -- ezr.same, i.e. Cliff's
delta plus Kolmogorov-Smirnov, so a reading has to clear both
before we point at it.  The size of that same Cliff's delta
orders the list (worst first), and its sign says which way the
poor batches lean: above the good ones is 'high', below is
'low'.  Being rank based, it needs no units and no bell curve,
and a reading that merely wobbles never makes the list at all.

Speed.  The forest is the only slow part: six trees over 600
sampled rows each, grown at most once per train set and capped
by a wall clock, then a few tens of thousands of tree walks for
the shuffling.  `dragging` is a sort and two linear scans per
reading.  Both answers together run in a few seconds on 1279
rows, well inside the budget, and each is cached so a second
call costs nothing.
"""
import sys, time, random
from math import sqrt
import ezr, step2

POOR   = 6.0   # dragging: under this, the panel was unimpressed
MIN    = 12    # dragging: too few batches to compare
EPS    = 0     # same(): 0 means cliffs and ks, without cohen
TREES  = 6     # blame: trees in the forest
CUTS   = 8     # blame: splits to try, per reading
LEAF   = 16    # blame: fewest rows in a leaf
SAMPLE = 600   # blame: rows shown to any one tree
EVAL   = 400   # blame: unseen rows scored per tree
REPEAT = 2     # blame: shuffles per reading, per tree
BUDGET = 25.0  # blame: seconds to spend growing the forest

#-- the batches --------------------------------------------
def _num(v: ezr.ATOM) -> bool:
  "Is this cell a usable number?"
  return not isinstance(v, bool) and isinstance(v, (int, float))

def _setup(train: ezr.ROWS) -> ezr.o:
  "The single-goal table: Quality+ the only y, readings the xs."
  if not train or not train[0]:
    raise ValueError("step6: no train rows given")
  if not all(isinstance(s, str) for s in train[0]):
    raise ValueError("step6: train[0] must be the header")
  names = list(train[0])
  at    = step2.goalAt(names)
  data  = [r for r in train[1:] if at < len(r) and _num(r[at])]
  if not data: raise ValueError("step6: no scores in train")
  ys  = [float(r[at]) for r in data]
  tbl = ezr.Tbl([step2.xnames(names, at)] + data)
  if not tbl.x: raise ValueError("step6: no lab readings")
  return ezr.o(names=names, at=at, data=data, tbl=tbl,
               mu=sum(ys)/len(ys), trees=None, rank=None)

_CACHE: dict = {}

def _it(train: ezr.ROWS) -> ezr.o:
  "All we know about one train set, built once."
  key = (id(train), len(train), id(train[0]))
  if key not in _CACHE:
    _CACHE.clear()                 # one train set at a time
    _CACHE[key] = _setup(train)
  return _CACHE[key]

def _bad(c: ezr.o, row: ezr.ROW) -> bool:
  "Is this a poor batch?  Under POOR on the panel's own scale."
  y = float(row[c.at])
  return y > POOR if c.names[c.at][-1] == "-" else y < POOR

def _avg(vs: list) -> float:
  "Mean of what we have, 0 if we have nothing."
  return sum(vs)/len(vs) if vs else 0.0

#-- the forest ---------------------------------------------
def _forest(c: ezr.o) -> list:
  """Trees over the readings alone.  Each remembers the readings
  it was offered and the rows it never saw."""
  if c.trees is None:
    tbl  = c.tbl
    data = tbl.rows
    xs   = tbl.x
    k    = max(2, (2*len(xs) + 2)//3)
    n    = min(len(data), SAMPLE)
    old  = {key: ezr.the[key] for key in ("Leaf", "Cuts")}
    t0, out = time.time(), []
    try:
      ezr.the.Leaf, ezr.the.Cuts = LEAF, CUTS
      for _ in range(max(1, TREES)):
        idx   = [random.randrange(len(data)) for _ in range(n)]
        tbl.x = random.sample(xs, min(k, len(xs)))
        node  = step2.fit(ezr.tree(tbl, [data[i] for i in idx]),
                          c.at, c.mu)
        seen  = set(idx)
        out  += [ezr.o(node=node,
                       offered={col.at for col in tbl.x},
                       oob=[data[i] for i in range(len(data))
                            if i not in seen])]
        if time.time() - t0 > BUDGET: break
    finally:
      tbl.x = xs                # leave the table as we found it
      ezr.the.update(old)
    c.trees = out
  return c.trees

#-- view one: what the trees leaned on ---------------------
def _spread(at: int, rows: ezr.ROWS) -> tuple:
  "How many scores these rows hold, and how they vary."
  col = ezr.adds(r[at] for r in rows if _num(r[at]))
  return col.n, ezr.div(col)**2

def _gain(c: ezr.o, t: ezr.o) -> dict:
  "Per reading, the drop in score spread its splits bought."
  at, out = c.at, {}
  def walk(node: ezr.NODE) -> None:
    if not node.kids: return
    a, b = node.kids
    (np_, vp) = _spread(at, node.rows)
    (na, va), (nb, vb) = _spread(at, a.rows), _spread(at, b.rows)
    drop = np_*vp - (na*va + nb*vb)
    out[node.at] = out.get(node.at, 0.0) + max(0.0, drop)
    walk(a); walk(b)
  walk(t.node)
  n, v = _spread(at, t.node.rows)
  tot  = n*v or 1.0
  return {j: g/tot for j, g in out.items()}

#-- view two: what the trees cannot do without -------------
def _mse(at: int, node: ezr.NODE, rows: ezr.ROWS) -> float:
  "Mean squared error of one tree on these rows."
  n = e = 0
  for r in rows:
    if _num(r[at]):
      d = ezr.leaf(node, r).q - float(r[at])
      e += d*d
      n += 1
  return e/n if n else 0.0

def _shuffled(c: ezr.o, t: ezr.o, rows: ezr.ROWS) -> dict:
  """Per reading, how much worse this tree gets when we scramble
  that reading among rows the tree never saw."""
  at   = c.at
  base = _mse(at, t.node, rows)
  out  = {}
  for j in sorted(t.offered):
    vs, tot = [r[j] for r in rows], 0.0
    for _ in range(max(1, REPEAT)):
      random.shuffle(vs)
      fake = []
      for r, v in zip(rows, vs):
        r2 = list(r); r2[j] = v; fake += [r2]
      tot += _mse(at, t.node, fake)
    out[j] = max(0.0, tot/max(1, REPEAT) - base)
  return out

#-- blame --------------------------------------------------
def _corr(c: ezr.o, j: int) -> float:
  "Plain correlation of one reading with the score."
  xys = [(float(r[j]), float(r[c.at])) for r in c.data
         if _num(r[j]) and _num(r[c.at])]
  n = len(xys)
  if n < 3: return 0.0
  mx  = sum(x for x, _ in xys)/n
  my  = sum(y for _, y in xys)/n
  sxy = sum((x-mx)*(y-my) for x, y in xys)
  sxx = sum((x-mx)**2 for x, _ in xys)
  syy = sum((y-my)**2 for _, y in xys)
  return sxy/sqrt(sxx*syy) if sxx > 0 and syy > 0 else 0.0

def _rank(c: ezr.o) -> list:
  """Every reading with its combined score, best first.  Each
  view is scaled against its own leader before we average."""
  if c.rank is None:
    ts    = _forest(c)
    cands = sorted(col.at for col in c.tbl.x)
    gain  = {j: [] for j in cands}
    shuf  = {j: [] for j in cands}
    for t in ts:
      g    = _gain(c, t)
      rows = t.oob
      if len(rows) > EVAL: rows = random.sample(rows, EVAL)
      s = _shuffled(c, t, rows) if rows else {}
      for j in t.offered:    # only trees dealt it get to vote
        gain[j] += [g.get(j, 0.0)]
        shuf[j] += [s.get(j, 0.0)]
    gs = {j: _avg(gain[j]) for j in cands}
    ss = {j: _avg(shuf[j]) for j in cands}
    gm = max(gs.values(), default=0.0)
    sm = max(ss.values(), default=0.0)
    if gm <= 0 and sm <= 0:          # a flat forest: fall back
      out = [(abs(_corr(c, j)), j) for j in cands]
    else:
      out = [(((gs[j]/gm if gm > 0 else 0.0) +
               (ss[j]/sm if sm > 0 else 0.0)) / 2, j)
             for j in cands]
    out.sort(key=lambda z: (-z[0], c.names[z[1]]))
    c.rank = out
  return c.rank

def blame(train: ezr.ROWS, k: int = 3) -> list:
  """The K lab readings that matter most for whether a batch
  scores well, most important first.  Readings only: a goal
  column can never appear here."""
  k = int(k)
  if k <= 0: return []
  return [_it(train).names[j] for _, j in _rank(_it(train))[:k]]

#-- dragging -----------------------------------------------
def _cliffs(xs: ezr.NUMS, ys: ezr.NUMS) -> float:
  "Sorted in.  Cliff's delta: +1 all of XS above, -1 all below."
  gt = lt = j = k = 0
  for x in xs:
    while j < len(ys) and ys[j] <  x: j += 1; k = j
    while k < len(ys) and ys[k] <= x: k += 1
    gt += j
    lt += len(ys) - k
  return (gt - lt) / (len(xs) * len(ys))

def dragging(train: ezr.ROWS) -> list:
  """Which readings are dragging the poor batches down: for each,
  the name and whether it sits too 'high' or too 'low' in batches
  scoring under 6, worst first.  Only readings whose poor and
  good batches are really distinguishable make the list."""
  c = _it(train)
  poor, good = [], []
  for r in c.data: (poor if _bad(c, r) else good).append(r)
  if len(poor) < MIN or len(good) < MIN: return []
  out = []
  for col in c.tbl.x:
    j  = col.at
    xs = sorted(float(r[j]) for r in poor if _num(r[j]))
    ys = sorted(float(r[j]) for r in good if _num(r[j]))
    if len(xs) < MIN or len(ys) < MIN: continue
    if ezr.same(xs, ys, EPS): continue      # nothing to see
    d = _cliffs(xs, ys)
    if d: out += [(abs(d), c.names[j], "high" if d > 0 else "low")]
  out.sort(key=lambda z: (-z[0], z[1]))
  return [(name, side) for _, name, side in out]

#-- demo ---------------------------------------------------
def main(args: list[str]) -> None:
  "step6.py [file.csv] -- what drags the poor batches down."
  rows = list(ezr.csv(args[0] if args else ezr.the.File))
  random.seed(ezr.the.Seed)
  data = rows[1:]
  random.shuffle(data)
  n     = max(1, len(data)//5)
  train = [rows[0]] + data[n:]
  c     = _it(train)
  t0    = time.time()
  top   = blame(train, 3)
  t1    = time.time()
  drag  = dragging(train)
  t2    = time.time()
  print(f"# blame, from {len(train)-1} batches"
        f"   in {t1-t0:.1f}s")
  for i, name in enumerate(top, 1): print(f"  {i}. {name}")
  print("# every reading, scaled to the leader  (r = plain"
        " correlation)")
  ranked = _rank(c)
  big    = ranked[0][0] or 1.0
  for s, j in ranked:
    print(f"  {c.names[j]:>16} {ezr.say(s/big,2):>5}"
          f"   r {ezr.say(_corr(c, j),2)}")
  poor = sum(1 for r in c.data if _bad(c, r))
  print(f"# dragging, in {poor} poor batches of {len(c.data)}"
        f"   in {t2-t1:.1f}s")
  if not drag: print("  (nothing stands out)")
  for name, side in drag:
    print(f"  {name:>16} too {side}")

if __name__ == "__main__": main(sys.argv[1:])
