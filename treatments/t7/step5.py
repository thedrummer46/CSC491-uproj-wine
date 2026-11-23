import sys, random
from ezr import Tbl, acquire, ranker, the, csv
from step1 import best

START = 8
CHECK = 5

def taste(rows, label, budget=20):
	"Ask 13: which batches to taste, in order: acquire's picks, then the tree's top guesses"
	tbl = Tbl(rows)
	seen = set()
	def once(row):
		seen.add(id(row))
		return label(row)
	old = the.Start
	the.Start = min(START, budget - CHECK)
	try:
		tasted = acquire(tbl, budget - CHECK, once)
	finally:
		the.Start = old
	rank = ranker(Tbl([rows[0]] + tasted))
	rest = [r for r in tbl.rows if id(r) not in seen]
	return tasted + [label(r) for r in sorted(rest, key=rank)[:CHECK]]

def choose(rows, label, budget=20):
	"Ask 14: the best batch tasted, high Quality+ and low Alcohol-"
	tasted = taste(rows, label, budget)
	return best(Tbl([rows[0]] + tasted))[0]

def main(file):
	"Demo: taste 20 batches, compare the pick with the best in the whole cellar"
	random.seed(1)
	rows = list(csv(file))
	names = rows[0]
	goals = [at for at, n in enumerate(names) if n[-1] in "+-"]
	hidden = [["?" if at in goals else v for at, v in enumerate(r)] for r in rows[1:]]
	real = {id(h): r for h, r in zip(hidden, rows[1:])}
	def label(h):
		return real[id(h)]
	pick = choose([names] + hidden, label)
	top = best(Tbl(rows))[0]
	print(f"Tasted 20 of {len(hidden)} batches.")
	print(f"  Pick:          Alcohol {pick[-2]}, Quality {pick[-1]}")
	print(f"  Cellar's best: Alcohol {top[-2]}, Quality {top[-1]}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")