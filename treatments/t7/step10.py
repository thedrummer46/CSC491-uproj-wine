import sys, random
from statistics import median
from ezr import Tbl, ydist, same, csv
from step1 import best, summary
from step2 import guess
from step5 import choose

REPEATS = 20   # runs of each side in vs_random

def compare(ours, theirs):
	"Lower is better on both sides: 'same' if ezr can't tell them apart, else which median wins"
	if same(ours, theirs): return 'same'
	return 'better' if median(ours) < median(theirs) else 'worse'

def vs_random(rows, label, budget=20):
	"Ask 19: our choose() picks vs the best of BUDGET random tastings, REPEATS times each"
	ours = [choose(rows, label, budget) for _ in range(REPEATS)]
	luck = [best(Tbl([rows[0]] + [label(r) for r in random.sample(rows[1:], budget)]))[0]
	        for _ in range(REPEATS)]
	tbl = Tbl([rows[0]] + ours + luck)
	return compare([ydist(tbl, r) for r in ours], [ydist(tbl, r) for r in luck])

def vs_baseline(train, test):
	"Ask 20: guess()'s per-batch error on TEST vs always predicting TRAIN's mean Quality+"
	at = train[0].index("Quality+")
	typical = summary(train)["Quality+"]
	ours = [abs(guess(train, r) - r[at]) for r in test[1:]]
	base = [abs(typical - r[at]) for r in test[1:]]
	return compare(ours, base)

def main(file):
	"Demo: both races on an 80/20 split"
	random.seed(1); rows = list(csv(file)); names = rows[0]
	cut = int(0.8 * len(rows)); train, test = rows[:cut], [names] + rows[cut:]
	goals = [at for at, n in enumerate(names) if n[-1] in "+-"]
	hidden = [["?" if at in goals else v for at, v in enumerate(r)] for r in train[1:]]
	real = {id(h): r for h, r in zip(hidden, train[1:])}
	def label(h):
		return real[id(h)]
	print("vs random tasting:", vs_random([names] + hidden, label))
	print("vs the mean:      ", vs_baseline(train, test))

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")