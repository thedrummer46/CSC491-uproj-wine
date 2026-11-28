import sys, random
from statistics import quantiles
from ezr import xdist, csv
from step2 import grow
from step4 import near

MARGIN = 0.25   # how far past TRAIN's range a reading may go before it looks wrong
SAMPLE = 100    # train batches used to set the novelty bar
LIMITS = {}

def limits(train):
	"Built once per TRAIN: the table, each reading's allowed (lo, hi), and the novelty bar"
	key = id(train)
	if key not in LIMITS:
		tbl, _ = grow(train)
		box = {}
		for c in tbl.x:
			vs = [r[c.at] for r in tbl.rows]
			lo, hi = min(vs), max(vs)
			pad = MARGIN * (hi - lo)
			box[c.at] = (lo - pad, hi + pad)
		some = random.Random(1).sample(tbl.rows, min(SAMPLE, len(tbl.rows)))
		gaps = [xdist(tbl, r, near(tbl, r, 2)[1]) for r in some]
		LIMITS[key] = (tbl, box, quantiles(gaps, n=20)[-1])
	return LIMITS[key]

def odd(train, row):
	"Ask 17: any reading negative, or well outside anything TRAIN has seen"
	_, box, _ = limits(train)
	return any(row[at] < 0 or not lo <= row[at] <= hi for at, (lo, hi) in box.items())

def novel(train, row):
	"Ask 18: farther from its nearest TRAIN batch than 95% of TRAIN's own batches are"
	tbl, _, bar = limits(train)
	return xdist(tbl, row, near(tbl, row, 1)[0]) > bar

def main(file):
	"Demo: check 20 unseen batches, then a copy of one with a typo"
	rows = list(csv(file)); train, test = rows[:-20], rows[-20:]
	print("odd:", sum(odd(train, r) for r in test), "of", len(test))
	print("novel:", sum(novel(train, r) for r in test), "of", len(test))
	typo = list(test[0]); typo[0] *= 10
	print(f"{train[0][0]} {test[0][0]:g} -> {typo[0]:g}: odd before {odd(train, test[0])}, after {odd(train, typo)}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")