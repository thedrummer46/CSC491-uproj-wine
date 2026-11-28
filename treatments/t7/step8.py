import sys
from statistics import median
from ezr import csv
from step2 import grow, guess, quality
from step3 import SIZE
from step6 import readings

def paths(node, path=()):
	"Every leaf, with the (split node, branch) pairs on the way to it"
	if not node.kids:
		yield (node, path)
	else:
		for branch, kid in enumerate(node.kids):
			yield from paths(kid, path + ((node, branch),))

def fix(train, row, only=None):
	"Asks 15-16: the cheapest move into a leaf predicted Quality+ 6 or more"
	tbl, node = grow(train, SIZE)
	ok = {c.at for c in readings(tbl) if only is None or c.txt in only}
	best, least = {}, 1e30
	for (lf, path) in paths(node):
		if quality(tbl, lf.rows) < 6: continue
		new = {}
		for (n, branch) in path:
			if n.go(row) != (branch == 0):
				new[n.at] = median(r[n.at] for r in lf.rows)
		if not new.keys() <= ok: continue
		cost = sum(abs(v - row[at]) / tbl.cols[at].sd for at, v in new.items())
		if cost < least:
			best, least = new, cost
	return {tbl.names[at]: v for at, v in best.items()}

def main(file):
	"Demo: fix 3 poor batches, freely and with sulphates and acids only"
	rows = list(csv(file)); train = rows
	poor = [r for r in train[1:] if guess(train, r) < 6][:3]
	acids = ["Sulphates", "Fixedacidity", "Volatileacidity", "Citricacid"]
	for row in poor:
		for only in (None, acids):
			change = fix(train, row, only)
			moved = list(row)
			for name, v in change.items():
				moved[train[0].index(name)] = v
			print(f"{change}: {guess(train, row):.1f} -> {guess(train, moved):.1f}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")