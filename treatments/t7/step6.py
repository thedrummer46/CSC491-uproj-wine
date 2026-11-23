import sys
from ezr import clone, same, csv
from step2 import table, grow

def splits(node):
	"Every node of the tree that splits (has kids), top down"
	if node.kids:
		yield node
		for kid in node.kids:
			yield from splits(kid)

def readings(tbl):
	"The lab-reading columns, minus Alcohol: a goal in the real header, so never blamed"
	return [c for c in tbl.x if c.txt != "Alcohol"]

def blame(train, k=3):
	"Ask 10: the K readings the guess tree leans on most"
	tbl, node = grow(train)
	ok = {c.at for c in readings(tbl)}
	weight = {}
	for n in splits(node):
		if n.at in ok:
			name = tbl.names[n.at]
			weight[name] = weight.get(name, 0) + len(n.rows)
	return sorted(weight, key=weight.get, reverse=True)[:k]

def dragging(train):
	"Ask 9: readings really different in poor batches (Quality+ below 6), biggest gap first"
	tbl = table(train)
	at = tbl.names.index("Quality+")
	poor = clone(tbl, [r for r in tbl.rows if r[at] < 6])
	good = clone(tbl, [r for r in tbl.rows if r[at] >= 6])
	out = []
	for c in readings(tbl):
		if not same([r[c.at] for r in poor.rows], [r[c.at] for r in good.rows]):
			p, g = poor.cols[c.at].mu, good.cols[c.at].mu
			out.append((abs(p - g) / c.sd, c.txt, "high" if p > g else "low"))
	return [(name, way) for gap, name, way in sorted(out, reverse=True)]

def main(file):
	"Demo: the 3 readings that matter most, and what's off in poor batches"
	rows = list(csv(file))
	print("Matter most for Quality:", ", ".join(blame(rows)))
	print("In poor batches:")
	for name, way in dragging(rows):
		print(f"  {name} is too {way}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")
