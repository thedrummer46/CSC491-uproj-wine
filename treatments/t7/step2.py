import sys
from ezr import Tbl, tree, leaf, leafs, the, csv

TREES = {}

def table(train):
	"Train rows as a table predicting Quality+"
	names = ["Alcohol" if n == "Alcohol-" else n for n in train[0]]
	return Tbl([names] + train[1:])

def grow(train, size=4):
	"The tree for TRAIN, built once and reused from TREES"
	key = (id(train), size)
	if key not in TREES:
		tbl = table(train)
		old = the.Leaf
		the.Leaf = size
		try:
			TREES[key] = (train, tbl, tree(tbl, tbl.rows))
		finally:
			the.Leaf = old
	_, tbl, node = TREES[key]
	return tbl, node

def qualities(tbl, rows):
	"Quality+ of each row"
	at = tbl.names.index("Quality+")
	return [row[at] for row in rows]

def quality(tbl, rows):
	"Mean Quality+ of rows"
	qs = qualities(tbl, rows)
	return sum(qs) / len(qs)

def guess(train, row):
	"Ask 3: walk ROW down TRAIN's tree, predict its leaf's mean Quality+"
	tbl, node = grow(train)
	return quality(tbl, leaf(node, row).rows)


def kinds(train):
	"Ask 4: a few groups of similar batches with their expected Quality+"
	tbl, node = grow(train, (len(train) - 1) // 8)
	groups = [(z.rows, quality(tbl, z.rows)) for z in leafs(node)]
	return sorted(groups, key=lambda g: g[1], reverse=True)

def main(file):
	"Demo: show the kinds, and guess 5 batches the tree never saw"
	rows = list(csv(file))
	train, test = rows[:-5], rows[-5:]
	print("Kinds of batch (size, expected Quality+):")
	for group, q in kinds(train):
		print(f"  {len(group):>4}  {q:.2f}")
	at = rows[0].index("Quality+")
	print("\nGuesses for 5 unseen batches:")
	for row in test:
		print(f"  predicted {guess(train, row):.2f}, actual {row[at]}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")