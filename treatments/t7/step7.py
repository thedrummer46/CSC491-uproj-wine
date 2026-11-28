import sys
from ezr import csv
from step2 import grow, guess

def verdict(train, row):
	"Ask 11: 'good' if ROW's predicted Quality+ is 6 or more, else 'poor'"
	return 'good' if guess(train, row) >= 6 else 'poor'

def why(train, row):
	"Ask 12: the tree's tests on ROW's path to its leaf, tightest per reading"
	tbl, node = grow(train)
	tight = {}
	while node.kids:
		yes = node.go(row)
		op = '<=' if yes else '>'
		key = (tbl.names[node.at], op)
		pos = tight.get(key, node.v)
		tight[key] = min(node.v, pos) if op == '<=' else max(node.v, pos)
		node = node.kids[0 if yes else 1]
	return [(name, op, v) for (name, op), v in tight.items()]

def explain(train, row):
	"Asks 11-12: the verdict and its reason in plain English"
	v = verdict(train, row)
	q = guess(train, row)
	words = {'<=': 'at most', '>': 'above'}
	reasons = [f"{name} {words[op]} {bound:g}" for name, op, bound in why(train, row)[:3]]
	return f"Predicted {v} (Quality about {q:.1f}) because " + " and ".join(reasons)

def main(file):
	"Demo: explain 3 unseen batches"
	rows = list(csv(file)); train, test = rows[:-3], rows[-3:]
	for row in test:
		print(explain(train, row))

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")