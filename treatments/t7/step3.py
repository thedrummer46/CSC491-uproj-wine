import sys, random
from statistics import mean
from ezr import leaf, same, csv
from step2 import grow, qualities, guess

SIZE = 32

def certify(train, a, b):
	"Ask 5: which batch will score better, or 'same' if the gap is just noise"
	tbl, node = grow(train, SIZE)
	la, lb = leaf(node, a), leaf(node, b)
	qa, qb = qualities(tbl, la.rows), qualities(tbl, lb.rows)
	if la is lb  or  same(qa, qb):
		return "same"
	return "A" if mean(qa) > mean(qb) else "B"

def accuracy(train, test):
	"Ask 6: share of TEST batches whose guessed score rounds to the true Quality+"
	at = test[0].index("Quality+")
	hits = [round(guess(train, row)) == row[at] for row in test[1:]]
	return mean(hits)

def main(file):
	"Demo: certify a few random pairs of unseen batches, then report accuracy"
	random.seed(1)
	rows = list(csv(file)) 
	cut = int(0.8 * len(rows))
	train, test = rows[:cut], [rows[0]] + rows[cut:]
	at = test[0].index("Quality+")
	for _ in range(5):
		a, b = random.sample(test[1:], 2)
		print(f"Batch A: {a[at]}, Batch B: {b[at]}, Certify: {certify(train, a, b)}")
	print(f"Accuracy: {accuracy(train, test):.2%}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")