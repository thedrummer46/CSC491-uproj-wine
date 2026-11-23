import sys, heapq
from ezr import addRow, xdist, csv
from step2 import table, quality

K = 10

def start(names):
	"An empty model: a table that knows only the header (Alcohol as a reading)"
	return table([names])

def learn(model, row):
	"Add one scored batch; the column stats update in place, nothing is refit"
	addRow(model, row)
	return model

def near(model, row, k=K):
	"The K batches seen so far that are most like ROW"
	return heapq.nsmallest(k, model.rows, key=lambda r: xdist(model, row, r))

def predict(model, row):
	"Ask 8: mean Quality+ of ROW's nearest neighbours so far"
	return quality(model, near(model, row))

def worth(model, row):
	"Ask 7: send to the panel if predicted Quality+ is 6 or more"
	return predict(model, row) >= 6

def main(file):
	"Demo: stream batches in, show the guesses improving"
	rows = list(csv(file))
	cut = int(0.8 * len(rows))
	train, test = rows[1:cut], rows[cut:]
	at = rows[0].index("Quality+")
	model = start(rows[0])
	for i, row in enumerate(train, 1):
		learn(model, row)
		if i % 300 == 0:
			err = sum(abs(predict(model, r) - r[at]) for r in test) / len(test)
			print(f"{i:>5} batches seen: average miss {err:.2f} points")
	for row in test[:5]:
		print(f"  Quality {row[at]}: send to panel? {worth(model, row)}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")