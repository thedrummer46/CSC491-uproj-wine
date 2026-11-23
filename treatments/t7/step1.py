import sys
from ezr import Tbl, ydist, mids, csv, say

PICK = 0.05

def best(tbl, rows=None):
	"Return rows sorted by distance from heaven"
	return sorted(rows or tbl.rows, key=lambda row: ydist(tbl, row))

def remember(rows):
	"Ask 1: the exceptional batches, top 5%"
	data = best(Tbl(rows))
	return data[:int(len(data) * PICK)]

def summary(rows):
	"Ask 2: {column name: typical value} for every column"
	data = Tbl(rows)
	return {data.names[at]: v for at, v in mids(data).items()}

def main(file):
	"Demo: print the exceptional batches and a normal batch"
	rows = list(csv(file))
	print("Exceptional batches (Alcohol-, Quality+):")
	for row in remember(rows):
		print(" ", row[-2], row[-1])
	print("\nA normal batch:")
	for name, val in summary(rows).items():
		print(f" {name:>16} {say(val)}")

if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "data/Wine_quality.csv")