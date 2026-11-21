#!/usr/bin/env bash
# Lines of new code per step, per treatment: the data for the Results histogram.
# Usage: scripts/newcode.sh            -> CSV to stdout: treatment,step,added
# Needs tags step1, step2, ... (compares stepN-1..stepN; step1 is compared to the root commit).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

tags=$(git tag -l 'step*' | sort -V)
root=$(git rev-list --max-parents=0 HEAD | tail -1)

echo "treatment,step,added"
for t in treatments/t*/; do
  t=${t%/}
  prev=$root
  for tag in $tags; do
    added=$(git diff --numstat "$prev" "$tag" -- "$t" ':!*.md' | awk '{s+=$1} END {print s+0}')
    echo "$(basename "$t"),$tag,$added"
    prev=$tag
  done
done
