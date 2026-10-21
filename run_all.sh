#!/usr/bin/env bash
# Reproduces every number and figure in the Phase 1 and Phase 2 results.
set -e
cd "$(dirname "$0")"
mkdir -p figs
python3 make_architecture.py   # Figure 1
python3 run_phase1.py          # Figures 2 and 5, results_phase1.json
python3 run_validation.py      # Figures 3 and 4, results_validation.json
python3 run_phase2.py          # Figures 6-8, results_phase2.json (about 1.5 minutes)
python3 check_results.py       # compares the output with the values printed in the papers
