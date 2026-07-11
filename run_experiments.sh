#!/usr/bin/env bash
# One-command launcher: sets up the environment, runs all experiments,
# then generates the figures. Works on a local machine (creates a venv)
# and on Google Colab (installs directly, no venv).
set -e
cd "$(dirname "$0")"

if [ -n "$COLAB_RELEASE_TAG" ] || [ -d "/content" ] || [ "$1" = "--no-venv" ]; then
    PY=python3
    $PY -m pip install -q -r requirements.txt
else
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment (.venv)..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    PY=python
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi

$PY src/run_experiments.py
$PY src/make_figures.py

echo ""
echo "Done. Metrics: results/experiments.json + results/test_results.json"
echo "Figures: results/figures/"
