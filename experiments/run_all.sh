#!/usr/bin/env bash
# Reproduce every result of the repository, in order.
# Usage:  bash experiments/run_all.sh [ollama-model]      (default qwen2.5:3b)
# The PER tables (phy/tables/*.npz) are committed; regenerate them with
#   python -m phy.ser_montecarlo --receiver tracking && python -m phy.ser_montecarlo --receiver preamble
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen2.5:3b}"

python -m pytest -q
python experiments/e1_phy_validation.py
python experiments/e2_satellite_pass.py
python experiments/e3_scalability.py                        # baselines + ALOHA check
python experiments/e4_anomalies.py                          # baselines (incl. ML training)
python experiments/e4_anomalies.py --policies LLM-Agent --backend ollama --model "$MODEL"
python experiments/e6_ablations.py --backend ollama --model "$MODEL"
python experiments/e3_scalability.py --policies LLM-Agent --backend ollama --model "$MODEL"
python experiments/e5_agent_cost.py
python experiments/summarize.py
