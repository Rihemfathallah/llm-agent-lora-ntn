#!/usr/bin/env bash
# Agent runs done in the development sandbox (no Ollama): Qwen3-0.6B Q4_K_M
# through llama.cpp.  On a laptop with Ollama use run_all.sh instead.
set -euo pipefail
cd "$(dirname "$0")"
GGUF="${GGUF:-/home/claude/llm/Qwen3-0.6B-Q4_K_M.gguf}"
ARGS=(--backend llamacpp --model qwen3-0.6b-q4km --model-path "$GGUF")
python3 e4_anomalies.py --policies LLM-Agent "${ARGS[@]}"
python3 e6_ablations.py "${ARGS[@]}"
python3 e3_scalability.py --policies LLM-Agent --skip-aloha "${ARGS[@]}"
python3 e5_agent_cost.py
