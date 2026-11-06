# Architecture and design decisions

This file records where the implementation departs from, or adds to, the
original specification (`docs/PROJECT_SPEC.md`), and why.

## Data flow

```
phy/receiver.py ──Monte Carlo──▶ phy/tables/per_{preamble,tracking}.npz
                                         │  PerTable (bilinear lookup)
channel/ (Hata, LEO pass, TR 38.811) ────┤
                                         ▼
netsim/simulator.py  event loop: readings → TX start (SNR, collisions) → TX end (PER draw)
   │  NetworkServer: delivered uplinks, frame-counter gaps, noise floor
   │  RollingZScoreDetector (tick 60 s, 5-min blocks)
   ▼
policies/  Controller.attach / on_uplink (class-A) / on_tick (triggers) / choose (oracle)
   fixed · ADR · ML-HGB · Oracle · Triggered-Rules · LLM-Agent (+ ablations)
   │
experiments/ e1…e6 ──▶ results/*.csv ──▶ results/*.png  (--plot-only regenerates figures)
                     └▶ results/traces/e4|e6/*.jsonl   (every agent decision, full trace)
```

## Decisions that differ from the specification

| Topic | Specification | Implementation | Reason |
|---|---|---|---|
| Doppler in the PHY | `exp(j2π(f_D t + ½ f_D' t²))` over the packet, residual + rate | Drift over the *whole packet* relative to the preamble estimate, two receiver models | The first version restarted the drift at every symbol, which hides the effect. |
| PER | `1 − (1 − SER)^n` | Packet-level Monte Carlo | Drift concentrates errors at the end of the packet; independence does not hold. |
| `set_tx_params` | `(device_ids, sf, tx_power_dbm, channel, link)` | `(target, sf?, tx_power_dbm?, channels?, link?, duration_s?)` | Group targets (a 3B model cannot list hundreds of ids); channel mask instead of one channel (devices hop); lease so that overrides expire. |
| Control plane | not specified | Class A (+ satellite broadcast during passes) | Without it, any controller reconfigures devices instantly, which overstates every adaptive policy. |
| Triggers | 4 triggers | Same 4 + 10-min cooldown + "satellite sets in < 2 min" warning | Without cooldown a 30-min event calls the agent every minute; the warning lets commands be delivered before the pass ends. |
| Extra baseline | — | `Triggered-Rules`: same triggers, telemetry and override mechanism, hand-written rules | Separates the value of the LLM's reasoning from the value of the event-triggered architecture. |
| Oracle | lowest-energy (SF, power) with PER ≤ 1 % | Also chooses link and channel; waits when no gateway can receive; ignores collisions | Upper bound at link level. Not an upper bound on network PDR under heavy load (it puts everyone at the lowest SF). |
| ML classifier | trained on oracle labels | Trained on oracle labels for the device's *next* reading, from simulations on 8 training seeds disjoint from the evaluation seeds, behaviour policy ADR + 30 % exploration; applied through class-A downlinks | Fair information (anomaly flag included, pitfall 3) and no train/test leakage. |
| Detector | rolling z-score on noise floor, SNR, PER | Same statistics on 5-min blocks, binomial test for loss, Poisson test for load, 2-sample persistence, rising-edge alerts | The first version produced hundreds of false alerts per 3-h run. |
| LLM backend | Ollama | Ollama **or** llama.cpp (GGUF), with a response cache | Same code in a sandbox without Ollama; cached answers make every figure reproducible without the model. |

## How to add a model to the study

```bash
ollama pull qwen2.5:7b
python experiments/e4_anomalies.py --policies LLM-Agent --backend ollama --model qwen2.5:7b
python experiments/e6_ablations.py --backend ollama --model qwen2.5:7b
python experiments/e5_agent_cost.py
python experiments/summarize.py
```
