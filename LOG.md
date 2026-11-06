# Project Execution Log: llm-lora-ntn-agent

## Project Milestones

- **Phase 1: PHY CSS Layer**
  - Implemented `phy/css.py`: CSS chirp modulation, de-chirping, FFT symbol demodulation, Doppler quadratic phase injection, Semtech AN1200.13 ToA formula.
  - Implemented `phy/ser_montecarlo.py`: AWGN and Doppler Monte Carlo simulation, lookup table generation.
  - Validated 0 demodulation errors in noise-free conditions and key Doppler insight (SF12 drift/bin = 11.88).

- **Phase 2: Channel Modeling & Link Budget**
  - Implemented `channel/satellite.py`: 2D Keplerian orbit, slant range, FSPL, Doppler shift/rate, 3GPP TR 38.811 shadow fading & clutter loss.
  - Implemented `channel/link_budget.py` and `channel/terrestrial.py`: Okumura-Hata urban and rural models, unified link budget calculation, feasible SF evaluation.

- **Phase 3: Network Simulator**
  - Implemented `netsim/simulator.py`: Pure ALOHA event loop with `heapq`, 6 dB co-channel capture effect, SX1276 datasheet energy model, EU868 1% duty cycle.
  - Implemented `netsim/devices.py`: Device spatial distribution model.
  - Sanity check: ALOHA throughput matches theoretical limit ($S = 0.1836$ at $G = 0.47$).

- **Phase 4: Baselines**
  - Implemented `policies/fixed.py`: Fixed-SF12 Max Power and Fixed-SF7 policies.
  - Implemented `policies/adr.py`: Semtech recommended ADR algorithm (20-uplink history, installation margin, step-down/step-up).
  - Implemented `policies/oracle.py` & `policies/ml_classifier.py`: Prescient upper-bound and gradient boosting baseline.

- **Phase 5: Anomaly Injection & Detection**
  - Implemented `netsim/anomalies.py`: AnomalySchedule (jamming, deep fade, satellite pass end, gateway outage, traffic surge) and RollingZScoreDetector.

- **Phase 6: LLM Agent & Tool Registry**
  - Implemented `policies/llm_agent/tools.py`: 5 simulator tools (`get_link_stats`, `get_satellite_status`, `get_anomaly_report`, `get_network_load`, `set_tx_params`).
  - Implemented `policies/llm_agent/prompt.py`: Domain-specific prompt with LoRa facts and EU868 constraints.
  - Implemented `policies/llm_agent/agent.py`: TriggerChecker with 4 triggers and Ollama controller with fallback to ADR.

- **Experiments E1 to E6**
  - **E1 (`experiments/e1_phy_validation.py`)**: Produced `results/e1a_ser_vs_snr_awgn.png/csv` and `results/e1b_ser_vs_doppler.png/csv`.
  - **E2 (`experiments/e2_satellite_pass.py`)**: Produced `results/e2_satellite_pass.png/csv`.
  - **E3 (`experiments/e3_scalability.py`)**: Produced `results/e3_scalability.png/csv`.
  - **E4 (`experiments/e4_anomalies.py`)**: Produced `results/e4_anomalies.png/csv`.
  - **E5 (`experiments/e5_agent_cost.py`)**: Produced `results/e5_agent_cost.png/csv`.
  - **E6 (`experiments/e6_ablations.py`)**: Produced `results/e6_ablations.png/csv`.

---

## 2026-11-06 — Review, corrections, full re-run

Every number below is copied from `results/summary.md` (generated from the CSV files).

### Bugs found in the 2026-11-03/05 version, and fixed

| # | Problem | Consequence | Fix |
|---|---|---|---|
| 1 | AWGN noise std `10^(−SNR/20)` per component | complex noise power ×2: every SER curve shifted by 3 dB | `noise_std_per_component` = sqrt(10^(−SNR/10)/2); Monte Carlo now matches the exact noncoherent formula |
| 2 | Doppler drift restarted at every symbol | the drift over the packet (the real effect) was missing; E1b showed no error below 800 Hz/s | packet-level model, drift relative to the preamble estimate, receivers A (preamble) and B (tracking) |
| 3 | TR 38.811 values "representative" (invented) | clutter 3.8 dB at 10°, 0 at 90° | real tables 6.6.1-1 / 6.6.2-3 (LOS prob., σ_SF LOS/NLOS, NLOS CL 16–19.5 dB), LOS/NLOS draw per packet |
| 4 | SX1276 current table did not match the datasheet | energy wrong by up to ×2.7 | datasheet RFO points 7 dBm/20 mA, 13 dBm/29 mA |
| 5 | `terrestrial_path_loss` passed the distance as the gateway height | wrong terrestrial SNR | argument order fixed + test |
| 6 | `OraclePolicy()` / `MLClassifierPolicy()` crashed (`super().__init__("…")`), `time_on_air` called with swapped arguments | oracle and ML never ran | rewritten |
| 7 | Agent tools registered as `_get_link_stats`… | every LLM tool call would have been "unknown tool" | explicit JSON schemas with the spec names |
| 8 | Detector recall mixed alert counts and event counts; per-channel loss from frame-counter gaps | hundreds of false alerts per 3-h run in a smoke test | 5-min blocks, binomial/Poisson tests, persistence 2, rising-edge alerts |
| 9 | Duty cycle per channel | 3 % effective duty cycle (channels share sub-band g1) | per device on g1 |
| 10 | Satellite link not in the network simulator; same SNR for every device | ADR = fixed SF7 in E3; no satellite scenario possible | per-device Hata + shadowing, LEO passes, class-A control plane |
| 11 | E4–E6 were scripted/mocked | no measurement of the agent | real LLM runs, traces saved |

Found and fixed during this session (before the final runs): traffic surge as a faster periodic
timer synchronised every device at the end of the surge (now extra Poisson readings); gateway
silence counted uplinks received by the satellite; ADR recorded the SNR of overridden uplinks as if
measured at its own power; single-row HistGradientBoosting predictions two orders of magnitude slower with OpenMP
thread start-up (now 1 thread).

### What was run

```
python -m phy.ser_montecarlo --receiver tracking      # ~45 min, 400 packets/point
python -m phy.ser_montecarlo --receiver preamble
python experiments/e1_phy_validation.py
python experiments/e2_satellite_pass.py
python experiments/e3_scalability.py
python experiments/e4_anomalies.py
# agent (E4, E6, E3): Qwen3-0.6B Q4_K_M via llama.cpp, 2 CPU threads
ARGS="--backend llamacpp --model qwen3-0.6b-q4km --model-path Qwen3-0.6B-Q4_K_M.gguf"
python experiments/e4_anomalies.py --policies LLM-Agent $ARGS
python experiments/e6_ablations.py $ARGS
python experiments/e3_scalability.py --policies LLM-Agent --skip-aloha $ARGS
python experiments/e5_agent_cost.py && python experiments/summarize.py
```

Ollama and Hugging Face were unreachable from the machine used for these runs. The model is Qwen3-0.6B Q4_K_M
(unsloth GGUF, sha256 ac2d9771…524a, 396 705 472 bytes, checked against the published provenance
file). On a laptop: `ollama pull qwen3:0.6b` gives the same model family; the reference model of the
protocol is `qwen2.5:3b`.

### What was observed

- E1: MC vs exact SER within 0.02 (absolute); ideal receiver reaches PER 1 % within −1.4…+0.3 dB of
  the Semtech thresholds. Receiver A loses SF12 from 50 Hz/s, SF11 from 150, SF10 from 200;
  receiver B loses SF12 at 300 Hz/s, other SFs fine up to 1000 Hz/s.
- E2: 600 km zenith pass 508 s above 10°, max Doppler 19.7 kHz, max rate 252.3 Hz/s.
  Receiver B: SF12 feasible at zenith from 650 km (Doroshkin et al. measured ~550 km).
  Receiver A: SF12 usable on 39 % of the pass (100 % if Doppler ignored).
- E3: ALOHA check error ≤ 0.0026. ADR 97.5 % → 95.4 % PDR from N = 100 to 1000; oracle 98.1 → 89.4 %
  (lowest-energy choice puts everyone on SF7); ML-HGB 76.3 / 75.4 %.
- E4: Triggered-Rules vs ADR: +10.5 (S1), 0.0 (S2), +15.0 (S3), +12.9 (S4), +0.5 (S5) points, 5/5 seeds
  in S1/S3/S4. Agent (0.6B): 0.0 in S1–S3, −0.1 in S4, −17.5 ± 23.9 in S5 (2/5 seeds: every device
  to SF12 at the surge onset).
- E5: 287 decisions in 25 runs, median 5.78 s, p95 14.67 s, 7.65 LLM calls per simulated hour,
  6 changes applied, 5/11 set calls rejected (all "target group is empty").
- E3 with the agent: equal to ADR up to N = 200; below ADR in 7 of 30 runs at higher load (worst
  −15.5 points, N = 800), each time after a false alert coinciding with a periodic review.
  Over E3 + E4, 15 of the 16 applied changes were the same action (all devices → SF12, 10 dBm).
- E6: without the anomaly tool or without the LoRa facts, 0 set_tx_params calls in 271 decisions
  each → identical to ADR everywhere (S5: 94.1 % vs 76.6 % for the full agent).
- Reproducibility: all 105 agent runs (E4 25, E6 50, E3 30) replayed from `results/llm_cache/`
  with `--replay-only` give identical PDR and decision counts.

### What failed or surprised

- The ML classifier trained on oracle labels under-provisions (it imitates an oracle that knows each
  packet's fading). Kept as is: it is the specified baseline; the fix is a different label.
- The oracle is not an upper bound under heavy load (S5, E3 at N = 1000).
- A predicted-load guardrail (reject actions with offered load G > 0.5) was tried after seeing the
  S5 failure and dropped: at the surge onset the higher rate is not observed yet, so it would not
  have blocked the harmful action.

### Next

1. Run E4 and E6 with `qwen2.5:3b` (and `7b`) through Ollama on the laptop.
2. Consequence-aware safety layer (rollback watchdog, staged rollout, twin simulation).
3. Store-and-forward buffering; several passes per day with Earth rotation.
