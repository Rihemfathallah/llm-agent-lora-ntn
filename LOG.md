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
