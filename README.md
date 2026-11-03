# llm-lora-ntn-agent

Event-triggered LLM agent for adaptive LoRa parameter control (spreading factor, TX power, channel, link) in hybrid terrestrial + LEO-satellite IoT networks. A simulation study that runs entirely on a laptop CPU: no radio hardware, no GPU.

>  The PHY layer, channel models and network simulator are implemented and measured. The LLM agent is implemented but **not yet evaluated end to end**: the current E4–E6 outputs come from a scripted policy or a mock, not from the agent. Details in [Status](#status).

## Research question

Can a small local LLM (qwen2.5:3b through Ollama), called only on events, keep a hybrid terrestrial/satellite LoRa network healthy under anomalies better than Adaptive Data Rate (ADR), and at what cost in decision latency and number of calls?

Events considered: jamming, deep fade, end of a satellite pass, gateway outage, traffic surge.

Policies compared (target design): fixed SF7, fixed SF12, ADR, ML classifier trained on oracle labels, oracle upper bound, LLM agent with ADR fallback.

## Scope and assumptions

- EU868, 125 kHz bandwidth, 20-byte payload, 3 channels (868.1 / 868.3 / 868.5 MHz), 1% duty cycle.
- Terrestrial link: Okumura-Hata + log-normal shadowing.
- Satellite link: circular orbit (default 600 km), 2D overhead pass, Earth rotation ignored, free-space loss plus 3GPP TR 38.811 shadowing and clutter.
- Collisions: pure ALOHA with a 6 dB capture threshold.
- Energy: SX1276-style TX current model.
- The network simulator does not simulate chirps; it reads a PER model (a 0/1 SNR threshold until the lookup tables are generated).

## Architecture

```
phy/ (CSS modem, Monte Carlo)  -->  PER model
                                      |
channel/ (Hata, LEO pass, link budget) --> netsim/ (event-driven simulator, anomalies, detector)
                                              ^   |
                                  actions     |   | telemetry, triggers
                                              |   v
                          policies/ (fixed, ADR, oracle, ML, LLM agent + tools)
                                              |
                          experiments/ --> results/ (CSV, PNG)
```

The LLM agent runs on top of ADR. It is invoked on triggers (detector alert, high PER, satellite elevation crossing, periodic review), may call up to 6 tools per decision, and falls back to ADR on error. Tools: `get_link_stats`, `get_satellite_status`, `get_anomaly_report`, `get_network_load`, `set_tx_params` (validated).

## Status

| Component | State |
|---|---|
| PHY: CSS modem, AWGN and Doppler Monte Carlo (`phy/`) | Implemented. Doppler is applied within each symbol only; drift across a packet relative to the preamble is not modelled. PER lookup tables not generated yet. |
| Channel models (`channel/`) | Implemented. Satellite geometry and Doppler cross-checked against an independent calculation. |
| Network simulator (`netsim/`) | Implemented (ALOHA, capture, duty cycle, energy). Terrestrial scenarios only so far. |
| Anomalies and detector | Implemented: 5 event types, rolling z-score detector. |
| Baselines: fixed, ADR, oracle, ML classifier | Implemented. In E3, ADR behaves like fixed SF7 because the SNR is constant. |
| LLM agent (`policies/llm_agent/`) | Ollama tool-calling loop, 5 tools, validation, ADR fallback. Not evaluated. Known issues: no trigger cooldown, overrides have no expiry. |
| Tests | None yet. |

## Experiments

| ID | Measures | State |
|---|---|---|
| E1 | SER vs SNR (AWGN) and SER vs Doppler rate | Measured, 500 trials per point |
| E2 | SNR, Doppler and feasible SFs over a LEO pass | Measured. SF feasibility uses a heuristic rule (SNR above threshold and packet drift below one FFT bin), not yet validated against E1 |
| E3 | PDR and energy vs number of devices | Measured, terrestrial only, constant SNR |
| E4 | Anomaly recovery, ADR vs agent | **Scripted oracle** reacts at a fixed delay after onset. The real agent is not used. The CSV column named `LLM-Agent` will be renamed |
| E5 | Agent decision cost | **Mock** latency and invalid actions. Not a measurement |
| E6 | Ablations | **Scripted** variants. Not a measurement |

## What the data show so far

Measured values only, in the simplified setting described above.

**E2, LEO pass (600 km, 868 MHz).** Maximum Doppler shift is about 19.7 kHz at 10 degrees elevation. The Doppler rate at zenith is about 252 Hz/s. A pass above 10 degrees lasts about 8.5 minutes.

**E1b, Doppler sensitivity.** At threshold + 5 dB SNR with 500 trials per point, SER is 0 for every SF up to 600 Hz/s. SF12 reaches 0.078 at 800 Hz/s and 0.758 at 1000 Hz/s. With this receiver model, the Doppler rate at zenith at 600 km is well below the failure region. Published measurements report a similar picture for orbits above about 550 km (see References), so whether SF11/SF12 fail near zenith depends on drift across the packet, which is not yet modelled.

**E3, scalability** (3600 s, 600 s mean interval, 3 channels, SNR fixed at 5 dB, 5 seeds):

| Policy | PDR at 100 devices | PDR at 1000 devices | Energy per delivered packet (mJ), 100 / 1000 |
|---|---|---|---|
| Fixed SF12 | 86.7% | 23.2% | 411.5 / 1538.3 |
| Fixed SF7 | 99.1% | 93.9% | 15.4 / 16.3 |
| ADR | 99.1% | 93.9% | 15.4 / 16.3 |

SF12 saturates ALOHA because of its time on air (about 1.32 s). ADR matches fixed SF7 here only because every device has the same SNR.

## Known limitations

- E4–E6 do not evaluate the LLM agent yet.
- The simulator has no satellite link: `link="satellite"` is not connected to the pass model.
- Every device has the same fixed SNR in E3 and E4, so ADR has nothing to adapt to.
- PER is a 0/1 threshold; the Monte Carlo lookup tables are not generated.
- Constants taken from standards and datasheets (TR 38.811 tables, SX1276 current draw) must be checked against the source documents.
- Random streams are not yet keyed per device and transmission, so policies do not share identical channel realisations. Statistics use 3 to 5 seeds without paired analysis.
- Agent latency is recorded but does not advance simulated time.

## Roadmap

| # | Step | Branch |
|---|---|---|
| 1 | Trigger cooldown and override TTL in the agent | `fix/agent-cooldown-ttl` |
| 2 | E4 with the real Ollama agent (jamming first), traces saved | `feat/e4-real-agent` |
| 3 | E5 and E6 from real decisions | `feat/e5-e6-real` |
| 4 | Per-device SNR from distance; ADR distinct from fixed SF7 | `feat/per-device-snr` |
| 5 | Generate PER lookup tables and use them in the simulator | `feat/per-tables` |
| 6 | Packet-level Doppler model; make E1 and E2 consistent | `feat/packet-level-doppler` |
| 7 | Satellite link inside the network simulator | `feat/satellite-netsim` |
| 8 | pytest suite, keyed RNG, paired statistics | `test/basics` |

## Getting started

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

E1–E3 need no LLM. For the agent, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull qwen2.5:3b
```

Run an experiment (outputs go to `results/`):

```bash
python experiments/e1_phy_validation.py
python experiments/e2_satellite_pass.py
python experiments/e3_scalability.py
python experiments/e4_anomalies.py      # scripted oracle, not the LLM agent
python experiments/e5_agent_cost.py     # mock latency, no Ollama needed
python experiments/e6_ablations.py      # scripted variants
```

## Repository layout

```
phy/          css.py, ser_montecarlo.py, tables/ (empty until generated)
channel/      satellite.py, terrestrial.py, link_budget.py
netsim/       simulator.py, devices.py, anomalies.py
policies/     fixed.py, adr.py, oracle.py, ml_classifier.py, llm_agent/
experiments/  e1 to e6
results/      CSV and PNG outputs
docs/         PROJECT_SPEC.md, ARCHITECTURE.md, PLAYBOOK.md
AGENTS.md     rules for coding assistants working on this repo
LOG.md        development log
```

## Conventions

- Every number in this README comes from a CSV in `results/`, produced by a script in `experiments/`.
- Experiments use fixed seeds; outputs are regenerated by the scripts, not edited by hand.
- Design decisions that override the original specification are in `docs/ARCHITECTURE.md`.

## References

- A. Doroshkin et al., "Experimental Study of LoRa Modulation Immunity to Doppler Effect in CubeSat Radio Communications," IEEE Access, 2019.
- "First Flight Testing of LoRa Modulation in Satellite Radio Communications in Low-Earth Orbit," IEEE Access, 2022.
- Semtech, LoRa modem design guide (time-on-air formula), AN1200.13.
- 3GPP TR 38.811, Study on New Radio (NR) to support non-terrestrial networks.

## License

To be added.
