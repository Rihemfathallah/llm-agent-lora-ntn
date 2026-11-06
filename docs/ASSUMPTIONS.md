# Assumptions and model limits

Living file: every simplification in the code is listed here (AGENTS.md rule 8).

## Physical layer

| # | Assumption | Where | Expected impact |
|---|---|---|---|
| A4 | Coding rate 4/5. Hamming(5,4) only detects errors, so one wrong symbol loses the packet (CRC). PER is computed packet by packet (no `1 − (1 − SER)^n` shortcut). The header block (8 symbols, CR 4/8) is treated like CR 4/5 (its single-bit correction is ignored). | `phy/receiver.py` | Slightly pessimistic for the header. |
| A5a | Receiver A ("preamble"): the static Doppler is estimated on the preamble and removed once; no tracking during the payload. | `phy/receiver.py` | Pessimistic bound (simple SDR receiver). |
| A5b | Receiver B ("tracking"): first-order decision-directed loop on the fractional FFT peak offset (Jacobsen estimator), initialised at the last preamble up-chirp, no knowledge of the drift rate. Default in the network experiments. | `phy/receiver.py` | A model of a commercial receiver, not Semtech's proprietary demodulator. Its predicted SF12 limit (E2b) is compared with the 550 km measured by Doroshkin et al. |
| A5c | Perfect timing synchronisation; no residual static frequency error (parameter `residual_cfo_hz`, default 0); SNR constant during a packet. | `phy/receiver.py` | Optimistic. |
| A5d | Reduced-rate symbols (header block; payload when DE = 1) carry SF−2 bits and are decided by rounding the FFT peak to a multiple of 4. | `phy/css.py` | Matches the LoRa low-data-rate optimisation. |
| A5e | No implementation loss: the ideal noncoherent receiver reaches PER = 1 % within 1.4 dB of the Semtech thresholds (E1a), so no calibration was applied. | `phy/` | Slightly optimistic at SF11–12. |
| A5f | PER tables: 400 packets per point, 1 dB SNR steps, 15 Doppler-rate points; bilinear interpolation, no interpolation across SFs. | `phy/ser_montecarlo.py` | PER resolution ≈ 0.25 %. |

## Channel

| # | Assumption | Where | Expected impact |
|---|---|---|---|
| A1 | No Earth rotation; circular orbit; closed-form pass geometry for a given maximum elevation. Every device sees the same elevation (the area is small compared with the footprint). | `channel/satellite.py` | Pass duration and Doppler slightly off (a few %). |
| A2 | TR 38.811 S-band suburban/rural tables (LOS probability, LOS/NLOS shadow fading, NLOS clutter loss) used for 868 MHz. One LOS/NLOS draw per packet. | `channel/satellite.py` | Clutter loss probably overestimated at 868 MHz. |
| A7 | Atmospheric and ionospheric losses neglected (sub-GHz, elevation > 10°). | `channel/` | Negligible. |
| A8 | Okumura–Hata rural, h_b = 30 m, h_m = 1.5 m. Remote devices (30–45 km) are beyond the 20 km validity range; Hata is extrapolated only to decide that they are out of terrestrial coverage. | `netsim/devices.py` | Coverage boundary approximate. |
| A10 | Terrestrial shadowing fixed per device (σ = 7 dB) plus per-packet variation (σ = 3 dB, log-normal). No multipath fading model beyond that. | `netsim/devices.py` | Optimistic for mobile devices; devices are static sensors. |
| A11 | Jamming: interference I dB above the noise floor on one channel at the terrestrial gateway; SNR loss = 10·log10(1 + 10^(I/10)). | `netsim/anomalies.py` | — |

## Network

| # | Assumption | Where | Expected impact |
|---|---|---|---|
| A3 | Different SFs are orthogonal (no inter-SF interference). | `netsim/simulator.py` | Collisions underestimated at high load. |
| A12 | Capture: the stronger of two overlapping same-channel same-SF packets survives if ≥ 6 dB stronger; otherwise both are lost. No timing (preamble-lock) condition. | `netsim/simulator.py` | — |
| A13 | A packet is addressed to one gateway (its link) and only interferes with packets addressed to that gateway. The satellite also serves devices outside the area; that background traffic is not modelled. | `netsim/simulator.py` | Satellite collisions underestimated. |
| A14 | Duty cycle 1 % per device on sub-band g1 (868.0–868.6 MHz holds the three channels): silent 99·ToA after each packet. A reading that cannot be sent before the next one is dropped. The first version enforced it per channel (3 % in effect). | `netsim/simulator.py` | — |
| A15 | Unconfirmed uplinks, no retransmission, no buffering: a reading not sent ("wait") or lost is not delivered. Store-and-forward is an extension. | `netsim/simulator.py` | Penalises "wait" in the reading delivery ratio, not in PDR. |
| A16 | Control plane, LoRaWAN class A: a network-side decision (ADR, ML, agent, rules) reaches a device only in the downlink after one of its *delivered* uplinks; downlinks never fail. During a pass, a broadcast satellite downlink can reach every device (`satellite_broadcast_downlink`). The oracle is not constrained (upper bound). | `netsim/simulator.py` | The broadcast downlink is optimistic; it is what makes "move devices to the satellite during an outage" possible at all. |
| A17 | Device-side ADR back-off (ADR_ACK_LIMIT 64 + ADR_ACK_DELAY 32 uplinks) not modelled: at one uplink per 10 min it acts after ≈ 16 h, outside the 3 h horizon. | `policies/adr.py` | None within the horizon. |
| A18 | ADR history is cleared when ADR changes the parameters (configurable); ADR and ML start from ADR's fixed point on 20 earlier uplinks (warm start, separate random stream), so a 3 h run measures steady state, not convergence. | `policies/adr.py` | — |
| A19 | Traffic surge = extra event-driven readings (Poisson) on top of the periodic schedule, total rate × 5. | `netsim/devices.py` | — |

## Energy

| # | Assumption | Where | Expected impact |
|---|---|---|---|
| A9 | Transmit energy only: E = 3.3 V · I_tx(P) · ToA. I_tx from the SX1276 datasheet RFO points (7 dBm: 20 mA, 13 dBm: 29 mA), linear in dBm, extrapolated to 2 and 14 dBm. Receive windows, sleep and MCU energy not counted. | `netsim/energy.py` | Absolute energy underestimated; comparisons between policies hold. |

## Agent and evaluation

| # | Assumption | Where | Expected impact |
|---|---|---|---|
| A6 | Simulated time is frozen while the agent thinks; wall-clock latency is recorded separately (E5). | `policies/llm_agent/agent.py` | Agent reactivity overestimated if its latency exceeds the reporting interval. |
| A20 | The network server, the detector, the ML classifier, the rules and the agent only use network-server-side information (delivered uplinks, losses inferred from frame-counter gaps, gateway noise-floor measurements, satellite ephemeris). Exception: `get_network_load` reports the true collision rate as a stand-in for a gateway estimate. | `policies/`, `netsim/` | — |
| A21 | Events last 30 min (jamming spec range: 2–10 min). With class-A control and one uplink per 10 min, no network-side controller can act on events shorter than about two reporting periods. | `configs/s*.yaml` | Shorter events would show no difference between controllers. |
| A22 | In-sandbox agent runs use Qwen3-0.6B (Q4_K_M GGUF, llama.cpp) because Ollama and Hugging Face were unreachable; the reference model of the protocol (qwen2.5:3b via Ollama) runs on the author's laptop with the same scripts. | `results/` | 0.6B results are a lower bound on what a 3B/7B model does. |
