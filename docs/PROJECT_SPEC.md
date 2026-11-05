# Project 2: LLM Agent for Adaptive Modulation in Hybrid Terrestrial–Satellite IoT Networks

**Extracted from the detailed build plan for Rihem Fathallah's research portfolio.**
Target profile: machine learning for 5G/6G and non-terrestrial wireless networks.
Everything runs on a personal computer: no radio hardware, no GPU.

---

## Contents

1. Summary and planning
2. Ground rules
3. Project details (research question, scenario, LoRa PHY, channel, phases, experiments, repository, pitfalls, reading list)
4. Setup for this project
5. Interview preparation
6. CV lines

---

## 1. Summary and planning

| Item | Value |
|---|---|
| Mitacs projects it prepares for | Langar (ETS, GenAI modulation agent), Trajkovic (SFU, anomaly detection), Jaafar (NTN) |
| Difficulty | Medium |
| Time to first result | ~5 weeks |
| Planned period | November 2026 (phases 1–4 and a first agent run on one scenario); full benchmark January–April 2027 |
| Milestone for end of November | Validated LoRa PHY, network simulator, ADR baseline, first agent run |

**Why this position in the order:** Project 2 reuses the channel-modeling habits from Project 1 (link budget, Monte Carlo, baselines, config files).

**Shared thread with the other projects:** *can a learning-based controller beat a classical method in a wireless network, and at what cost?*

If you fall behind, cut extensions, never baselines.

---

## 2. Ground rules (shared by all three projects)

1. **Only real results.** Your CV says "In progress" for these projects. Keep it that way until you have figures. Every number in a report or on your CV must come from a script in your repository that anyone can re-run.
2. **Git from day one.** One repository per project, commits at least every working session, a README with the main figure at the top.
3. **Reproducibility.** Parameters in a YAML config file, fixed random seeds, raw results saved as CSV, and one script per figure that regenerates it from the CSV.
4. **Always compare against baselines**, including a simple one (random, fixed, rule-based) and, when possible, an upper bound. A method with no baseline is a demo, not research.
5. **Keep an experiment log.** A `LOG.md` file with the date, what you ran, what you saw and what you will try next. This log is extremely useful in interviews: it lets you tell the story of your work in detail.
6. **Write while you build.** Open the IEEE conference template on Overleaf on day one. Each project ends with a 4–6 page report: Introduction, System Model, Proposed Method, Results, Conclusion.

---

## 3. Project: LLM Agent for Adaptive Modulation in Hybrid Terrestrial–Satellite IoT Networks

### 3.1 Research question

Can an LLM agent that reads link telemetry and calls control tools choose LoRa transmission parameters better than a rule-based adaptive data rate (ADR) algorithm and a trained ML classifier, especially during events the rules don't anticipate (jamming, deep fades, the end of a satellite pass)? And what does it cost in decision latency?

### 3.2 Scenario

- A rural area with 100 to 1,000 LoRa end devices (environmental sensors). Each sends a 20-byte payload about every 10 minutes, with random jitter.
- A **terrestrial LoRa gateway** at the center covers devices within a few kilometers.
- A **LEO satellite gateway** at 600 km altitude passes over the area a few times per day and is usable when its elevation angle is above a minimum (for example 10°).
- EU868 band: three default channels (868.1, 868.3 and 868.5 MHz), 125 kHz bandwidth, 1% duty cycle.
- The controller decides, per device or group: **link** (terrestrial, satellite or wait), **spreading factor** SF7–SF12, **transmit power** 2–14 dBm, and **channel**.

### 3.3 LoRa physical layer

**Key formulas**

```
Chips per symbol:    N = 2^SF
Symbol duration:     T_s = 2^SF / BW
Bit rate:            R_b = SF · (BW / 2^SF) · 4 / (4 + CR)        CR ∈ {1, 2, 3, 4}
```

**Time on air** (Semtech LoRa modem designer's guide):

```
T_preamble = (n_preamble + 4.25) · T_s
n_payload  = 8 + max( ceil( (8·PL − 4·SF + 28 + 16·CRC − 20·IH) / (4·(SF − 2·DE)) ) · (CR + 4), 0 )
ToA        = T_preamble + n_payload · T_s
```

`PL` = payload bytes, `CRC = 1`, `IH = 0` (explicit header), `DE = 1` when `T_s > 16 ms` (SF11 and SF12 at 125 kHz), `n_preamble = 8`.
Sanity check: SF7, 125 kHz, 20 bytes, CR = 1 gives about 56.6 ms. Write a unit test for it.

**Demodulation SNR thresholds** (commonly used values)

| SF | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| Required SNR (dB) | −7.5 | −10 | −12.5 | −15 | −17.5 | −20 |

**Receiver sensitivity:** `S [dBm] = −174 + 10·log10(BW) + NF + SNR_req`.

**Chirp spread spectrum (CSS) at baseband**, sampled at `f_s = BW`, for symbol value `m ∈ {0, …, N−1}`:

```
x_m[n] = exp( j2π · ( n²/(2N) + (m/N − 1/2) · n ) ),     n = 0, …, N−1
```

Demodulation: multiply the received symbol by `conj(x_0[n])`, which leaves a pure tone at frequency `m/N`; take the FFT and pick the strongest bin.

**Doppler.** Apply a frequency offset that drifts over the packet: multiply the samples by `exp( j2π · (f_D·t + ½·f_D'·t²) )` with `t = n / BW`. Real receivers estimate and remove the static offset `f_D` from the preamble, so model a residual error plus the **Doppler rate** `f_D'`, which keeps drifting during the packet.

**The key physical insight to verify.** An FFT bin is `BW / 2^SF` wide: about 976 Hz at SF7 but only about 30.5 Hz at SF12 (125 kHz). A 20-byte packet lasts about 57 ms at SF7 and about 1.3 s at SF12. For a 600 km LEO pass at 868 MHz, the Doppler rate near zenith is roughly `(f_c/c) · v²/h ≈ 275 Hz/s`. Over an SF12 packet, the frequency drifts by hundreds of hertz, many times the bin width, while at SF7 the drift stays small. So the most robust setting on the ground (SF12) can be the most fragile over a satellite. That tension is what makes adaptive modulation over LEO an interesting research problem. Confirm it with your simulator rather than assuming it.

### 3.4 Channel and link budget

**Terrestrial link: Okumura–Hata** (valid 150–1,500 MHz, 1–20 km), `f` in MHz, `d` in km, heights in meters:

```
a(h_m)  = (1.1·log10 f − 0.7)·h_m − (1.56·log10 f − 0.8)
L_urban = 69.55 + 26.16·log10 f − 13.82·log10 h_b − a(h_m) + (44.9 − 6.55·log10 h_b)·log10 d
L_rural = L_urban − 4.78·(log10 f)² + 18.33·log10 f − 40.94
```

Add log-normal shadowing with a standard deviation of 6–8 dB (config parameter).

**Satellite link geometry** (`R_E = 6371 km`, altitude `h`, elevation angle `ε`):

```
Slant range:   d(ε) = sqrt( (R_E·sin ε)² + h² + 2·h·R_E ) − R_E·sin ε
FSPL [dB]:     20·log10(d_km) + 20·log10(f_MHz) + 32.45
Delay:         τ = d / c
Orbital speed: v = sqrt( G·M_E / (R_E + h) )  ≈ 7.56 km/s at 600 km
```

**Pass model:** simulate an overhead pass in 2D: the satellite moves on a circle of radius `R_E + h` at angular speed `v / (R_E + h)`. Compute the range `d(t)`, the elevation, the Doppler shift `f_D(t) = −(f_c / c) · dd/dt` and the Doppler rate numerically at each time step. Ignoring Earth's rotation is fine for this project; state it as an assumption. Check the numbers: maximum Doppler around ±20 kHz, a pass lasting about ten minutes, delay of a few milliseconds.

**Extra satellite losses:** shadow fading and clutter loss by elevation angle and environment from **3GPP TR 38.811**. Its tables are given for S-band and Ka-band, so use the S-band rural values as an approximation for 868 MHz and say so in the report. Atmospheric absorption is negligible at sub-GHz frequencies for elevations above 10°.

**Link budget:**

```
SNR = P_tx + G_tx + G_rx − L_path − L_shadow − L_clutter − (−174 + 10·log10(BW) + NF)
```

Make the satellite receive antenna gain a parameter. Run the budget before anything else: you will probably find that the satellite link only closes at high SF near zenith, which is exactly where the Doppler rate is worst. That is a result worth a figure.

### 3.5 Phases

**Phase 1: LoRa PHY simulator and lookup tables (weeks 1–2)**

- Implement CSS modulation and demodulation.
- Monte Carlo symbol error rate (SER) vs SNR for SF7–SF12 in AWGN. The curves should shift by about 2.5 dB per SF step, consistent with the threshold table.
- SER vs Doppler rate for each SF.
- Convert SER to packet error rate: `PER = 1 − (1 − SER)^(n_symbols)`, a simplification that ignores coding; state it.
- Save **lookup tables** `PER(SNR, SF, Doppler rate)`.

> **Important:** never simulate chirp samples inside the network simulator; it would be far too slow. The network simulator only reads the lookup tables.

**Phase 2: Link-level channel (week 2)**

- Terrestrial Hata + shadowing, satellite pass geometry, FSPL, delay, Doppler, TR 38.811 losses, link budget → SNR per packet.
- Figure: SNR, Doppler shift and Doppler rate over a pass, with the feasible SFs shaded.

**Phase 3: Network simulator (weeks 3–4)**

- Event-driven simulator (a `heapq` event queue or SimPy).
- Pure ALOHA access: a transmission fails by collision if another packet on the same channel and same SF overlaps it in time, unless it is at least 6 dB stronger (capture effect). Treat different SFs as orthogonal (a simplification; imperfect orthogonality is a known extension).
- Duty cycle: after a transmission of length ToA, the device stays silent for `99 · ToA`.
- Success = no collision, and a Bernoulli draw using the PER lookup table.
- Energy per transmission: `E = V · I_tx(P_tx) · ToA`, with the supply current per output power taken from the SX1276 datasheet table.
- Sanity check against theory: pure ALOHA throughput `S = G · e^(−2G)` peaks at about 18.4% for `G = 0.5`.

**Phase 4: Baselines (week 4)**

| Baseline | Rule |
|---|---|
| Fixed SF12, max power | Most robust on the ground, highest energy |
| Fixed SF7 | Fastest, least robust |
| ADR (based on Semtech's recommended algorithm) | After 20 uplinks: `margin = max(SNR of last 20) − SNR_req(SF) − 10 dB`, `N_step = floor(margin / 3)`. While `N_step > 0` and `SF > 7`: lower SF by 1. Then while `N_step > 0` and power above minimum: lower power by 3 dB. If `N_step < 0`: raise power toward maximum |
| ML classifier | Scikit-learn `HistGradientBoostingClassifier` trained on oracle labels (below) |
| Oracle (upper bound) | For each decision, tries every (SF, power) on the true upcoming channel and picks the lowest-energy setting with PER ≤ 1% |

Classifier features: last *k* SNR values, PER over the last window, current SF and power, link type, satellite elevation, Doppler rate, anomaly flag.

**Phase 5: Anomaly injection and detection (week 5)**

| Event | How to simulate | What a good controller should do |
|---|---|---|
| Jamming burst | Interference 10–20 dB above the noise floor on one channel for 2–10 minutes | Move channel, raise SF or power, or wait |
| Deep fade / blockage | Extra 15–25 dB loss for a subset of devices for a few minutes | Temporarily raise SF or power, then return |
| End of satellite pass | Elevation falls below the minimum | Stop satellite transmissions, buffer or switch link |
| Gateway outage | Terrestrial gateway down for 30 minutes | Move edge devices to the satellite during passes |
| Traffic surge | Five times the normal reporting rate | Spread devices across SFs and channels |

The right column describes expected good behavior to test, not results.

Detector: rolling z-score on each channel's noise floor, SNR and PER, optionally Isolation Forest. Report its precision and recall against the injected labels.

**Phase 6: LLM agent (weeks 5–6)**

*Model:* local, through Ollama. Start with `qwen2.5:3b` or `llama3.2:3b` (both support tool calling); try `qwen2.5:7b` if you have 16 GB RAM.

*Architecture:* ADR runs by default. The agent is called **only on triggers**:

- the anomaly detector raises an alert;
- PER over the last packets exceeds 10%;
- a satellite elevation threshold is crossed;
- a periodic review every 30 simulated minutes.

This keeps the number of LLM calls and the latency low, and it's a design choice you can defend in an interview.

*Tools exposed to the agent:*

| Tool | Returns / does |
|---|---|
| `get_link_stats(scope, window_s)` | Mean and minimum SNR, PER, current SF, power, link, ToA, remaining duty-cycle budget |
| `get_satellite_status(device_id)` | Elevation now and in 1, 5 and 10 minutes, Doppler shift and rate, time until pass end |
| `get_anomaly_report(scope)` | Detector output: suspected type, start time, affected channels or devices, confidence |
| `get_network_load()` | Packets per SF and channel, collision rate |
| `set_tx_params(device_ids, sf, tx_power_dbm, channel, link)` | Validated by the simulator; returns "applied" or "rejected" with the reason |

*System prompt contents:* the agent's role; LoRa facts (threshold table, ToA doubling per SF step, energy proportional to ToA and current, high SF fragile under LEO Doppler rate, duty cycle); the objective (maximize delivery ratio first, then minimize energy); the rule "read before you write" (call a `get_` tool before any `set_` call).

*Guardrails:* validate every tool call against a JSON schema, reject out-of-range values and count them, cap tool calls at 6 per decision, and fall back to ADR on timeout.

*Loop sketch (ollama Python library):*

```python
import json
import ollama

TOOLS = [get_link_stats, get_satellite_status, get_anomaly_report,
         get_network_load, set_tx_params]          # plain Python functions with docstrings
REGISTRY = {f.__name__: f for f in TOOLS}

def run_agent(event_description):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": event_description}]
    for _ in range(6):
        response = ollama.chat(model="qwen2.5:3b", messages=messages, tools=TOOLS,
                               options={"temperature": 0, "seed": 0})
        messages.append(response.message)
        if not response.message.tool_calls:
            break
        for call in response.message.tool_calls:
            result = REGISTRY[call.function.name](**call.function.arguments)
            # check the ollama-python examples for the exact tool-message format of your version
            messages.append({"role": "tool", "content": json.dumps(result),
                             "tool_name": call.function.name})
    return messages   # save it: the reasoning trace is part of your results
```

*Time handling:* pause simulated time while the agent thinks, so its latency doesn't distort the simulation, but record the real wall-clock latency and discuss whether it would fit an IoT network where devices report every few minutes.

### 3.6 Experiments

| ID | Experiment | Output |
|---|---|---|
| E1 | PHY validation | SER vs SNR per SF (AWGN); SER vs Doppler rate per SF |
| E2 | Satellite pass | SNR, Doppler and feasible SFs over time |
| E3 | Scalability, no anomalies | Delivery ratio and energy per delivered packet vs number of devices (100–1,000), every policy |
| E4 | Anomaly scenarios | Delivery ratio over time around each event; recovery time per policy |
| E5 | Cost of the agent | Latency (median and 95th percentile), LLM calls per hour, invalid-action rate, 3B vs 7B model |
| E6 | Ablations | Agent without the anomaly tool; agent without LoRa facts in the prompt |

Run every experiment with 5 seeds and report mean ± standard deviation.

**Metric definitions**

- **Packet delivery ratio (PDR):** delivered packets / sent packets.
- **Energy per delivered packet (mJ):** total transmit energy / delivered packets.
- **Recovery time:** time from event start until PDR returns to within 5% of its pre-event level.
- **Invalid-action rate:** rejected `set_tx_params` calls / total `set_tx_params` calls.

### 3.7 Repository structure

```
llm-lora-ntn-agent/
├── README.md
├── LOG.md
├── configs/                 # scenarios S1–S5 as YAML
├── phy/
│   ├── css.py               # modulation, demodulation, Doppler
│   ├── ser_montecarlo.py
│   └── tables/              # PER lookup tables
├── channel/
│   ├── terrestrial.py       # Hata + shadowing
│   ├── satellite.py         # orbit, slant range, Doppler, TR 38.811 losses
│   └── link_budget.py
├── netsim/
│   ├── simulator.py         # event-driven ALOHA + duty cycle
│   ├── devices.py
│   └── anomalies.py
├── policies/
│   ├── fixed.py
│   ├── adr.py
│   ├── ml_classifier.py
│   ├── oracle.py
│   └── llm_agent/
│       ├── tools.py
│       ├── prompt.py
│       └── agent.py
├── experiments/             # one script per experiment E1–E6
├── results/
└── report/
```

### 3.8 Common pitfalls

1. **Sample-level simulation inside the network simulator:** use lookup tables.
2. **LLM randomness:** set temperature 0 and a fixed seed, and still run several seeds.
3. **Unfair information:** if the agent sees anomaly reports, give the ML classifier the anomaly flag too, and discuss the difference.
4. **Hidden assumptions:** list them all in the report (no Earth rotation, S-band TR 38.811 tables, orthogonal SFs, PER from uncoded SER).
5. **Accept any honest outcome.** The agent may match ADR in normal conditions and only help during rare events, or not help at all. Either result is valuable if it's well measured.

### 3.9 Reading list

- L. Vangelista, "Frequency Shift Chirp Modulation: The LoRa Modulation," *IEEE Signal Processing Letters*, 2017.
- Semtech, *LoRa Modem Designer's Guide* (AN1200.13), for the time-on-air formula.
- M. Bor, U. Roedig, T. Voigt and J. Alonso, "Do LoRa Low-Power Wide-Area Networks Scale?," *ACM MSWiM*, 2016.
- 3GPP TR 38.811, *Study on New Radio (NR) to Support Non-Terrestrial Networks*.
- J. A. Fraire, S. Céspedes and N. Accettura, "Direct-to-Satellite IoT: A Survey of the State of the Art and Future Research Perspectives," 2019.
- S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," *ICLR*, 2023.
- Search keywords for recent work: "LoRa LEO satellite Doppler", "LoRaWAN ADR", "large language models for wireless network management".

---

## 4. Setup for this project

| Item | What to install |
|---|---|
| Python | Miniconda, a dedicated environment, Python 3.10 or 3.11 |
| Scientific stack | `pip install numpy scipy matplotlib pandas scikit-learn pyyaml` |
| PyTorch (CPU) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` (optional here) |
| Ollama | `curl -fsSL https://ollama.com/install.sh | sh`, then `ollama pull qwen2.5:3b` |
| Python client | `pip install ollama` (and `simpy` if you choose SimPy over `heapq`) |
| Writing | VS Code, Git, Overleaf account (IEEE conference template) |

**Hardware:** 8 GB RAM is enough for the 3B model; 16 GB makes `qwen2.5:7b` comfortable.

---

## 5. Interview preparation

Prepare these answers based on what you actually did.

- Why can SF12 fail over a LEO satellite even with enough SNR? *(Long time on air combined with Doppler rate drift.)*
- Why not call the LLM for every packet? *(Latency, energy, cost; event-triggered design.)*
- How do you stop the agent from taking invalid or harmful actions? *(Schema validation, range checks, fallback to ADR, invalid-action metric.)*
- How do you know the agent adds value over ADR? *(Same scenarios and seeds, ablations, recovery time after events.)*
- What are the limits of your satellite model? *(No Earth rotation, S-band TR 38.811 tables as approximation, orthogonal SFs.)*

---

## 6. CV lines

Keep **"In progress"** on your CV until you have figures. Then use the metrics this project naturally produces: delivery ratio or energy gain over ADR, recovery time after anomalies, median decision latency.

Example format: *"Built an event-triggered LLM agent for LoRa parameter control over terrestrial–LEO links, improving [metric] by [X]% over ADR with a median decision latency of [Y] s."* The bracketed numbers must come from your results folder.
