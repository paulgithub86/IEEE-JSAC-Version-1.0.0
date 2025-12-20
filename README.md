# JSAC Reproducibility Package (IEEE JSAC) 🧪📈

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)

> ✅ **MIT Licensed.** If you use this code in academic work, please **cite the paper** ✅️.

---

## ⚠️ Important safety / liability warning

This software is provided **“as is”**, without warranty of any kind (see `LICENSE`).  

By running any scripts in this repository, you acknowledge that execution may consume significant CPU/GPU/RAM resources and may cause crashes, freezes, data loss, or other unexpected behavior depending on your environment.⚠️

**You are solely responsible for how you run this code.** The authors/contributors are **not responsible** for any damage, malfunction, loss of data, loss of productivity, hardware failure, or other issues (direct or indirect) that may occur during installation or execution.⚠️

> 🔆 **Practical advice:** run inside a virtual environment or container, and avoid running on production machines.

---

This repository contains the simulation code, plotting scripts, and CSV logs used to generate the **main performance plots** in our **accepted IEEE JSAC paper**:

- **Fig. 7:** Convergence behavior + mean end-to-end latency  
- **Fig. 8:** Privacy budget evolution + sensing measure (dB)

These materials are released to support **transparent verification and reproducibility** of the reported numerical findings.

---

## 1) Purpose & citation request 🎯

Reproducibility is critical for validating scientific claims. To support independent verification and follow-up research, we publicly release the core simulation scripts and logs used to produce the primary plots (Fig. 7–Fig. 8).

> 📌 **If you extend or adapt this codebase, please cite the paper and link back to this repository.**

---

## 2) Where parameter settings are documented 📚

All key hyperparameters, modeling assumptions, and mechanisms are documented in the paper:

- **Section V-B:** *Implementation Declaration (Reproducibility and Fidelity)*

> This repository is the executable companion to **Section V-B**: it implements those settings and reproduces the logs/plots corresponding to Fig. 7–Fig. 8.

---

## 3) Repository contents 🗂️

### ✅ Core simulator
- `JSAC_Public_Code.py`  
  End-to-end **Q-FDRL** simulator (system-level prototype), including:
  - Clustered V2X ISAC Markov game
  - Finite-blocklength UL + front-haul (with outage handling)
  - Sensing quality Πᵢ(t) and sensing-driven data generation
  - Differential privacy accounting + DP noise injection
  - RSU-side federated aggregation (AdaGrad-preconditioned FedAvg)
  - Two-hop QKD/OTP abstraction with key buffers
  - Hybrid policy: Transformer encoder + VQC head (see code)

### 📈 Plot scripts
- `Fig_7_Convergence.py` → builds Fig. 7 from `convergence.csv`, `latency_traces.csv`  
- `Fig_8_Privacy_Budget.py` → builds Fig. 8 from `privacy_budget.csv`, `convergence.csv`

### 📊 Logged results (CSV)
- `convergence.csv` — per-episode summary (`episode, mean_return, epsilon, mean_latency_ms, mean_sensing_db, ...`)
- `privacy_budget.csv` — privacy tracking (`episode, epsilon, ...`)
- `latency_traces.csv` — step-level latency traces (`episode, step, veh_id, T_tot_ms, ...`)
- `ul_sinr_traces.csv`, `fh_sinr_traces.csv`, `sensing_traces.csv`

---

## 4) Quickstart (recommended order) ⚡

### Step 1 — Run simulator (regenerate CSV logs)
```bash
python JSAC_Public_Code.py
```

### Step 2 — Generate Fig. 7 (convergence + latency)
```bash
python Fig_7_Convergence.py
```

### Step 3 — Generate Fig. 8 (privacy + sensing)
```bash
python Fig_8_Privacy_Budget.py
```

✅ **Expected outputs**
- `convergence.csv`, `privacy_budget.csv`, `latency_traces.csv`
- `ul_sinr_traces.csv`, `fh_sinr_traces.csv`, `sensing_traces.csv`
- `convergence_Fig_7.png`, `privacy_sensing_Fig_8.png`

---

## 5) Software requirements 🧰

- **Python:** 3.9+ recommended

Main packages:
- `numpy`, `pandas`, `scipy`, `sympy`, `tqdm`
- `tensorflow` (keras)
- `cirq`
- `matplotlib`
- `openpyxl` *(only if XLSX export is enabled)*

Install (example):
```bash
pip install numpy pandas scipy sympy tqdm tensorflow cirq matplotlib openpyxl
```

> 🔁 **Reproducibility note:** Seeds are set in code where applicable. Minor run-to-run variation may occur due to backend non-determinism (e.g., some TensorFlow kernels). **Trends and reported scales should remain consistent.**

---

## 6) Scope / modeling disclaimer 🧾

This is a **system-level prototype** aligned with the paper. Full waveform-level RF PHY and detailed QKD optics are **not** simulated; their effects are represented via controlled abstractions (e.g., finite-blocklength approximation, free-space path loss, thermal noise, simplified interference/self-interference terms, and a BB84-inspired key-rate formula) to keep execution lightweight for convergence studies.

---

## 7) License ✅

This project is released under the **MIT License** (see `LICENSE`).

---

## 8) Citation 🧷

### BibTeX of the main Article 🌸
```bibtex
@article{PAUL_JSAC_QFDRL_2026,
  title   = {Large AI Model Driven Quantum-Enhanced Transformer-VQC Federated DRL for Privacy Preservation in Vehicular Networks},
  author  = {Anal Paul and Keshav Singh},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2026},
  note    = {Code: this repository},
}
```

---

## 9) Contact ✉️

Email: `apaul@ieee.org`
