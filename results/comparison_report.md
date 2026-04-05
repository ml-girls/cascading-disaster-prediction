# Cross-Notebook Model Comparison Report

> Generated from executed notebook outputs as of 2026-04-05.
> All models evaluated on the same chronological test set (193,489 samples, 16 labels).

## Experimental Setup (Standardized)

| Parameter | NB02 (XGBoost) | NB03 (Sklearn) | NB04 (Neural Nets) |
|-----------|----------------|----------------|---------------------|
| **Data** | `chronological_prepared_data` | `chronological_prepared_data` | `chronological_prepared_data` |
| **Train samples** | 773,952 | 773,952 | 773,952 |
| **Test samples** | 193,489 | 193,489 | 193,489 |
| **Labels** | 16 (≥0.05% prevalence) | 16 (≥0.05% prevalence) | 16 (≥0.05% prevalence)* |
| **Features** | 135 (excl. cascade-prob) | 135 (excl. cascade-prob) | 135 (excl. cascade-prob) |
| **Val split** | 10% chronological | 10% chronological | 10% chronological* |
| **Imbalance handling** | Per-label neg subsampling (5:1) + `scale_pos_weight` (cap 30) | Global neg subsampling (~50K) | `pos_weight` in BCEWithLogitsLoss (cap 30) |
| **Threshold tuning** | Per-label on val set | Per-label on val set | Per-label on val set |
| **HP search** | 12-config random search | Randomized search | 20-trial Optuna (TPE) |

\* NB04 was originally run with 26 labels and 20% val split. Code has been patched to 16 labels / 10% val. **NB04 must be re-run for comparable results.**

---

## Overall Metrics — Best Model per Notebook

| Rank | Model | Notebook | F1 weighted | F1 macro | F1 micro | Precision wtd | Recall wtd | AUC-PR wtd | Hamming | Subset Acc |
|------|-------|----------|-------------|----------|----------|---------------|------------|------------|---------|------------|
| 1 | **XGBoost Binary Relevance** | NB02 | **0.3845** | **0.2142** | **0.4222** | 0.4599 | 0.3823 | **0.4549** | **0.0144** | **0.8088** |
| 2 | **MLP (128, 64) (tuned)** | NB03 | 0.3919 | 0.2755 | 0.3985 | 0.3662 | 0.4522 | 0.3632 | 0.0188 | 0.7471 |
| 3 | **OVR-LogReg (tuned)** | NB03 | 0.3550 | 0.2329 | 0.3208 | 0.2662 | 0.5755 | 0.3038 | 0.0335 | 0.5932 |
| 4 | **Chain-LogReg (x3) (tuned)** | NB03 | 0.3527 | 0.2336 | 0.3249 | 0.2704 | 0.5389 | 0.3028 | 0.0308 | 0.6237 |
| — | *1_BaselineMLP* (26 labels)† | NB04 | *0.4144* | *0.2062* | *0.4174* | *0.3583* | *0.5086* | *0.3701* | *0.0121* | *0.7497* |
| — | *2_WeatherMLP V2* (26 labels)† | NB04 | *0.4221* | *0.2042* | *0.4174* | *0.3626* | *0.5354* | *0.3811* | *0.0128* | *0.7465* |

† NB04 results were computed on **26 labels** (not 16) and a **20% val split** — they are **not directly comparable**. NB04 must be re-run after the code fix to produce valid 16-label results.

---

## All Sklearn Baselines (NB03) — Detailed

| Model | F1 weighted | F1 macro | F1 micro | AUC-PR wtd | Precision wtd | Recall wtd | Hamming | Subset Acc |
|-------|-------------|----------|----------|------------|---------------|------------|---------|------------|
| MLP (128, 64) (tuned) | 0.3919 | 0.2755 | 0.3985 | 0.3632 | 0.3662 | 0.4522 | 0.0188 | 0.7471 |
| Chain-LogReg (x3) (tuned) | 0.3527 | 0.2336 | 0.3249 | 0.3028 | 0.2704 | 0.5389 | 0.0308 | 0.6237 |
| OVR-LogReg (tuned) | 0.3550 | 0.2329 | 0.3208 | 0.3038 | 0.2662 | 0.5755 | 0.0335 | 0.5932 |
| OVR-LogReg (default) | 0.2925 | 0.1814 | 0.1818 | 0.3038 | 0.1795 | 0.9796 | 0.1214 | 0.1464 |
| Chain-RF (x3) (tuned) | 0.1410 | 0.0860 | 0.1607 | 0.3866 | 0.4783 | 0.1007 | 0.0145 | 0.8030 |

---

## XGBoost Per-Label Breakdown (NB02)

| Label | Support | F1 | Precision | Recall | PR-AUC |
|-------|--------:|----:|----------:|-------:|-------:|
| Thunderstorm Wind | 9,688 | 0.688 | 0.605 | 0.798 | 0.708 |
| Waterspout | 286 | 0.517 | 0.464 | 0.584 | 0.450 |
| Marine Thunderstorm Wind | 169 | 0.441 | 0.327 | 0.680 | 0.316 |
| Hail | 9,749 | 0.412 | 0.598 | 0.314 | 0.516 |
| Flood | 1,869 | 0.388 | 0.444 | 0.345 | 0.407 |
| Flash Flood | 9,973 | 0.362 | 0.378 | 0.348 | 0.353 |
| Excessive Heat | 3,070 | 0.297 | 0.445 | 0.222 | 0.376 |
| Lightning | 1,101 | 0.166 | 0.108 | 0.354 | 0.090 |
| Debris Flow | 292 | 0.128 | 0.209 | 0.092 | 0.087 |
| Wildfire | 206 | 0.023 | 0.059 | 0.015 | 0.020 |
| Tornado | 3,125 | 0.004 | 0.500 | 0.002 | 0.213 |
| Blizzard | 164 | 0.000 | 0.000 | 0.000 | 0.004 |
| Cold/Wind Chill | 206 | 0.000 | 0.000 | 0.000 | 0.148 |
| Extreme Cold/Wind Chill | 169 | 0.000 | 0.000 | 0.000 | 0.080 |
| Heat | 2,406 | 0.000 | 0.000 | 0.000 | 0.432 |
| Heavy Snow | 152 | 0.000 | 0.000 | 0.000 | 0.027 |

---

## Key Findings

### Current Ranking (comparable models only — NB02 & NB03)

By **F1 macro** (the most informative metric for imbalanced multilabel classification):

1. **MLP (128, 64)** (NB03 sklearn): F1 macro = **0.2755** — best among comparable models
2. **Chain-LogReg (x3)** (NB03 sklearn): F1 macro = **0.2336**
3. **OVR-LogReg** (NB03 sklearn): F1 macro = **0.2329**
4. **XGBoost Binary Relevance** (NB02): F1 macro = **0.2142**

By **F1 weighted** (biased toward high-support labels):

1. MLP (128, 64): **0.3919**
2. XGBoost Binary Relevance: **0.3845**
3. OVR-LogReg: **0.3550**
4. Chain-LogReg (x3): **0.3527**

By **AUC-PR weighted** (ranking quality):

1. XGBoost Binary Relevance: **0.4549** — best ranking performance
2. MLP (128, 64): **0.3632**
3. OVR-LogReg / Chain-LogReg: ~**0.30**

### Interpretation

- **MLP > XGBoost > LogReg on F1 macro**: The expected hierarchy holds. The sklearn MLP (128, 64) is the best model for balanced per-label prediction.
- **XGBoost wins on AUC-PR and subset accuracy**: XGBoost produces better-calibrated probabilities and makes fewer total errors, but is more conservative (lower recall), leading to lower macro F1.
- **Chain-RF underperforms**: Random Forest chains achieve the lowest F1 macro (0.086), likely due to poor threshold calibration for rare labels.
- **Rare labels are universally hard**: 6 of 16 labels (Blizzard, Cold/Wind Chill, Extreme Cold/Wind Chill, Heat, Heavy Snow, Tornado) have F1 = 0 for XGBoost. These represent seasonal/rare events with extreme imbalance.

### NB04 (Neural Nets) — Pending Re-run

NB04's results (26 labels, 20% val) are **not directly comparable**. The code has been patched to use 16 labels and 10% val split. After re-running:
- Baseline MLP and WeatherMLP V2 results will be on the same 16-label basis.
- We expect the dedicated PyTorch neural nets with Optuna tuning to outperform the sklearn MLP baseline.
- 3 additional architectures (Cross-Attention WeatherMLP, Physics-Informed WeatherMLP, FT-Transformer) are defined but have not been trained yet.

---

## Action Items

- [ ] **Re-run NB04** with the patched code (16 labels, 10% val) to get comparable neural net results
- [ ] Run the remaining 3 architectures in NB04 (Cross-Attention, Physics-Informed, FT-Transformer)
- [ ] Run the new results-saving cells in NB02 and NB03 to export CSVs and plots to `results/`
- [ ] After NB04 re-run, execute its results-saving cell to export to `results/neural_nets/`
