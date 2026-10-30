# Section 4: Experimental Results and Comparative Analysis

## 4.1 Evaluation Protocol and Locked Chronological Test Partition
To ensure rigorous real-world evaluation and avoid the optimistic bias of random train-test splits in temporal crime data, all models were evaluated under a strict chronological partition:
- **Training Set**: 1 January 2023 to 30 June 2024 (388,307 records) — used exclusively for parameter optimization, categorical encoding vocabularies, and Bayesian smoothing statistics.
- **Validation Set**: 1 July 2024 to 31 December 2024 (131,449 records) — used strictly for hyperparameter selection, early stopping, and decision-threshold tuning.
- **Locked Test Set**: 1 January 2025 to 31 December 2025 (236,329 records) — strictly reserved as an unseen future test cohort.

Decisions, thresholds, and feature selection were finalized without consulting test set labels.

---

## 4.2 Spatio-Temporal Forecasting: Recurrent vs. Tabular Models (Pipeline 1)
In the spatio-temporal forecasting formulation, crime incidents were aggregated into 274 Police Beats across 36 consecutive months, creating 9,900 Beat-Month panels with 5-level violent crime burden targets ($[16.0, 23.0, 30.0, 39.0]$ frozen quintiles).

Table 1 summarizes the forecasting performance on 3,300 locked 2025 test sequences:

### Table 1: Spatio-Temporal Sequence Forecasting Performance (2025 Locked Test)
| Model Architecture | Feature Set | Test Accuracy | Macro F1-Score | Weighted F1 | Macro OvR ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|
| **Majority Baseline** | Naive | 0.2215 | 0.0725 | 0.0803 | 0.5000 |
| **Logistic Regression (lag $t-1$)** | Baseline (6 Groups) | 0.4797 | 0.4697 | 0.4741 | 0.7977 |
| **Logistic Regression (lag $t-1$)** | Enhanced (11 Features)| 0.4770 | 0.4696 | 0.4739 | 0.8012 |
| **LSTM Recurrent Model** | Baseline (6 Groups) | 0.5124 | 0.5071 | 0.5107 | 0.8218 |
| **LSTM Recurrent Model** | Enhanced (11 Features)| 0.4955 | 0.4969 | 0.5021 | 0.8101 |
| **GRU Recurrent Model** | Baseline (6 Groups) | **0.5188** | **0.5111** | **0.5176** | **0.8224** |
| **GRU Recurrent Model** | Enhanced (11 Features)| 0.5185 | 0.5095 | 0.5154 | 0.8200 |

Both recurrent architectures (GRU and LSTM) substantially outperformed the tabular and naive baselines, with GRU achieving the highest Macro F1 (0.5111) and ROC-AUC (0.8224), proving that modeling multi-month temporal dependencies ($[t-3, t-2, t-1] \to t$) provides superior predictive capability over isolated single-month lags.

---

## 4.3 Incident-Level Domestic Classification & Threshold Optimization (Pipeline 2)
For incident-level classification on 236,927 test records, six machine learning and deep learning models were evaluated. Because domestic incidents comprise only 18.1% of all reported crimes, a default classification cutoff of $0.50$ is suboptimal. The optimal decision threshold $t^*$ was determined on validation predictions to maximize $\min(\text{Accuracy}, \text{Balanced Accuracy})$, and subsequently applied to the 2025 test set.

### Table 2: Model Performance on 2025 Held-Out Test Set (236,927 Incidents)
| Model | Model Family | Selected Cutoff $t^*$ | Test Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Test ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MLP (Entity Embeddings)** | Deep Learning | 0.560 | 0.8319 | **0.8339** | 0.5381 | **0.8371** | 0.6551 | **0.9133** |
| **FT-Transformer** | Deep Learning | 0.590 | **0.8325** | 0.8334 | **0.5393** | 0.8348 | **0.6553** | 0.9130 |
| **XGBoost** | Classical ML | 0.520 | 0.8303 | 0.8307 | 0.5354 | 0.8315 | 0.6513 | 0.9119 |
| **HistGradientBoosting** | Classical ML | 0.550 | 0.8313 | 0.8306 | 0.5373 | 0.8294 | 0.6521 | 0.9114 |
| **Random Forest (200 trees)** | Classical ML | 0.510 | 0.8305 | 0.8299 | 0.5359 | 0.8290 | 0.6510 | 0.9103 |
| **Logistic Regression** | Classical ML | 0.530 | 0.8271 | 0.8276 | 0.5299 | 0.8285 | 0.6463 | 0.9094 |

Neural architectures with learned categorical embeddings (MLP and FT-Transformer) demonstrated a slight predictive advantage, capturing subtle non-linear interactions across high-cardinality categorical variables like `Beat` (274 categories) and `Location Description` (130 categories).

---

## 4.4 Shuffled-Label Sanity Check & Target Leakage Audit
To confirm that the high discriminative power (ROC-AUC $> 0.91$) reflects genuine underlying patterns rather than latent target leakage:
1. All training labels $y_{\text{train}}$ were randomly permuted to destroy the relationship between predictors and targets.
2. A model was fitted to the shuffled training data and evaluated against the unchanged test set.
3. The resulting test ROC-AUC dropped to **0.5319** ($\approx 0.500$), demonstrating that without real predictive signals, the pipeline collapses to random chance. This verifies the complete absence of target leakage.
