# Chicago Crime: Predicting Domestic Incidents (Leakage-Free)

A capstone project that predicts whether a reported Chicago crime is classified as **Domestic (`True` or `False`)** using selected crime-related features. The project compares classical machine learning models and PyTorch deep learning architectures, with an emphasis on leakage prevention and chronological evaluation.

The final modeling pipeline uses three features:

- `Primary Type`
- `Location Description`
- `Beat`

The six models with supplied test metrics achieve approximately **83% accuracy** and **0.91 ROC-AUC** on the held-out 2025 test set. An additional **FT-Transformer** implementation is now included in `dl_model3.py`; its final metrics should be added to the comparison table after running the script.

## Project Goals

- Explore Chicago crime patterns through exploratory data analysis (EDA).
- Select informative features using mutual information and greedy forward selection.
- Prevent target and future-data leakage.
- Compare classical ML models with deep learning models.
- Evaluate with accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC.
- Use a shuffled-label sanity check to help verify that the evaluation pipeline is not producing artificially high scores.

## Dataset

**Source:** [Chicago Data Portal — Crimes - 2001 to Present](https://data.cityofchicago.org/)

Download the dataset and save it as:

```text
data/raw/Crimes_-_2001_to_Present_20260902.csv
```

The raw CSV is not stored in this repository because of its large file size.

## Methodology

### 1. Chronological split

The data is split by time without random shuffling.

| Split | Period | Purpose |
|---|---|---|
| Training | 2023 to June 2024 | Fit preprocessing and models |
| Validation | July to December 2024 | Feature selection, early stopping, and threshold tuning |
| Test | All of 2025 | Final evaluation; 236,927 rows |

### 2. Leakage audit

Potentially leaking or inappropriate fields are excluded, including:

- `Description`, which may explicitly contain domestic-related wording.
- `IUCR` and `FBI Code`, which may reveal closely related crime classifications.
- `Block`, IDs, and duplicate coordinate fields.
- `Updated On`, which can contain later record-update information.
- `Year`, which is not part of the final feature set.
- `Arrest`, the other outcome label.

Categorical encoders are fitted using training data only. The decision threshold is selected using validation data and is not tuned on the test set. The threshold objective maximizes `min(accuracy, balanced accuracy)`.

A shuffled-label sanity check is also used: when labels are randomized, test ROC-AUC should fall to around 0.5.

## Feature Selection

### Mutual information

Mutual information scores each feature independently.

| Feature | Mutual information |
|---|---:|
| Primary Type | 0.1340 |
| Location Description | 0.1014 |
| Longitude | 0.0855 |
| Latitude | 0.0854 |
| Beat | 0.0252 |
| Community Area | 0.0209 |
| Ward | 0.0182 |
| District | 0.0181 |
| IsWeekend | 0.0150 |
| DayOfWeek | 0.0085 |
| Hour | 0.0059 |
| Month | 0.0048 |
| LatLon_Missing | 0.0004 |

### Greedy forward selection

Features are added one at a time based on the increase in validation ROC-AUC. The search stops when the improvement from adding another feature is below 0.003 AUC.

| Step | Feature added | Validation ROC-AUC |
|---|---|---:|
| 1 | Primary Type | 0.8457 |
| 2 | Location Description | 0.9121 |
| 3 | Beat | 0.9178 |

The feature-selection reports are saved in `reports/`.

## Models

### Classical machine learning

| Model | Encoding |
|---|---|
| Logistic Regression | One-hot |
| Random Forest | Target encoding |
| HistGradientBoosting | Target encoding |
| XGBoost | One-hot |

### Deep learning

| Model | Implementation | Input representation |
|---|---|---|
| MLP with Embeddings | PyTorch (`dl_model.py`) | Learned categorical embeddings |
| FT-Transformer | PyTorch (`dl_model3.py`) | Categorical tokens, feature offsets, and Transformer encoder |

The FT-Transformer implementation builds category vocabularies from training data only; unseen categories map to index `0`. It uses validation ROC-AUC for early stopping and learning-rate scheduling, then chooses a decision threshold on the validation set.

**Repository status note:** `dl_model2.py` (the earlier Wide & Deep implementation) is not present in the current root file listing. Wide & Deep is therefore retained below as a previously reported experiment, not as a currently runnable script in this repository. Add its script back if you want the experiment to be reproducible from this repo.

## Reported Results

The table below contains the six models for which test metrics were supplied. These results use the full 2025 test set.

| Model | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP (Embeddings) | 0.57 | 0.8327 | 0.8336 | 0.5397 | 0.8350 | 0.6556 | 0.9136 |
| Logistic Regression | 0.53 | 0.8271 | 0.8276 | 0.5299 | 0.8285 | 0.6463 | 0.9094 |
| Random Forest | 0.51 | 0.8305 | 0.8299 | 0.5359 | 0.8290 | 0.6510 | 0.9103 |
| HistGradientBoosting | 0.55 | 0.8313 | 0.8306 | 0.5373 | 0.8294 | 0.6521 | 0.9114 |
| XGBoost | 0.52 | 0.8312 | 0.8311 | 0.5371 | 0.8309 | 0.6524 | 0.9125 |

- The MLP with embeddings has the highest reported ROC-AUC: **0.9136**.
- Wide & Deep has the highest reported accuracy: **83.32%**.
- Logistic Regression remains competitive with the more complex models.
- Precision is lower than recall, so false-positive predictions remain an important consideration.

The FT-Transformer test metrics are not listed here because no final metric values were supplied. Run `dl_model3.py` and add its measured results before comparing it quantitatively with the other models.

## Repository Structure

The current root-level files and folders include:

```text
CAPSTONE-PROJECT/
├── build_dataset.py       # Dataset preparation, leakage audit, splits, feature selection
├── data_utils.py          # Shared loading, threshold, metrics, and result helpers
├── ml_models.py            # Classical ML model training
├── dl_model.py             # MLP with embeddings
├── dl_model3.py            # FT-Transformer (PyTorch)
├── evaluate_ml.py          # Classical ML evaluation and chart generation
├── requirements.txt
├── data/
│   └── processed/          # Processed dataset files
├── notebooks/              # EDA and experiments
├── outputs/                # Results and visualizations
├── reports/                # Feature-selection reports
└── src/                    # Supporting source modules
```

The raw dataset and trained model artifacts may be excluded from Git because of their size. Folder contents can change as experiments are added.

## Setup

Python 3.11 is recommended. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Core dependencies include:

```text
numpy
pandas
scikit-learn>=1.3
matplotlib
seaborn
xgboost
torch
joblib
```

Use `scikit-learn >= 1.3` for `TargetEncoder`. For reproducibility, record the package versions used in your environment, for example with `pip freeze`.

## How to Run

Run commands from the repository root.

### 1. Build the processed dataset

Make sure the raw CSV is in `data/raw/`, then run:

```bash
python build_dataset.py
```

The default target is `Domestic`. The script also supports `--target Arrest`, but downstream modeling scripts are configured for `Domestic`.

### 2. Train the classical ML models

```bash
python ml_models.py
```

### 3. Train the deep learning models

```bash
python dl_model.py
python dl_model3.py
```

`dl_model.py` trains the MLP with embeddings. `dl_model3.py` trains the FT-Transformer and saves its result row using the shared result utilities.

### 4. Generate the final comparison

```bash
python evaluate_ml.py
```

Check `data_utils.py` and `evaluate_ml.py` if you change how result rows are saved. Ensure the final results file retains the deep-learning rows rather than replacing them with only the classical ML rows. The current `evaluate_ml.py` should be checked against the intended seven-model comparison before treating its output as the complete final table.

Trained models are not guaranteed to be present in the repository; regenerate them by running the relevant training scripts.

## Notebooks and Outputs

The `notebooks/` directory contains exploratory analysis, baseline-model experiments, feature-selection work, and earlier ML/DL comparisons. The `outputs/` directory contains generated charts and result files, while `reports/` contains feature-selection evidence.

The final leakage-free results should be distinguished from earlier notebook experiments if their data splits or evaluation procedures differ.

## Limitations

- The model predicts the `Domestic` label recorded in historical reports; it does not independently establish the nature of an incident.
- Historical crime records can contain reporting and geographic biases.
- The selected features indicate associations, not causes.
- Results on Chicago's 2025 records may not generalize to other cities or future periods.
- Predictions should not be used as the sole basis for enforcement or other high-impact decisions.

## Author

**P. Aravinda Valli**

## Repository

[CAPSTONE-PROJECT on GitHub](https://github.com/PadmasaleAravindaValli/CAPSTONE-PROJECT)
