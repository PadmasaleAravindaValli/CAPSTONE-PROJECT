# Chicago Crime: Predicting Domestic Incidents Using Machine Learning and Deep Learning

A capstone project that analyses Chicago crime data and predicts whether a reported crime incident is classified as domestic (`Domestic = True/False`). The project uses exploratory data analysis, feature selection, classical machine learning, and deep-learning models to study patterns in reported crime incidents.

The main focus is to build a reliable prediction pipeline, prevent data leakage, and compare different models using a separate future time period for testing.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Methodology](#project-methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Selection](#feature-selection)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Preventing Data Leakage](#preventing-data-leakage)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

Crime is a complex social issue influenced by several factors, including crime type, location, and the circumstances in which an incident is reported. Analysing crime records can help identify patterns and understand how different features relate to reported incidents.

This project uses the Chicago Crimes dataset to predict the `Domestic` target variable. It combines data preprocessing, exploratory data analysis, feature selection, and predictive modelling in a structured workflow.

The project compares five machine-learning models with two deep-learning models. Special attention is given to avoiding data leakage, selecting useful features, and evaluating model performance on data from a later time period.

**Project title:** Computational Analysis of Crime and Social System Using Machine Learning

**Prediction task:** Domestic incident classification

**Target variable:** `Domestic`

**Problem type:** Binary classification

**Dataset source:** [Chicago Data Portal — Crimes](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

## Objectives

- Analyse Chicago crime records and understand crime-related patterns.
- Clean and preprocess the data for predictive modelling.
- Explore crime distributions across different categories and locations.
- Identify informative features using feature-selection techniques.
- Develop and compare five machine-learning models.
- Develop and evaluate two deep-learning models.
- Prevent data leakage during preprocessing, feature selection, and evaluation.
- Compare models using multiple classification metrics.
- Evaluate predictive performance on a separate chronological test set.

## Dataset

The project uses the Chicago Crimes dataset available through the Chicago Data Portal.

The dataset contains reported crime records with information about incidents, their locations, and other crime-related characteristics.

Examples of available fields include:

- `Date` — date and time associated with the incident.
- `Primary Type` — reported crime category.
- `Location Description` — description of the incident location.
- `Beat` — police beat associated with the incident.
- `District` — police district.
- `Community Area` — community area associated with the incident.
- `Latitude` and `Longitude` — geographic coordinates.
- `Arrest` — whether an arrest was recorded.
- `Domestic` — whether the incident was classified as domestic.

The original dataset contains additional columns, but not all are suitable for prediction. Some fields can reveal information related to the target or introduce data leakage, so they are excluded from the final selected feature set.

### Selected Features

The final selected feature set contains three categorical features.

| Feature | Description |
|---|---|
| `Primary Type` | Category of the reported crime |
| `Location Description` | Description of where the incident occurred |
| `Beat` | Police beat associated with the incident |

**Target variable:** `Domestic`

These features are used to train the models to distinguish between domestic and non-domestic incident classifications.

### Dataset Availability

Download the dataset from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2).

Place the downloaded CSV at the expected location:

`data/raw/Crimes_-_2001_to_Present_20260902.csv`

The raw dataset is large and is not included in the GitHub repository. It must be downloaded separately before running the complete preprocessing pipeline.

## Project Methodology

The project follows these stages:

1. **Data Collection:** Obtain the Chicago Crimes dataset.
2. **Data Preprocessing:** Prepare the records and handle data-quality issues.
3. **Exploratory Data Analysis:** Study crime distributions, temporal patterns, and location-based patterns.
4. **Feature Selection:** Identify useful predictors using mutual information and forward feature selection.
5. **Data Splitting:** Separate training, validation, and testing data chronologically.
6. **Machine-Learning Model Development:** Train five classical machine-learning models.
7. **Deep-Learning Model Development:** Train an MLP with Embeddings and an FT-Transformer.
8. **Model Evaluation:** Compare model performance using multiple classification metrics.
9. **Result Analysis:** Analyse the strengths and limitations of the models and generate comparison results.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to understand the dataset before model training.

The project notebooks explore crime distributions, temporal patterns, geographic patterns, and data quality.

The analysis includes:

- Distribution of crime types.
- Crime patterns across days and times.
- Crime counts across locations and police beats.
- Missing values and data-quality checks.
- Relationships between candidate features and the target variable.
- Distribution of domestic and non-domestic incidents.

The notebooks contain visualisations and exploratory experiments that help understand the dataset and prepare it for feature selection and model development.

Generated graphs and other outputs are organised under the project's output directories.

## Feature Selection

Feature selection helps identify informative input variables while reducing unnecessary features.

### 1. Mutual Information

Mutual information estimates how much information an individual feature provides about the target variable.

The associated report is:

`reports/mutual_info_domestic.csv`

The feature-selection experiments examine variables such as `Primary Type`, `Location Description`, `Longitude`, `Latitude`, and `Beat`.

### 2. Forward Feature Selection

Forward feature selection adds features incrementally and evaluates their contribution using validation ROC-AUC.

The associated report is:

`reports/feature_selection_domestic.csv`

The recorded selection results are:

| Step | Feature Added | Validation ROC-AUC |
|---|---|---:|
| 1 | `Primary Type` | 0.8457 |
| 2 | `Location Description` | 0.9121 |
| 3 | `Beat` | 0.9178 |

The final selected feature set consists of `Primary Type`, `Location Description`, and `Beat`.

Feature selection uses training and validation data rather than the final test set.

## Machine Learning Models

Five machine-learning models are included in the final comparison.

| Model | Description |
|---|---|
| Logistic Regression | A linear classification algorithm used as a baseline model. |
| Random Forest | An ensemble model that combines predictions from multiple decision trees. |
| HistGradientBoosting | A gradient-boosting model that learns patterns through sequential decision-tree training. |
| XGBoost | A gradient-boosting algorithm designed for efficient and accurate prediction. |
| LightGBM | A gradient-boosting framework designed for efficient training on structured datasets. |

These models provide different approaches to learning patterns in categorical crime data.

The classical models use suitable feature encoding and preprocessing methods, depending on the implementation.

## Deep Learning Models


Two deep-learning models are included in the project.

### 1. MLP with Embeddings

## Reported Results


The Multilayer Perceptron (MLP) with Embeddings represents categorical input features using learned numerical embeddings. These representations are passed through neural-network layers to predict whether a reported incident is classified as domestic.

The implementation is available in:

`dl_model.py`

- The MLP with embeddings has the highest reported ROC-AUC: **0.9136**.
- Logistic Regression remains competitive with the more complex models.
- Precision is lower than recall, so false-positive predictions remain an important consideration.

### 2. FT-Transformer

FT-Transformer is a transformer-based architecture designed for tabular data. It uses attention mechanisms to learn relationships between input features and generate predictions.

The implementation is available in:

`dl_model3.py`

Both deep-learning models use PyTorch. Their results are compared with the classical machine-learning models to understand how neural-network approaches perform on the selected crime features.

## Preventing Data Leakage

Data leakage occurs when a model receives information during training that would not legitimately be available at prediction time. Leakage can lead to misleadingly high evaluation results.

The project applies several safeguards:

- Excludes `Description`, which can reveal information directly related to domestic-incident classification.
- Excludes other potentially problematic fields from the final selected feature set.
- Uses a chronological split instead of randomly mixing records across time.
- Fits preprocessing and encoding steps using training data only, where applicable.
- Performs feature selection using training and validation data.
- Selects classification thresholds using validation data rather than the final test set.
- Uses a shuffled-label sanity check to investigate whether the model depends on genuine target-related patterns.

### Chronological Data Split

| Partition | Time Period |
|---|---|
| Training | 2023 to June 2024 |
| Validation | July to December 2024 |
| Testing | January to December 2025 |

The reported test set contains **236,927 records**.

This chronological approach evaluates the models on a later period than the training data, providing a more realistic assessment of their ability to generalise over time.

## Experimental Results

The performance of five machine-learning models and two deep-learning models was compared using the held-out test dataset.

The evaluation includes accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC. Classification thresholds are selected using validation data.

### Performance Comparison of All Seven Models

| Category | Model | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ML | LightGBM | 0.69 | 0.8669 | 0.8092 | 0.6335 | 0.7160 | 0.6723 | 0.9114 |
| ML | Logistic Regression | 0.53 | 0.8271 | 0.8276 | 0.5299 | 0.8285 | 0.6463 | 0.9094 |
| ML | Random Forest | 0.51 | 0.8305 | 0.8299 | 0.5359 | 0.8290 | 0.6510 | 0.9103 |
| ML | HistGradientBoosting | 0.55 | 0.8313 | 0.8306 | 0.5373 | 0.8294 | 0.6521 | 0.9114 |
| ML | XGBoost | 0.52 | 0.8312 | 0.8311 | 0.5371 | 0.8309 | 0.6524 | 0.9125 |
| DL | MLP with Embeddings | 0.55 | 0.8318 | 0.8331 | 0.5380 | 0.8352 | 0.6545 | 0.9130 |
| DL | FT-Transformer | 0.57 | 0.8340 | 0.8327 | 0.5423 | 0.8306 | 0.6562 | 0.9131 |

**Test dataset:** 236,927 records, January–December 2025.

**Results source:** `outputs/leakfree_results.csv`

*The values above represent the recorded model results. Confirm that the results file contains these exact values before publishing the README or using the numbers in a research paper.*

### Results Analysis

- **Highest accuracy:** LightGBM achieved 86.69% accuracy.
- **Highest balanced accuracy:** MLP with Embeddings achieved 83.31% balanced accuracy.
- **Highest precision:** LightGBM achieved 63.35% precision.
- **Highest recall:** MLP with Embeddings achieved 83.52% recall.
- **Highest F1-score:** LightGBM achieved an F1-score of 67.23%.
- **Highest ROC-AUC:** FT-Transformer achieved a ROC-AUC of 91.31%, closely followed by the MLP with Embeddings at 91.30%.

LightGBM achieved the highest accuracy and F1-score among the seven models. However, its balanced accuracy and recall were lower than those of several other models. This shows why accuracy alone is not enough to compare models, especially when the target classes are imbalanced.

The MLP with Embeddings achieved the highest balanced accuracy and recall, while FT-Transformer achieved the highest ROC-AUC. The results indicate that different models have different strengths depending on the evaluation metric.

Overall, the comparison highlights the importance of evaluating several metrics instead of selecting a model based only on accuracy.

These models identify statistical patterns in historical crime records. Their predictions do not establish the causes of domestic incidents and should not be treated as proof of wrongdoing by any individual.

## Project Structure

The repository contains Python scripts, Jupyter notebooks, reports, and generated outputs.

```text
CAPSTONE-PROJECT/
│
├── data/
│   └── processed/
│
├── notebooks/
│   ├── crime_analysis_EDA.ipynb
│   ├── crime_analysis.ipynb
│   ├── Base_Models.ipynb
│   ├── Chicago_Crime_Base_Models.ipynb
│   ├── 02_Crime_Feature_Selection.ipynb
│   ├── 03_Leakage_Safe_Cluster_Feature_Selection.ipynb
│   ├── 04_Chicago_Selected_Feature_Models.ipynb
│   ├── Chicago_Crime_5ML_2DL_Balanced.ipynb
│   ├── Chicago_Crime_Clean_ML_DL.ipynb
│   └── fixed_code.ipynb
│
├── outputs/
│   ├── EDA_graphs/
│   └── leakfree_results.csv
│
├── reports/
│   ├── mutual_info_domestic.csv
│   └── feature_selection_domestic.csv
│
├── build_dataset.py
├── data_utils.py
├── ml_models.py
├── dl_model.py
├── dl_model3.py
├── evaluate_ml.py
├── requirements.txt
├── .gitignore
└── README.md
```

*This is a representative structure of the main project files. Actual notebook names and generated directories may differ depending on the current repository contents and local experiments.*

### Main Python Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Prepares the data, performs preprocessing and leakage checks, creates chronological splits, and supports feature selection. |
| `data_utils.py` | Provides shared data-processing, evaluation, threshold-selection, and results functions. |
| `ml_models.py` | Implements the classical machine-learning experiments. |
| `dl_model.py` | Implements the MLP with Embeddings. |
| `dl_model3.py` | Implements the FT-Transformer model. |
| `evaluate_ml.py` | Generates evaluation outputs and classical-model comparisons. |
| `requirements.txt` | Lists Python package dependencies. |

The notebooks contain exploratory data analysis, baseline models, feature-selection experiments, and additional modelling experiments. The final reported comparison should be distinguished from earlier exploratory results.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Deep Learning:** PyTorch
- **Data Visualisation:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Development Tools:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

## Installation

### Prerequisites

- Python compatible with the versions required by the project dependencies.
- Git.
- The Chicago Crimes CSV dataset.
- Sufficient memory and storage to process the dataset.

### 1. Clone the Repository

```bash
git clone https://github.com/PadmasaleAravindaValli/CAPSTONE-PROJECT.git
cd CAPSTONE-PROJECT
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the installed package versions are compatible with the project scripts and notebooks.

## How to Run

Run the commands from the project root directory. The required dataset must be downloaded and placed at the expected path before running the data pipeline.

### Step 1: Prepare the Required Directories

```powershell
New-Item -ItemType Directory -Force models, reports, outputs, data/processed
```

Create `data/raw/` as well if it does not already exist.

### Step 2: Prepare the Dataset

Download the Chicago Crimes CSV from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2).

Place it in the expected raw-data directory, then run:

```bash
python build_dataset.py
```

### Step 3: Run the Machine-Learning Experiments

```bash
python ml_models.py
```

This runs the classical-model experiments supported by the script. Confirm that the LightGBM experiment is included in the current implementation if you want to reproduce all five machine-learning rows in the results table.

### Step 4: Run the MLP with Embeddings

```bash
python dl_model.py
```

This runs the MLP deep-learning experiment.

### Step 5: Run FT-Transformer

```bash
python dl_model3.py
```

This runs the FT-Transformer experiment using the prepared data and the required dependencies.

### Step 6: Generate Evaluation Results

```bash
python evaluate_ml.py
```

This runs the evaluation script for the models supported by its implementation. Verify which model results it includes before assuming it regenerates the complete seven-model comparison.

**Note:** `data_utils.py` provides shared functions imported by other scripts and is not intended to be run independently.

The exact execution order and output locations should follow the current implementation of the repository scripts and notebooks.

## Evaluation Metrics

The project uses the following metrics to compare model performance:

- **Accuracy:** The proportion of all predictions that are correct.
- **Balanced Accuracy:** The average recall across the two classes.
- **Precision:** The proportion of predicted domestic incidents that are actually domestic.
- **Recall:** The proportion of actual domestic incidents correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well a model distinguishes positive cases from negative cases across different classification thresholds.

Using multiple metrics provides a more complete understanding of model performance than accuracy alone. This is particularly important when the target classes are imbalanced.

## Limitations

- The project uses reported Chicago crime data, so the results may not generalise to other cities.
- Reported crime records do not necessarily represent every incident that occurs.
- Predictive relationships do not establish the causes of domestic incidents.
- Changes in reporting practices and data distributions can affect model performance.
- Imbalanced target classes can affect the interpretation of accuracy and other metrics.
- Predictions should not be treated as proof of wrongdoing or used as the sole basis for decisions about individuals.
- The raw dataset is large and must be downloaded separately.
- Earlier notebook experiments may produce results that differ from the final leakage-free pipeline.
- Reproducing all seven results requires the corresponding model implementations, compatible dependencies, and consistent data-processing settings.

## Future Improvements

- Evaluate models on additional future time periods.
- Improve model reproducibility by recording package versions, parameters, and random seeds.
- Investigate model calibration and prediction confidence.
- Improve interpretability using appropriate feature-importance and explanation techniques.
- Test additional leakage-safe features.
- Organise generated graphs, trained models, and evaluation outputs consistently.
- Compare models under consistent preprocessing and validation procedures.
- Evaluate whether the observed performance remains stable as new crime records become available.

## Author

**P. Aravinda Valli**

GitHub: [PadmasaleAravindaValli](https://github.com/PadmasaleAravindaValli)

Project Repository: [CAPSTONE-PROJECT](https://github.com/PadmasaleAravindaValli/CAPSTONE-PROJECT)
