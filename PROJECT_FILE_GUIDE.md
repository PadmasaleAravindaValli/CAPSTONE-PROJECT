# Chicago Crime Project: Complete File Guide & Short Naming Index

This document provides a clear, simple guide explaining **what is inside every notebook and Python file** in your project, along with **intuitive short renames** so you can easily navigate and explain them to your project review committee ("Sir").

---

## 1. Jupyter Notebooks Directory (`notebooks/`)

| # | Current Notebook Name | Recommended Short Rename | What is Inside in Simple Words | Key Outputs Rendered |
|---|----------------------|--------------------------|--------------------------------|----------------------|
| **1** | `crime_analysis_EDA.ipynb` | `01_EDA.ipynb` | **Exploratory Data Analysis**: 9 visual charts showing crime patterns by hour, month, day of week, arrest rates, and domestic violence distribution. | 9 EDA graphs (`outputs/EDA_graphs/`) |
| **2** | `02_Crime_Feature_Selection.ipynb` | `02_Feature_Selection.ipynb` | **Data Audit & Leakage Prevention**: Identifies columns that leak future information (like text description) and drops them. | Initial correlation analysis |
| **3** | `Feature_Engineering_and_Dataset_Creation.ipynb` | `03_Build_Two_Datasets.ipynb` | **Two-Stage Dataset Creation**: Builds `dataset1_important_features.csv` (14 core features) and `dataset2_enhanced_features.csv` (29 enhanced features). | Saved processed datasets in `data/processed/` |
| **4** | `Dataset1_ML_and_DL_Modeling.ipynb` | `04_Dataset1_Models.ipynb` | **Base Model Benchmarks**: Trains Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, and PyTorch Deep MLP on Dataset 1. | ROC curves, PR curves, Confusion matrices |
| **5** | `Dataset2_ML_and_DL_Modeling.ipynb` | `05_Dataset2_Models.ipynb` | **Enhanced Model Benchmarks & Ablation**: Trains all 5 models on Dataset 2 and demonstrates empirical lift over Dataset 1. | Lift comparison charts, Confusion matrices |
| **6** | `Feature_Ablation_and_Comparison.ipynb` | `06_Feature_Ablation.ipynb` | **Feature Ablation & Importance**: Progressive 7-stage ablation, Mutual Information ranking, and Leave-One-Group-Out penalty analysis. | Progression charts & LOGO impact plots |
| **7** | `06_ML_Models.ipynb` | `archive_01_ML_Models.ipynb` | *Earlier baseline classical ML notebook*. | Baseline classifier metrics |
| **8** | `08_DL_MLP_Model.ipynb` | `archive_02_DL_MLP.ipynb` | *Earlier prototype of PyTorch MLP with embeddings*. | MLP training loss curves |
| **9** | `11_DL_FTTransformer_Model.ipynb` | `archive_03_DL_Transformer.ipynb` | *Deep learning FT-Transformer (Feature Tokenizer Transformer) prototype*. | Transformer training metrics |
| **10** | `Base_Models.ipynb` | `archive_04_Base_Models.ipynb` | *Initial exploratory script comparing standard classifiers*. | Initial comparison logs |
| **11** | `Chicago_Crime_Base_Models.ipynb` | `archive_05_Chicago_Baselines.ipynb`| *Early prototype baseline runs on raw Chicago crime files*. | Early accuracy charts |
| **12** | `chicago_domestic_classifier.ipynb` | `archive_06_Domestic_Classifier.ipynb` | *Consolidated single-notebook pipeline prototype*. | Prototype prediction reports |
| **13** | `crime_analysis.ipynb` | `archive_07_Crime_Analysis.ipynb` | *Early statistical summary of crime volume and categories*. | Volume tables |
| **14** | `05_Build_Dataset.ipynb` | `archive_08_Build_Dataset.ipynb` | *Early single-dataset cleaning script*. | Initial train/val split files |

> **Tip for Sir**: When demonstrating notebooks to your guide, focus on:
> 1. `01_EDA.ipynb` (to show data exploration)
> 2. `03_Build_Two_Datasets.ipynb` (to show clean data engineering with zero leakage)
> 3. `04_Dataset1_Models.ipynb` & `05_Dataset2_Models.ipynb` (to show ML & PyTorch Deep Learning benchmarks)
> 4. `06_Feature_Ablation.ipynb` (to prove why each feature was included)

---

## 2. Root Python Scripts (`./`)

| Current File Name | Recommended Short Name | What is Inside in Simple Words | How to Run It |
|-------------------|------------------------|--------------------------------|---------------|
| `app.py` | `app.py` | **Interactive Streamlit Web Dashboard**: Full graphical interface for police dispatch, including Live Risk Predictor, Ablation charts, Chicago Heatmap, and "What-If" Counterfactual Simulator. | `streamlit run app.py` |
| `api.py` | `api.py` | **FastAPI Production REST API**: Real-time microservice providing `/predict`, `/predict/batch`, `/metrics`, and `/fairness` endpoints with Swagger docs. | `uvicorn api:app --reload` |
| `build_two_datasets.py` | `01_build_data.py` | **Two-Stage Dataset Generator**: Loads raw Chicago crime CSV and generates both `dataset1` (14 features) and `dataset2` (29 features) with temporal splits. | `python build_two_datasets.py` |
| `data_utils.py` | `data_utils.py` | **Shared Helper Library**: Common utilities for loading splits, calculating metrics, and tuning thresholds (imported by other scripts). | Imported by scripts |
| `run_ablation_experiment.py`| `02_run_ablation.py` | **Model Ablation Script**: Compares Dataset 1 vs Dataset 2 on XGBoost, producing `ablation_d1_vs_d2.csv` and chart. | `python run_ablation_experiment.py` |
| `run_shap_explainability.py`| `03_run_shap.py` | **Explainable AI (SHAP)**: Generates TreeSHAP summary beeswarm and feature importance bar charts proving top drivers. | `python run_shap_explainability.py` |
| `generate_crime_map.py` | `04_build_map.py` | **Geospatial Heatmap Generator**: Generates interactive Folium Chicago map HTML with domestic hotspot clusters. | `python generate_crime_map.py` |
| `ml_models.py` | `archive_ml_models.py` | *Earlier classical ML baseline script*. | `python ml_models.py` |
| `dl_model.py` | `archive_dl_mlp.py` | *Earlier PyTorch MLP training script*. | `python dl_model.py` |
| `dl_model3.py` | `archive_dl_transformer.py` | *Earlier PyTorch FT-Transformer script*. | `python dl_model3.py` |
| `evaluate_ml.py` | `archive_evaluate_ml.py`| *Earlier evaluation and classification report script*. | `python evaluate_ml.py` |
| `chicago_domestic_classifier.py` | `archive_classifier.py` | *Earlier end-to-end classification pipeline*. | `python chicago_domestic_classifier.py` |
| `build_dataset.py` | `archive_build_dataset.py` | *Earlier single-dataset processing script*. | `python build_dataset.py` |

---

## 3. Modular Source Code Directory (`src/`)

The `src/` folder contains clean, numbered, independent execution modules:

| Script Name | Purpose in Simple Words | Outputs Generated |
|-------------|-------------------------|-------------------|
| `01_build_two_datasets.py` | Creates Dataset 1 (Core 14) and Dataset 2 (Enhanced 29) | `data/processed/*.csv` |
| `02_train_and_evaluate_models.py` | Benchmarks ML models on temporal splits | Baseline metric tables |
| `03_run_ablation_study.py` | Runs comparative ablation of Dataset 1 vs Dataset 2 | `outputs/tables/ablation_d1_vs_d2.csv` |
| `04_run_shap_explainability.py` | Computes TreeSHAP importance for feature explainability | `outputs/figures/shap_*.png` |
| `05_generate_crime_map.py` | Builds interactive Folium geospatial heatmap | `outputs/maps/chicago_domestic_heatmap.html` |
| `06_fairness_and_bias_audit.py` | Audits algorithmic fairness across 22 Chicago Police Districts | `outputs/tables/district_fairness_audit.csv` |
| `07_model_calibration.py` | Platt Scaling & Isotonic probability calibration | `outputs/tables/calibration_metrics.csv` |
| `08_stacking_ensemble.py` | Heterogeneous Stacking Meta-Ensemble (LR + HGB + XGB) | `outputs/tables/stacking_ensemble_results.csv` |
| `09_feature_ablation_and_comparison.py` | Progressive 7-stage ablation & Leave-One-Group-Out analysis | `outputs/tables/feature_ablation_progression.csv` |

---

## 4. Key Metrics Quick-Reference (To Speak Out Loud to Sir)

* **Dataset Size**: **756,085 crime records** across Chicago (2021 to 2025).
* **Positive Class Prevalence**: ~**18% domestic incidents** (handled using balanced class weights & threshold tuning).
* **Champion Model**: **XGBoost on Dataset 2**:
  * **ROC-AUC**: **0.9152**
  * **Recall (Domestic Detection Rate)**: **87.18%**
  * **Balanced Accuracy**: **83.55%**
* **Top 3 Predictive Drivers**:
  1. `Primary Crime Type Risk` (e.g., Stalking = 77.7%, Battery = 39.5%, Theft = 1.5%)
  2. `Location Domestic Risk` (e.g., Apartment = 46.8%, Residence = 44.5%, Street = 7.1%)
  3. `Venue Setting` (`Is_Residential` private home vs `Is_Public_Space` outdoor street)
