# Calories Burnt Prediction - Machine Learning Project

A complete regression project using `calories.csv`, four requested ML algorithms, a dark Streamlit dashboard, Plotly EDA, model evaluation, saved pipelines and a written report.

## Project structure
```text
calories_prediction_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── data/calories.csv
├── models/                 # created by training
└── outputs/                # reports, charts, metrics and predictions
```

## Setup
Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the models
```bash
python train_model.py
```
Training performs data cleaning, missing-value handling within pipelines, duplicate removal, IQR outlier inspection, EDA chart generation, an 80/20 split, three-fold cross-validation and GridSearchCV tuning. The four and only four estimators are Linear Regression, KNN, Decision Tree and Random Forest. Models and result files are written to `models/` and `outputs/`.

## Launch dashboard
```bash
streamlit run app.py
```
Dashboard pages: Overview, EDA, Model Performance, Make a Prediction and Data Quality.

## Evaluation
Regression metrics: MAE, RMSE and R². Model comparison is in `outputs/model_comparison.csv`. The best model by lowest test RMSE is also saved as `models/best_model.joblib`.

## Notes
- `User_ID` is an identifier and is excluded from model features.
- IQR outliers are reported but not automatically removed because extreme workout values may be valid.
- Preprocessing is stored inside each fitted pipeline to avoid train/test leakage.
- Predictions are estimates and are not medical measurements
