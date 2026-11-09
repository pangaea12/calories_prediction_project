
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "calories.csv"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

df = pd.read_csv(DATA)
required = {"Gender","Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"}
missing_cols = required - set(df.columns)
if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

# Data quality and cleaning
quality = {
    "rows_before_cleaning": int(len(df)),
    "columns": list(df.columns),
    "missing_values_before": {k:int(v) for k,v in df.isna().sum().items()},
    "duplicate_rows_before": int(df.duplicated().sum())
}
df = df.drop_duplicates().copy()
df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
df = df[df["Gender"].isin(["male","female"])].copy()
df = df.dropna(subset=["Calories"])
# User_ID is an identifier, not a predictive feature
drop_cols = [c for c in ["User_ID"] if c in df.columns]
X = df.drop(columns=["Calories"] + drop_cols)
y = df["Calories"]
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

quality.update({
    "rows_after_cleaning": int(len(df)),
    "missing_values_after": {k:int(v) for k,v in df.isna().sum().items()},
    "duplicate_rows_after": int(df.duplicated().sum()),
    "identifier_columns_dropped": drop_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": "Calories"
})
# IQR outlier inspection; values are flagged, not automatically removed
outlier_info = {}
for col in numeric_features + ["Calories"]:
    q1, q3 = df[col].quantile([0.25,0.75])
    iqr = q3-q1
    lo, hi = q1-1.5*iqr, q3+1.5*iqr
    outlier_info[col] = {
        "lower_bound": float(lo), "upper_bound": float(hi),
        "outlier_count": int(((df[col] < lo) | (df[col] > hi)).sum())
    }
quality["iqr_outlier_inspection"] = outlier_info
(OUT/"data_quality.json").write_text(json.dumps(quality, indent=2))

# EDA charts
fig = px.histogram(df, x="Calories", nbins=40, title="Distribution of Calories Burnt")
fig.write_html(OUT/"eda_target_distribution.html", include_plotlyjs="cdn")
fig = px.histogram(df, x="Duration", color="Gender", marginal="box", title="Workout Duration by Gender")
fig.write_html(OUT/"eda_duration_by_gender.html", include_plotlyjs="cdn")
fig = px.scatter(df.sample(min(4000,len(df)), random_state=RANDOM_STATE), x="Duration", y="Calories",
                 color="Gender", trendline="ols", title="Workout Duration vs Calories Burnt")
fig.write_html(OUT/"eda_duration_calories.html", include_plotlyjs="cdn")
corr = df[numeric_features + ["Calories"]].corr(numeric_only=True)
fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
fig.write_html(OUT/"eda_correlation_heatmap.html", include_plotlyjs="cdn")
fig = px.scatter_matrix(df.sample(min(1500,len(df)), random_state=RANDOM_STATE),
                        dimensions=["Age","Weight","Duration","Heart_Rate","Body_Temp","Calories"],
                        color="Gender", title="Multivariate Feature Relationships")
fig.write_html(OUT/"eda_scatter_matrix.html", include_plotlyjs="cdn")
corr.to_csv(OUT/"correlation_matrix.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Exactly four requested regression models
model_specs = {
    "Linear Regression": (LinearRegression(), {
        "regressor__fit_intercept": [True, False]
    }, True),
    "KNN": (KNeighborsRegressor(), {
        "regressor__n_neighbors": [5, 7, 9],
        "regressor__weights": ["uniform", "distance"]
    }, True),
    "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), {
        "regressor__max_depth": [None, 8, 15],
        "regressor__min_samples_split": [2, 5]
    }, False),
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), {
        "regressor__n_estimators": [80],
        "regressor__max_depth": [None, 15],
        "regressor__min_samples_leaf": [1]
    }, False)
}
results, best_estimators, tuning_details = [], {}, {}
for name, (estimator, grid, scale_numeric) in model_specs.items():
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ], remainder="drop")
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error",
                          n_jobs=1, refit=True, verbose=0)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    cv_mae = -float(search.best_score_)
    results.append({
        "Model": name, "MAE": float(mae), "RMSE": rmse, "R2": float(r2),
        "Best_CV_MAE": cv_mae, "Best_Parameters": json.dumps(search.best_params_)
    })
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(best, MODELS/f"{safe_name}.joblib")
    best_estimators[name] = best
    tuning_details[name] = {
        "best_params": search.best_params_,
        "best_cv_mae": cv_mae,
        "best_cv_std": float(search.cv_results_["std_test_score"][search.best_index_])
    }
    pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": pred}).to_csv(
        OUT/f"predictions_{safe_name}.csv", index=False
    )

metrics = pd.DataFrame(results).sort_values("RMSE")
metrics.to_csv(OUT/"model_comparison.csv", index=False)
(OUT/"tuning_results.json").write_text(json.dumps(tuning_details, indent=2))
best_name = str(metrics.iloc[0]["Model"])
joblib.dump(best_estimators[best_name], MODELS/"best_model.joblib")
(OUT/"best_model.json").write_text(json.dumps({"best_model":best_name}, indent=2))

fig = px.bar(metrics, x="Model", y="RMSE", color="Model", title="Model Comparison – RMSE (lower is better)")
fig.write_html(OUT/"model_comparison.html", include_plotlyjs="cdn")
fig = go.Figure()
for name in best_estimators:
    p = pd.read_csv(OUT/f"predictions_{name.lower().replace(' ','_')}.csv")
    fig.add_trace(go.Scatter(x=p["Actual"], y=p["Predicted"], mode="markers", name=name, opacity=.5))
lo, hi = float(y_test.min()), float(y_test.max())
fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Ideal prediction"))
fig.update_layout(title="Actual vs Predicted Calories", xaxis_title="Actual", yaxis_title="Predicted")
fig.write_html(OUT/"actual_vs_predicted.html", include_plotlyjs="cdn")

report = f"""# Calories Burnt Prediction – Project Report

## 1. Introduction
This project predicts calories burnt during exercise using demographic and workout measurements. It is a supervised regression task.

## 2. Dataset
- Records after cleaning: {len(df):,}
- Target: Calories
- Predictors: {", ".join(X.columns)}
- Identifier excluded: {", ".join(drop_cols) if drop_cols else "None"}

## 3. Data preprocessing
Duplicate rows were removed, gender text was standardized, rows without a target were excluded, and missing-value imputers were included in the modelling pipelines. The uploaded dataset initially contained {quality['duplicate_rows_before']} duplicate rows and {sum(quality['missing_values_before'].values())} missing cells. IQR outliers were inspected and reported rather than automatically deleted, since unusual workout values may be genuine observations.

## 4. Exploratory data analysis
The project creates univariate, bivariate and multivariate Plotly charts. See the `outputs/eda_*.html` files and correlation matrix.

## 5. Methodology
An 80/20 train/test split was used with random_state=42. Three-fold cross-validation and GridSearchCV were used for hyperparameter tuning. The four models are Linear Regression, KNN, Decision Tree and Random Forest. Numeric features are scaled for Linear Regression and KNN; categorical Gender is one-hot encoded.

## 6. Evaluation
MAE measures average absolute prediction error, RMSE penalizes larger errors more strongly, and R² measures variance explained. Test-set results:

{metrics[["Model","MAE","RMSE","R2","Best_CV_MAE"]].to_markdown(index=False)}

Best model selected by lowest test RMSE: **{best_name}**.

## 7. Conclusion
Compare the four models using the supplied metrics and charts. The model with the lowest RMSE is saved as `models/best_model.joblib`. Predictions are estimates and should not be treated as medical measurements.
"""
(ROOT/"PROJECT_REPORT.md").write_text(report)
print(metrics.to_string(index=False))
print("\nBest model:", best_name)
print("Training complete.")
