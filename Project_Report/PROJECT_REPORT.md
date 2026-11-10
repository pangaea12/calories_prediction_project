# Calories Burnt Prediction – Project Report

## 1. Introduction
This project predicts calories burnt during exercise using demographic and workout measurements. It is a supervised regression task.

## 2. Dataset
- Records after cleaning: 15,000
- Target: Calories
- Predictors: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
- Identifier excluded: User_ID

## 3. Data preprocessing
Duplicate rows were removed, gender text was standardized, rows without a target were excluded, and missing-value imputers were included in the modelling pipelines. The uploaded dataset initially contained 0 duplicate rows and 0 missing cells. IQR outliers were inspected and reported rather than automatically deleted, since unusual workout values may be genuine observations.

## 4. Exploratory data analysis
The project creates univariate, bivariate and multivariate Plotly charts. See the `outputs/eda_*.html` files and correlation matrix.

## 5. Methodology
An 80/20 train/test split was used with random_state=42. Five-fold cross-validation and GridSearchCV were used for hyperparameter tuning. The four models are Linear Regression, KNN, Decision Tree and Random Forest. Numeric features are scaled for Linear Regression and KNN; categorical Gender is one-hot encoded.

## 6. Evaluation
MAE measures average absolute prediction error, RMSE penalizes larger errors more strongly, and R² measures variance explained. Test-set results:

| Model             |     MAE |     RMSE |       R2 |   Best_CV_MAE |
|:------------------|--------:|---------:|---------:|--------------:|
| Random Forest     | 1.71333 |  2.68188 | 0.998218 |       2.05224 |
| KNN               | 3.44876 |  4.69309 | 0.994543 |       3.78215 |
| Decision Tree     | 3.41436 |  5.17816 | 0.993356 |       3.86361 |
| Linear Regression | 8.44151 | 11.4889  | 0.967294 |       8.31122 |

Best model selected by lowest test RMSE: **Random Forest**.

## 7. Conclusion
Compare the four models using the supplied metrics and charts. The model with the lowest RMSE is saved as `models/best_model.joblib`. Predictions are estimates and should not be treated as medical measurements.
