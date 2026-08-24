# Used Vehicle Price Prediction

A machine learning project that predicts the selling price of used vehicles based on attributes such as manufacturer, model, year, mileage, condition, and body type. The project covers the full pipeline: exploratory data analysis, missing-data treatment, categorical encoding, model comparison, hyperparameter tuning, and validation.

## Overview

The dataset used is the [Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) set, containing 558,837 records of vehicle sales with 16 attributes including make, model, trim, body type, transmission, condition, odometer reading, color, and selling price.

The goal is to build a regression model that accurately predicts `sellingprice` from the vehicle's characteristics, while carefully avoiding data leakage and validating that the final model generalizes well to unseen data.

## Data Preparation

### Missing values

Most columns had a small proportion of missing values (under 3%). The exception was `transmission`, with 11.69% missing — high enough to require a dedicated strategy rather than simply dropping rows.

An initial attempt used a Logistic Regression classifier to predict the missing `transmission` values from `make`, `year`, `trim`, `model`, and `body`. This produced a seemingly excellent accuracy of 96.49%, but a precision of 0.0 — revealing that the model was simply predicting the majority class (automatic) every time, masked by a strong class imbalance:

![Transmission distribution before imputation](nb_images/transmission_before_imputation.png)

Applying `class_weight='balanced'` dropped accuracy to 65.90% while precision only rose to 0.06, confirming the model could not reliably identify the minority class. Given this result, missing values were filled with the mode (`automatic`) instead — a simpler, cheaper, and equally reliable approach given the data's actual distribution:

![Transmission distribution after mode imputation](nb_images/transmission_after_imputation.png)

### Column selection and leakage prevention

The following columns were dropped before modeling:

- `vin` — unique identifier, no predictive value
- `seller` — very high cardinality, low informational value
- `mmr` — a market price estimate for the vehicle itself; using it as a feature would leak information nearly equivalent to the target
- `saledate` — not used as a modeling feature

After imputing `transmission` and removing these columns, remaining residual nulls in other fields (`make`, `model`, `trim`, `body`, `condition`, etc.) were dropped. This reduced the dataset from 558,837 to 546,976 rows — a loss of only **2.12%** — and from 16 to 12 columns.

### Categorical encoding

Categorical variables (`make`, `model`, `trim`, `body`, `transmission`, `state`) were transformed with `LabelEncoder`. Cardinality analysis showed 53 manufacturers, 768 models, 1,494 trims, and 85 body types — high enough that one-hot encoding was impractical, making label encoding a pragmatic (if imperfect) choice for this stage of the project.

## Model Comparison

Three regression models were trained to predict `sellingprice`:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | ~0.40 | ~5,194 | ~7,400 |
| LightGBM | ~0.884 | ~1,900 | ~3,100 |
| XGBoost | ~0.919 | ~1,530 | ~2,500 |

![Model comparison across R², MAE, and RMSE](nb_images/model_comparison.png)

Linear Regression underperformed significantly, largely because label-encoded nominal variables introduce an artificial ordinal relationship that linear models are sensitive to. Both gradient boosting models performed much better, with XGBoost coming out ahead.

## Hyperparameter Tuning and Validation

XGBoost was optimized using `RandomizedSearchCV` (15 iterations, 3-fold cross-validation, minimizing RMSE), producing a final model with:

- **R² ≈ 0.947**
- **MAE ≈ 1,323**
- **RMSE ≈ 2,235**

Model stability was confirmed through repeated 5-fold cross-validation, with R² consistently between 0.943 and 0.947 across runs:

![5-fold cross-validation results](nb_images/cross_validation.png)

A learning curve analysis further supported this: the gap between training and validation R² narrowed from ~0.08 to ~0.02 as the training set size increased, with validation performance still trending upward at the largest sample size tested — suggesting the model is not overfit and could benefit from additional data:

![Learning curve for the optimized XGBoost model](nb_images/learning_curve.png)

## Feature Importance

`year`, `make`, `odometer`, and `body` emerged as the most influential predictors, consistent with domain expectations around vehicle depreciation and market segmentation:

![XGBoost feature importance](nb_images/feature_importance.png)

## Residual Analysis

The residual plot for the optimized model shows no systematic bias (errors centered around zero), but reveals heteroscedasticity — error variance increases at higher predicted prices, with some notable underestimation outliers in the $40,000–$90,000 range:

![Residuals of the optimized XGBoost model](nb_images/residuals_plot.png)

This suggests the model is more reliable for low- and mid-value vehicles than for high-value ones, likely reflecting their lower representation in the training data.

## Key Findings

- The strongest correlations observed were between `odometer` and `year` (newer vehicles have lower mileage), and between `odometer` and `sellingprice` (higher mileage reduces resale value) — both consistent with the classic depreciation-by-usage effect.
- `mmr` was deliberately excluded from modeling to avoid data leakage, since it is itself a market price estimate for the same vehicle.
- Three independent checks (train/test gap, cross-validation, learning curve) confirmed the final model does not suffer from meaningful overfitting.

## Limitations and Future Work

- **Categorical encoding:** Label encoding imposes an artificial order on nominal variables. Target or frequency encoding could improve results, especially for the linear model.
- **Target distribution:** `sellingprice` was not transformed (e.g., log transformation) before modeling, which may partly explain the heteroscedasticity observed in residuals at higher price points.
- **High-value vehicles:** Prediction accuracy degrades for higher-priced vehicles, likely due to their lower representation in the dataset and the influence of factors not captured by the available features (rarity, provenance, condition nuances).

## Tech Stack

- Python, pandas, numpy
- scikit-learn (Logistic Regression, Linear Regression, `LabelEncoder`, `RandomizedSearchCV`, cross-validation)
- LightGBM, XGBoost
- matplotlib, seaborn

## Project Structure

```
.
├── README.md
├── notebook.ipynb
└── nb_images/
    ├── transmission_before_imputation.png
    ├── transmission_after_imputation.png
    ├── model_comparison.png
    ├── cross_validation.png
    ├── learning_curve.png
    ├── feature_importance.png
    └── residuals_plot.png
```
