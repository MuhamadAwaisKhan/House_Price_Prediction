# House Price Prediction (Regression)

Predicting median house value from housing features using the California Housing dataset.

## Problem

Given features about a housing block — median income, average rooms, population,
location-related stats — predict the median house value. This is a supervised
regression problem, useful for understanding how well simple vs. ensemble models
capture non-linear relationships in real-world tabular data.

## Dataset

California Housing dataset (20,433 records after cleaning), sourced via scikit-learn
with an automatic fallback to a GitHub-hosted mirror if the primary source is
unavailable. Features include median income, house age, average rooms/bedrooms,
population, and average occupancy per household.

## Approach

1. **EDA** — correlation heatmap and target distribution to understand feature
   relationships and skew.
2. **Feature engineering** — derived `RoomsPerHousehold`, `BedroomsPerRoom`, and
   `PopulationPerHousehold` to capture ratios that raw counts miss.
3. **Modeling** — trained and compared three models:
   - Linear Regression (baseline)
   - Random Forest
   - XGBoost
4. **Hyperparameter tuning** — GridSearchCV over Random Forest's `n_estimators`
   and `max_depth`.
5. **Evaluation** — MAE, RMSE, and R² on a held-out 20% test set.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.5505 | 0.7528 | 0.5856 |
| Random Forest | 0.4609 | 0.6492 | 0.6918 |
| XGBoost | 0.4559 | 0.6400 | **0.7005** |
| Tuned Random Forest (GridSearchCV) | 0.4597 | 0.6480 | 0.6930 |

Best Random Forest hyperparameters found: `max_depth=10`, `n_estimators=200`.

## Key Insight

XGBoost outperformed both plain and tuned Random Forest, suggesting gradient
boosting captured the non-linear relationships between income, location, and
price slightly better than bagging-based Random Forest on this dataset.
Interestingly, hyperparameter tuning gave only a marginal improvement over the
untuned Random Forest — a reminder that model choice mattered more than tuning
in this case.

## What I'd Improve With More Time

- Add geospatial features (distance to coast/city center) using the latitude/
  longitude columns, which were dropped in this version.
- Try a wider hyperparameter search space or Bayesian optimization instead of
  grid search.
- Add SHAP values for more interpretable feature importance.

## How to Run

```bash
pip install -r requirements.txt
python house_price_prediction.py
```

Outputs: `correlation_heatmap.png`, `target_distribution.png`,
`feature_importance.png`, `model_comparison.csv`, `house_price_model.pkl`.
