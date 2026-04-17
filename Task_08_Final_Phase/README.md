# Task 8 Final Phase: End-to-End ML Pipeline (Customer Churn)

This task builds a reusable and production-ready churn classification workflow using:

- `Pipeline` and `ColumnTransformer` for preprocessing and modeling
- `GridSearchCV` for hyperparameter tuning
- `joblib` export for deployment/reuse

## Dataset

Use the Telco Churn dataset CSV (for example: `Telco-Customer-Churn.csv`).

Expected target column:

- `Churn` (or aliases such as `Exited`, `is_churn`, `left`)

## Run

From project root:

```bash
python Task_08_Final_Phase/task8_customer_churn_pipeline.py --input-csv Task_08_Final_Phase/Telco-Customer-Churn.csv
```

Optional arguments:

```bash
python Task_08_Final_Phase/task8_customer_churn_pipeline.py \
  --input-csv path/to/Telco-Customer-Churn.csv \
  --output-dir Task_08_Final_Phase/outputs \
  --model-dir Task_08_Final_Phase/models \
  --cv-folds 5 \
  --scoring roc_auc
```

## What It Does

1. Cleans and standardizes column names
2. Handles mixed numeric/categorical preprocessing
3. Trains and tunes:
   - Logistic Regression
   - Random Forest
4. Evaluates on a held-out test split
5. Exports model artifacts and reports

## Outputs

Saved in `Task_08_Final_Phase/outputs`:

- `model_comparison.csv`
- `logistic_regression_evaluation.json`
- `random_forest_evaluation.json`

Saved in `Task_08_Final_Phase/models`:

- `logistic_regression_pipeline.joblib`
- `random_forest_pipeline.joblib`
- `best_churn_pipeline.joblib`
- `best_model_metadata.json`

