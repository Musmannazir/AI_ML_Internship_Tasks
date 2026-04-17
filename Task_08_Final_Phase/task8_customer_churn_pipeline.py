"""
Task 8 Final Phase: End-to-End ML Pipeline with scikit-learn Pipeline API

Objective:
Build reusable and production-ready churn prediction pipelines with preprocessing,
hyperparameter tuning, and model export.

Run example:
    python Task_08_Final_Phase/task8_customer_churn_pipeline.py --input-csv Task_08_Final_Phase/Telco-Customer-Churn.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 8 - Customer Churn End-to-End Pipeline")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="Task_08_Final_Phase/Telco-Customer-Churn.csv",
        help="Path to Telco Churn CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Task_08_Final_Phase/outputs",
        help="Directory where metrics and reports will be saved.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="Task_08_Final_Phase/models",
        help="Directory where trained pipeline artifacts are exported.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=2,
        help="Cross-validation folds for GridSearchCV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "f1", "accuracy"],
        help="Primary optimization metric for grid search.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["both", "logistic_regression", "random_forest"],
        help="Which model family to train and tune.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Maximum parallel CPU jobs used by GridSearchCV and RandomForest.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use smaller hyperparameter grids for faster and quieter training.",
    )
    return parser.parse_args()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [c.strip().lower().replace(" ", "_") for c in cleaned.columns]
    return cleaned


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Download the Telco churn dataset and pass its path with --input-csv."
        )

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    rename_map = {
        "customerid": "customer_id",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
        "seniorcitizen": "senior_citizen",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    target_aliases = ["churn", "exited", "is_churn", "left"]
    target_col = next((col for col in target_aliases if col in df.columns), None)
    if target_col is None:
        raise ValueError(f"No churn target found. Expected one of: {target_aliases}")
    if target_col != TARGET_COLUMN:
        df = df.rename(columns={target_col: TARGET_COLUMN})

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Telco datasets often store numeric values as strings with blanks.
    numeric_candidates = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_map = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(target_map)
    )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Target column must contain two classes for churn classification.")

    if len(df) < 50:
        raise ValueError("Dataset is too small after cleaning. Need at least 50 rows.")

    return df


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_features = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numerical_features = [c for c in X.columns if c not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numerical_features, categorical_features


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def run_grid_searches(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    cv_folds: int,
    scoring: str,
    random_state: int,
    model_selection: str,
    max_jobs: int,
    fast_mode: bool,
) -> Dict[str, Dict[str, object]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    logistic_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__class_weight": [None, "balanced"],
    }
    rf_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 8, 16],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 4],
        "classifier__class_weight": [None, "balanced"],
    }

    if fast_mode:
        logistic_grid = {
            "classifier__C": [0.1, 1.0],
            "classifier__solver": ["lbfgs"],
            "classifier__class_weight": [None, "balanced"],
        }
        rf_grid = {
            "classifier__n_estimators": [120, 200],
            "classifier__max_depth": [8, None],
            "classifier__min_samples_split": [2, 10],
            "classifier__class_weight": [None, "balanced"],
        }

    experiments = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=3000, random_state=random_state),
            "param_grid": logistic_grid,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=max_jobs),
            "param_grid": rf_grid,
        },
    }

    selected_models = list(experiments.keys()) if model_selection == "both" else [model_selection]

    results: Dict[str, Dict[str, object]] = {}

    for model_name in selected_models:
        config = experiments[model_name]
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", config["estimator"]),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["param_grid"],
            scoring=scoring,
            cv=cv,
            n_jobs=max_jobs,
            refit=True,
            verbose=1,
        )

        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[model_name] = {
            "search": search,
            "best_pipeline": best_pipeline,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "test_metrics": metrics,
            "classification_report": report,
        }

    return results


def save_artifacts(
    results: Dict[str, Dict[str, object]],
    output_dir: Path,
    model_dir: Path,
    scoring: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    best_model_name = None
    best_model_score = -np.inf

    for model_name, result in results.items():
        metrics = result["test_metrics"]
        summary_rows.append(
            {
                "model": model_name,
                "best_cv_score": result["best_cv_score"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

        joblib.dump(result["best_pipeline"], model_dir / f"{model_name}_pipeline.joblib")

        details = {
            "model": model_name,
            "optimized_for": scoring,
            "best_cv_score": result["best_cv_score"],
            "best_params": result["best_params"],
            "test_metrics": result["test_metrics"],
            "classification_report": result["classification_report"],
        }
        with (output_dir / f"{model_name}_evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(details, handle, indent=2)

        candidate = float(metrics.get("roc_auc", result["best_cv_score"]))
        if candidate > best_model_score:
            best_model_score = candidate
            best_model_name = model_name

    summary_df = pd.DataFrame(summary_rows).sort_values(by="roc_auc", ascending=False)
    summary_path = output_dir / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    if best_model_name is not None:
        best_pipeline = results[best_model_name]["best_pipeline"]
        joblib.dump(best_pipeline, model_dir / "best_churn_pipeline.joblib")

        metadata = {
            "selected_model": best_model_name,
            "selection_metric": "roc_auc",
            "selection_score": best_model_score,
            "optimized_for": scoring,
            "comparison_file": str(summary_path),
        }
        with (model_dir / "best_model_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print(f"Saved comparison table: {summary_path}")
    print(f"Saved model artifacts in: {model_dir}")
    print(f"Saved evaluation reports in: {output_dir}")


def main() -> None:
    args = parse_args()

    df = load_and_prepare_data(Path(args.input_csv))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    preprocessor, num_features, cat_features = build_preprocessor(X)
    print(f"Detected {len(num_features)} numeric and {len(cat_features)} categorical features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    results = run_grid_searches(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        random_state=args.random_state,
        model_selection=args.model,
        max_jobs=args.max_jobs,
        fast_mode=args.fast_mode,
    )

    save_artifacts(
        results=results,
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir),
        scoring=args.scoring,
    )


if __name__ == "__main__":
    main()

