"""
Task 6: House Price Prediction

Objective:
Predict house prices using property features such as size, bedrooms, and location.

Run examples:
    python Task_6/task6_house_price_prediction.py --input-csv Task_6/Housing.csv --model linear
    python Task_6/task6_house_price_prediction.py --input-csv Task_6/Housing.csv --model gradient_boosting
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor


REQUIRED_COLUMNS = ["square_feet", "bedrooms", "price"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6 - House Price Prediction")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="Task_6/Housing.csv",
        help="Path to Kaggle or local house-price CSV file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "gradient_boosting"],
        default="gradient_boosting",
        help="Regression model type.",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Use log1p(price) during training and transform back for predictions.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Provide a valid Kaggle dataset path via --input-csv."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common alternate names to expected feature names.
    rename_map = {
        "sqft": "square_feet",
        "sqft_living": "square_feet",
        "area": "square_feet",
        "bhk": "bedrooms",
        "bedroom": "bedrooms",
        "city": "location",
        "neighborhood": "location",
        "locality": "location",
        "house_price": "price",
        "saleprice": "price",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing}. Required: {REQUIRED_COLUMNS}."
        )

    # Keep all columns except unsupported identifiers if present.
    dropped_id_like = [c for c in ["id", "unnamed: 0"] if c in df.columns]
    cleaned = df.drop(columns=dropped_id_like).copy()

    # Normalize categorical string values.
    object_cols = cleaned.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip().replace({"": np.nan})

    # Attempt numeric conversion for non-categorical columns commonly numeric.
    for col in cleaned.columns:
        if col == "price":
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        elif col not in object_cols:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["price"])

    # Feature engineering for better price signal.
    if "square_feet" in cleaned.columns and "bedrooms" in cleaned.columns:
        denom = cleaned["bedrooms"].replace(0, np.nan)
        cleaned["area_per_bedroom"] = cleaned["square_feet"] / denom

    if "bathrooms" in cleaned.columns and "bedrooms" in cleaned.columns:
        denom = cleaned["bedrooms"].replace(0, np.nan)
        cleaned["bathrooms_per_bedroom"] = cleaned["bathrooms"] / denom

    if "stories" in cleaned.columns and "square_feet" in cleaned.columns:
        cleaned["area_x_stories"] = cleaned["square_feet"] * cleaned["stories"]

    if "parking" in cleaned.columns:
        cleaned["has_parking"] = (pd.to_numeric(cleaned["parking"], errors="coerce") > 0).astype(float)

    if len(cleaned) < 20:
        raise ValueError("Need at least 20 valid rows for a meaningful train/test split.")

    return cleaned


def build_pipeline(model_type: str, num_features: List[str], cat_features: List[str]) -> Pipeline:
    if model_type == "linear":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        # Tree-based models do not require scaling.
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    if model_type == "linear":
        model = LinearRegression()
    else:
        model = GradientBoostingRegressor(
            random_state=42,
            n_estimators=800,
            learning_rate=0.01,
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=10,
            subsample=0.6,
            max_features="log2",
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, r2, mape


def save_outputs(
    output_dir: Path,
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    mae: float,
    rmse: float,
    r2: float,
    mape: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({"actual_price": y_test.values, "predicted_price": y_pred})
    pred_csv = output_dir / f"house_actual_vs_predicted_{model_name}.csv"
    pred_df.to_csv(pred_csv, index=False)

    # Sort for an easier visual comparison.
    sorted_df = pred_df.sort_values("actual_price").reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_df.index, sorted_df["actual_price"], label="Actual", linewidth=2)
    plt.plot(sorted_df.index, sorted_df["predicted_price"], label="Predicted", linewidth=2)
    plt.title(f"House Price Prediction ({model_name})")
    plt.xlabel("Test Samples (sorted by actual price)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"house_actual_vs_predicted_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()

    metrics_path = output_dir / f"metrics_{model_name}.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"Model: {model_name}",
                f"MAE: {mae:.2f}",
                f"RMSE: {rmse:.2f}",
                f"R2: {r2:.4f}",
                f"MAPE (%): {mape:.2f}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved predictions: {pred_csv}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    df = load_data(csv_path)

    target = "price"
    features = [c for c in df.columns if c != target]

    # Split by data type so this works with Housing.csv and other Kaggle variants.
    cat_features = [c for c in features if str(df[c].dtype) == "object"]
    num_features = [c for c in features if c not in cat_features]

    X = df[features]
    y = df[target]

    # Stratify by price quantiles so train/test both represent low/mid/high price ranges.
    stratify_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify_bins
    )

    pipeline = build_pipeline(args.model, num_features, cat_features)

    if args.log_target:
        estimator = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        estimator = pipeline

    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    mae, rmse, r2, mape = evaluate(y_test.values, y_pred)

    print("House Price Prediction Results")
    print(f"Model: {args.model}")
    print(f"Rows used: {len(df)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    save_outputs(
        output_dir=Path("Task_6/outputs"),
        model_name=args.model,
        y_test=y_test,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
    )


if __name__ == "__main__":
    main()
