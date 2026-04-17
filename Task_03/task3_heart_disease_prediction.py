import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_heart_data(input_csv: str = "") -> pd.DataFrame:
    """Load heart disease data from local CSV or fallback public UCI source."""
    if input_csv:
        df = pd.read_csv(input_csv)
        return df

    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    try:
        df = pd.read_csv(uci_url, header=None, names=UCI_COLUMNS)
    except Exception as exc:
        raise ConnectionError(
            "Could not download dataset. Use --input-csv with your Kaggle Heart Disease dataset file."
        ) from exc

    return df


def standardize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize target column name and values to binary classes 0 and 1."""
    data = df.copy()

    if "target" not in data.columns:
        if "num" in data.columns:
            data = data.rename(columns={"num": "target"})
        elif "HeartDisease" in data.columns:
            data = data.rename(columns={"HeartDisease": "target"})

    if "target" not in data.columns:
        raise ValueError("Could not find target column. Expected one of: target, num, HeartDisease")

    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data["target"] = (data["target"] > 0).astype(int)
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: normalize placeholders, booleans, and target column."""
    data = df.replace("?", np.nan).copy()

    # Normalize common boolean text values used in some heart-disease CSVs.
    bool_map = {
        "TRUE": 1,
        "FALSE": 0,
        "true": 1,
        "false": 0,
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
    }
    data = data.replace(bool_map)

    data = standardize_target(data)

    # Drop identifier-like columns that should not be predictive features.
    for col in ["id"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    return data


def run_eda(data: pd.DataFrame, output_dir: Path) -> None:
    """Generate basic EDA plots for class balance and feature relationships."""
    sns.set_theme(style="whitegrid")

    # Class distribution
    plt.figure(figsize=(6, 4))
    class_counts = data["target"].value_counts().sort_index()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="Set2", hue=class_counts.index)
    plt.title("Heart Disease Class Distribution")
    plt.xlabel("Target (0: Low Risk, 1: At Risk)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(11, 8))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_correlation_heatmap.png", dpi=150)
    plt.close()

    # Histograms for numeric features
    numeric_cols = [col for col in data.columns if col != "target"]
    data[numeric_cols].hist(figsize=(12, 10), bins=18, edgecolor="black")
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "eda_feature_histograms.png", dpi=150)
    plt.close()


def build_model(model_name: str, x_train: pd.DataFrame) -> Pipeline:
    """Build a model pipeline with numeric/categorical preprocessing."""
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in x_train.columns if col not in numeric_cols]

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, numeric_cols),
            ("cat", categorical_preprocessor, categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "decision_tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        # Decision Trees do not require feature scaling.
        numeric_preprocessor = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_preprocessor, numeric_cols),
                ("cat", categorical_preprocessor, categorical_cols),
            ],
            remainder="drop",
        )
    else:
        model = LogisticRegression(max_iter=2000, random_state=42)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def evaluate_and_plot(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> tuple[float, float]:
    """Evaluate model and save ROC and confusion matrix plots."""
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # ROC curve
    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_confusion_matrix.png", dpi=150)
    plt.close()

    return acc, roc_auc


def feature_importance(
    pipeline: Pipeline,
    model_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Extract and save feature importance from trained model."""
    transformed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    if model_name == "decision_tree":
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    fi = pd.DataFrame({"feature": transformed_feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi.head(10), x="importance", y="feature", color="#4c72b0")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_top10.png", dpi=150)
    plt.close()

    return fi


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_candidates = [
        base_dir / "heart_disease_uci.csv",
        base_dir.parent / "heart_disease_uci.csv",
    ]

    parser = argparse.ArgumentParser(description="Task 3: Heart Disease Risk Prediction")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Path to Heart Disease dataset CSV. If omitted, uses heart_disease_uci.csv from project root when available.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "decision_tree"],
        help="Classifier to use",
    )
    args = parser.parse_args()

    if not args.input_csv:
        for candidate in default_candidates:
            if candidate.exists():
                args.input_csv = str(candidate)
                break

    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Task 3: Heart Disease Prediction ===")
    print(f"Model selected: {args.model}")
    if args.input_csv:
        print(f"Input dataset: {args.input_csv}")
    else:
        print("Input dataset: UCI fallback download")

    try:
        df_raw = load_heart_data(args.input_csv)
    except Exception as exc:
        print(f"Data loading failed: {exc}")
        print("Tip: Download Heart Disease UCI/Kaggle CSV and run with --input-csv <path>")
        return

    print(f"Loaded rows: {len(df_raw)}")
    print(f"Columns: {list(df_raw.columns)}")

    df = clean_data(df_raw)
    missing_before = int(df.isna().sum().sum())
    print(f"Total missing values before imputation: {missing_before}")

    run_eda(df, output_dir)

    x = df.drop(columns=["target"])
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_model(args.model, x_train)
    pipeline.fit(x_train, y_train)

    acc, roc_auc = evaluate_and_plot(pipeline, x_test, y_test, output_dir)
    fi = feature_importance(pipeline, args.model, output_dir)

    cleaned_path = output_dir / "cleaned_data_preview.csv"
    df.head(50).to_csv(cleaned_path, index=False)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nTop important features:")
    print(fi.head(10).to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {output_dir / 'eda_class_distribution.png'}")
    print(f"- {output_dir / 'eda_correlation_heatmap.png'}")
    print(f"- {output_dir / 'eda_feature_histograms.png'}")
    print(f"- {output_dir / 'evaluation_roc_curve.png'}")
    print(f"- {output_dir / 'evaluation_confusion_matrix.png'}")
    print(f"- {output_dir / 'feature_importance_top10.png'}")
    print(f"- {output_dir / 'feature_importance.csv'}")
    print(f"- {cleaned_path}")


if __name__ == "__main__":
    main()
