import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
DEFAULT_KAGGLE_DATASET = "ted8080/house-prices-and-images-socal"


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_tab_train: np.ndarray
    X_tab_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class HousingMultimodalDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        tabular_features: np.ndarray,
        targets: Optional[np.ndarray],
        image_transform: transforms.Compose,
    ) -> None:
        self.image_paths = list(image_paths)
        self.tabular_features = tabular_features.astype(np.float32)
        self.targets = None if targets is None else targets.astype(np.float32)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)
        tab = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        if self.targets is None:
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, tab, target


class MultimodalRegressor(nn.Module):
    def __init__(self, tabular_dim: int, pretrained_cnn: bool = False) -> None:
        super().__init__()

        if pretrained_cnn:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = models.resnet18(weights=weights)
        image_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        tabular_features = self.tabular_encoder(tabular)
        fused_features = torch.cat([image_features, tabular_features], dim=1)
        output = self.regressor(fused_features)
        return output.squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 9 - Multimodal Housing Price Prediction")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Path to housing CSV data (optional when using KaggleHub).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Directory containing house images (optional when using KaggleHub).",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["kagglehub", "local"],
        default="kagglehub",
        help="Data source to use. 'kagglehub' downloads dataset automatically.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help="Kaggle dataset identifier used when --dataset-source kagglehub.",
    )
    parser.add_argument("--target-column", type=str, default="price", help="Target column name.")
    parser.add_argument(
        "--image-column",
        type=str,
        default="",
        help="Column containing image file names or relative paths.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="",
        help="Column used to map IDs to image files when image-column is absent.",
    )
    parser.add_argument(
        "--drop-id-features",
        action="store_true",
        help="Drop ID-like fields from tabular features.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--image-size", type=int, default=224, help="Square size for image resizing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--pretrained-cnn",
        action="store_true",
        help="Use pretrained ResNet18 weights (requires one-time model download).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Task_9_Multimodal/outputs",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit dataset to first N rows (0=all, useful for testing).",
    )
    args, unknown = parser.parse_known_args()
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def infer_image_column(df: pd.DataFrame, user_col: str) -> Optional[str]:
    if user_col:
        col = user_col.strip().lower()
        return col if col in df.columns else None

    candidates = ["image", "image_path", "image_file", "image_name", "filename", "img"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_id_column(df: pd.DataFrame, user_col: str) -> Optional[str]:
    if user_col:
        col = user_col.strip().lower()
        return col if col in df.columns else None

    candidates = ["image_id", "id", "house_id", "property_id"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_image_path(
    raw_value: str,
    image_dir: Path,
    exts: Sequence[str],
    image_index_by_name: Dict[str, Path],
    image_index_by_stem: Dict[str, Path],
) -> Optional[Path]:
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return None

    candidate = Path(value)
    if candidate.is_file():
        return candidate

    joined = image_dir / candidate
    if joined.is_file():
        return joined

    by_name = image_index_by_name.get(candidate.name.lower())
    if by_name is not None:
        return by_name

    if candidate.suffix:
        return None

    for ext in exts:
        by_stem = image_index_by_stem.get(f"{value}{ext}".lower())
        if by_stem is not None:
            return by_stem

        p = image_dir / f"{value}{ext}"
        if p.is_file():
            return p

    by_stem = image_index_by_stem.get(value.lower())
    if by_stem is not None:
        return by_stem

    return None


def build_image_index(image_dir: Path, exts: Sequence[str]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    print(f"[build_image_index] Starting indexing for {image_dir}", flush=True)
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}

    import time
    t0 = time.time()
    for path in sorted(image_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        name_key = path.name.lower()
        stem_key = path.stem.lower()

        if name_key not in by_name:
            by_name[name_key] = path
        if stem_key not in by_stem:
            by_stem[stem_key] = path

    elapsed = time.time() - t0
    print(f"[build_image_index] Built index with {len(by_name)} files in {elapsed:.2f}s", flush=True)
    return by_name, by_stem


def build_image_mapping(
    df: pd.DataFrame,
    image_dir: Path,
    image_col: Optional[str],
    id_col: Optional[str],
) -> pd.DataFrame:
    exts = DEFAULT_IMAGE_EXTS
    mapped = df.copy()
    mapped["__image_path"] = None
    image_index_by_name, image_index_by_stem = build_image_index(image_dir, exts)

    if image_col and image_col in mapped.columns:
        mapped["__image_path"] = mapped[image_col].apply(
            lambda x: resolve_image_path(
                str(x),
                image_dir,
                exts,
                image_index_by_name=image_index_by_name,
                image_index_by_stem=image_index_by_stem,
            )
        )
    elif id_col and id_col in mapped.columns:
        mapped["__image_path"] = mapped[id_col].apply(
            lambda x: resolve_image_path(
                str(x),
                image_dir,
                exts,
                image_index_by_name=image_index_by_name,
                image_index_by_stem=image_index_by_stem,
            )
        )

    valid = mapped[mapped["__image_path"].notna()].copy()
    valid["__image_path"] = valid["__image_path"].astype(str)
    return valid


def detect_csv_from_dataset(dataset_root: Path, preferred_name: str = "") -> Path:
    csv_paths = sorted(dataset_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under dataset path: {dataset_root}")

    if preferred_name:
        preferred_lower = preferred_name.lower()
        for p in csv_paths:
            if p.name.lower() == preferred_lower:
                return p

    # Favor likely housing table names when present.
    keywords = ("house", "housing", "price", "sale")
    for p in csv_paths:
        if any(k in p.name.lower() for k in keywords):
            return p

    return csv_paths[0]


def detect_image_root_from_dataset(dataset_root: Path) -> Path:
    best_dir = None
    best_count = -1

    for d in [dataset_root] + [p for p in dataset_root.rglob("*") if p.is_dir()]:
        count = 0
        for ext in DEFAULT_IMAGE_EXTS:
            count += len(list(d.glob(f"*{ext}")))
        if count > best_count:
            best_count = count
            best_dir = d

    if best_dir is None or best_count <= 0:
        raise FileNotFoundError(f"No image files found under dataset path: {dataset_root}")

    return best_dir


def resolve_data_sources(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path]]:
    if args.dataset_source == "local":
        if not args.input_csv:
            raise ValueError("--input-csv is required when --dataset-source local")
        if not args.image_dir:
            raise ValueError("--image-dir is required when --dataset-source local")
        csv_path = Path(args.input_csv)
        image_dir = Path(args.image_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        return csv_path, image_dir, None

    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for --dataset-source kagglehub. Install with: pip install kagglehub"
        ) from exc

    dataset_path = Path(kagglehub.dataset_download(args.kaggle_dataset))
    csv_path = detect_csv_from_dataset(dataset_path, Path(args.input_csv).name if args.input_csv else "")
    image_dir = Path(args.image_dir) if args.image_dir else detect_image_root_from_dataset(dataset_path)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return csv_path, image_dir, dataset_path


def prepare_tabular_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    image_column: Optional[str],
    id_column: Optional[str],
    drop_id_features: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], ColumnTransformer]:
    excluded_cols = {target_column, "__image_path"}
    if image_column:
        excluded_cols.add(image_column)

    if drop_id_features and id_column:
        excluded_cols.add(id_column)

    feature_cols = [c for c in train_df.columns if c not in excluded_cols]
    if not feature_cols:
        raise ValueError("No tabular feature columns remain after exclusions.")

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return X_train.astype(np.float32), X_test.astype(np.float32), feature_cols, preprocessor


def create_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    seed: int,
    image_column: Optional[str],
    id_column: Optional[str],
    drop_id_features: bool,
) -> SplitData:
    y = pd.to_numeric(df[target_column], errors="coerce")
    valid_idx = y.notna()
    df = df.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].values.astype(np.float32)

    if len(df) < 40:
        raise ValueError("Need at least 40 valid rows with images for a stable multimodal split.")

    train_df, test_df, y_train, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=seed,
    )

    X_tab_train, X_tab_test, _, _ = prepare_tabular_features(
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
        image_column=image_column,
        id_column=id_column,
        drop_id_features=drop_id_features,
    )

    return SplitData(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        X_tab_train=X_tab_train,
        X_tab_test=X_tab_test,
        y_train=y_train,
        y_test=y_test,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, tabular, targets in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images, tabular)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for images, tabular, targets in loader:
        images = images.to(device)
        tabular = tabular.to(device)

        preds = model(images, tabular).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    targets_np = np.concatenate(all_targets) if all_targets else np.array([])
    return preds_np, targets_np


def save_outputs(
    output_dir: Path,
    train_losses: List[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"MAE": mae, "RMSE": rmse}

    metrics_path = output_dir / "multimodal_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"actual_price": y_true, "predicted_price": y_pred})
    pred_df.to_csv(output_dir / "actual_vs_predicted_multimodal.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.title("Training Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curve.png")
    plt.close()

    sorted_df = pred_df.sort_values("actual_price").reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_df.index, sorted_df["actual_price"], label="Actual", linewidth=2)
    plt.plot(sorted_df.index, sorted_df["predicted_price"], label="Predicted", linewidth=2)
    plt.title("Multimodal Housing Price Prediction (Test Set)")
    plt.xlabel("Test Sample (sorted by actual)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_plot.png")
    plt.close()

    return metrics


def main() -> None:
    print("[Main] Starting...", flush=True)
    args = parse_args()
    print("[Main] Arguments parsed", flush=True)
    set_seed(args.seed)

    print("[Main] Resolving data sources...", flush=True)
    csv_path, image_dir, dataset_path = resolve_data_sources(args)
    output_dir = Path(args.output_dir)

    print(f"[Main] Loading CSV from {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    if args.max_rows > 0:
        df = df.head(args.max_rows)

    target_col = args.target_column.strip().lower()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing in CSV columns: {list(df.columns)}")

    image_col = infer_image_column(df, args.image_column)
    id_col = infer_id_column(df, args.id_column)

    if image_col is None and id_col is None:
        raise ValueError(
            "Could not map rows to images. Provide --image-column with file names/paths or --id-column with IDs."
        )

    print(f"[Main] Building image mapping with id_col={id_col}...", flush=True)
    mapped = build_image_mapping(df, image_dir=image_dir, image_col=image_col, id_col=id_col)
    print(f"[Main] Mapped {len(mapped)} rows to images", flush=True)
    if len(mapped) < 40:
        raise ValueError(
            "Fewer than 40 rows matched with image files. Ensure image naming matches --image-column or --id-column."
        )

    split = create_split(
        df=mapped,
        target_column=target_col,
        test_size=args.test_size,
        seed=args.seed,
        image_column=image_col,
        id_column=id_col,
        drop_id_features=args.drop_id_features,
    )

    print(f"[Main] Created train/test split: {len(split.train_df)} train, {len(split.test_df)} test", flush=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("[Main] Creating datasets...", flush=True)
    train_ds = HousingMultimodalDataset(
        image_paths=[Path(p) for p in split.train_df["__image_path"].tolist()],
        tabular_features=split.X_tab_train,
        targets=split.y_train,
        image_transform=train_transform,
    )
    test_ds = HousingMultimodalDataset(
        image_paths=[Path(p) for p in split.test_df["__image_path"].tolist()],
        tabular_features=split.X_tab_test,
        targets=split.y_test,
        image_transform=eval_transform,
    )
    print("[Main] Datasets created, creating dataloaders...", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("[Main] Dataloaders created", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}", flush=True)

    print("[Main] Creating model...", flush=True)
    model = MultimodalRegressor(tabular_dim=split.X_tab_train.shape[1], pretrained_cnn=args.pretrained_cnn)
    model = model.to(device)
    print("[Main] Model created and moved to device", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("[Main] Starting training...", flush=True)
    train_losses: List[float] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(loss)
        print(f"Epoch {epoch:02d}/{args.epochs} - Train MSE: {loss:.4f}", flush=True)

    print("[Main] Training complete, evaluating...", flush=True)
    y_pred, y_true = evaluate_model(model, test_loader, device)
    metrics = save_outputs(output_dir=output_dir, train_losses=train_losses, y_true=y_true, y_pred=y_pred)
    print("[Main] Outputs saved", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "multimodal_regressor.pt"
    torch.save(model.state_dict(), model_path)

    print("\nTask 9 - Multimodal Housing Price Prediction Results")
    if dataset_path is not None:
        print(f"Dataset source: KaggleHub ({args.kaggle_dataset})")
        print(f"Downloaded dataset path: {dataset_path}")
    print(f"Rows used (with images): {len(mapped)}")
    print(f"Train rows: {len(split.train_df)}, Test rows: {len(split.test_df)}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"Saved model weights: {model_path}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()