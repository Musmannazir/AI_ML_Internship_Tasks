import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    except Exception as exc:
        raise ConnectionError(
            "Yahoo Finance download failed. Check internet/DNS or provide --input-csv."
        ) from exc
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check connectivity or provide --input-csv."
        )
    return df


def load_stock_data_from_csv(csv_path: str) -> pd.DataFrame:
    """Load historical stock data from a local CSV file."""
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix and target for next-day close prediction."""
    data = df[["Open", "High", "Low", "Volume", "Close"]].copy()
    data["Target_Next_Close"] = data["Close"].shift(-1)
    data = data.dropna()
    return data


def train_and_predict(
    data: pd.DataFrame, model_name: str, test_size: float = 0.2
) -> tuple[pd.DataFrame, float, float, float]:
    """Train selected regression model and return predictions on the time-ordered test split."""
    features = data[["Open", "High", "Low", "Volume"]]
    target = data["Target_Next_Close"]

    split_idx = int(len(data) * (1 - test_size))
    x_train, x_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

    if model_name == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    results = pd.DataFrame(
        {
            "Date": y_test.index,
            "Actual_Next_Close": y_test.values,
            "Predicted_Next_Close": y_pred,
        }
    ).set_index("Date")

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    return results, mae, rmse, r2


def plot_predictions(results: pd.DataFrame, ticker: str, output_dir: Path) -> str:
    """Plot actual vs predicted next-day close prices and save the figure."""
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["Actual_Next_Close"], label="Actual Next Close", linewidth=2)
    plt.plot(results.index, results["Predicted_Next_Close"], label="Predicted Next Close", linestyle="--", linewidth=2)
    plt.title(f"{ticker} - Actual vs Predicted Next-Day Close")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{ticker}_actual_vs_predicted.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day stock close price using regression models.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (example: AAPL, TSLA)")
    parser.add_argument("--period", type=str, default="5y", help="Historical period to download (example: 1y, 2y, 5y)")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "linear"],
        help="Model to train: random_forest or linear",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Optional local CSV path with Date, Open, High, Low, Close, Volume columns.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Task 2: Predict Future Stock Prices (Short-Term) ===")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ticker selected: {args.ticker}")
    print(f"Download period: {args.period}")
    print(f"Model selected: {args.model}")

    try:
        if args.input_csv:
            print(f"Data source: Local CSV ({args.input_csv})")
            df = load_stock_data_from_csv(args.input_csv)
        else:
            print("Data source: Yahoo Finance (yfinance)")
            df = load_stock_data(args.ticker, args.period)
    except Exception as exc:
        print(f"\nData loading failed: {exc}")
        print("Tip: If Yahoo Finance is blocked on your network, run with --input-csv <path_to_file>.")
        return

    print(f"Rows downloaded: {len(df)}")
    print("\nData sample:")
    print(df.head())

    data = build_features(df)
    print("\nFeatures used: Open, High, Low, Volume")
    print("Target: Next day's Close")
    print(f"Modeling rows after shift/dropna: {len(data)}")

    results, mae, rmse, r2 = train_and_predict(data, model_name=args.model, test_size=0.2)

    print("\nModel performance on test set:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    output_plot = plot_predictions(results, args.ticker, output_dir)
    output_csv = output_dir / f"{args.ticker}_actual_vs_predicted.csv"
    results.to_csv(output_csv)

    print("\nOutputs generated:")
    print(f"- Plot: {output_plot}")
    print(f"- Data: {output_csv}")


if __name__ == "__main__":
    main()
