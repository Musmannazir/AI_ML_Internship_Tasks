# AI and ML Internship Tasks

This repository contains practical internship tasks focused on data exploration, visualization, and machine learning modeling using Python.

## Project Structure

```text
AI_ML_Internship_Tasks/
├── Task_1/
│   ├── task1_iris_exploration.py
│   └── plots/
│       ├── scatter_sepal_length_vs_width.png
│       ├── histograms_numeric_features.png
│       └── boxplots_by_species.png
├── Task_2/
│   ├── task2_stock_prediction.py
│   ├── sample_stock_data.csv
│   └── outputs/
│       ├── AAPL_actual_vs_predicted.csv
│       └── AAPL_actual_vs_predicted.png
└── README.md
```

## Requirements

- Python 3.10 or newer
- pip
- Internet connection for live Yahoo Finance data in Task 2

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn yfinance
```

## Task 1: Iris Dataset Exploration and Visualization

### Objective

Load and inspect the Iris dataset, summarize statistics, and visualize feature relationships and distributions.

### What This Task Covers

- Data loading with pandas
- Basic dataset inspection with shape, columns, head, info, and describe
- Scatter plot for feature relationships
- Histograms for value distributions
- Box plots for outlier analysis

### Run

From repository root:

```bash
python Task_1/task1_iris_exploration.py
```

### Output

Generated images are saved in Task_1/plots.

## Task 2: Short-Term Stock Price Prediction

### Objective

Use historical stock data to predict the next day closing price using regression models.

### What This Task Covers

- Data fetching from Yahoo Finance using yfinance
- Time series style feature engineering for next-day prediction
- Regression modeling with Random Forest or Linear Regression
- Evaluation using MAE, RMSE, and R2
- Actual vs predicted closing price visualization

### Features and Target

- Features: Open, High, Low, Volume
- Target: Next day Close

### Run With Live Yahoo Finance Data

```bash
python Task_2/task2_stock_prediction.py --ticker AAPL --period 2y --model random_forest
```

Example with another stock:

```bash
python Task_2/task2_stock_prediction.py --ticker TSLA --period 1y --model linear
```

### Run With Local CSV Fallback

Use this if your network blocks Yahoo Finance.

```bash
python Task_2/task2_stock_prediction.py --ticker AAPL --model linear --input-csv Task_2/sample_stock_data.csv
```

### Output

Generated files are saved in Task_2/outputs.

## Notes

- Task 1 uses a non-interactive plotting backend to ensure plots are saved correctly in terminal-only environments.
- Task 2 includes graceful handling for Yahoo Finance connectivity issues and supports a local CSV fallback.

## Author

Muhammad Usman Nazir