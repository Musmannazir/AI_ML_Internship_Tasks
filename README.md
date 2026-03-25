# AI and ML Internship Tasks

This repository contains practical internship tasks focused on data exploration, visualization, and machine learning modeling using Python.

## Project Structure

```text
Internship_Tasks/
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
├── Task_3/
│   ├── task3_heart_disease_prediction.py
│   ├── heart_disease_uci.csv
│   ├── sample_heart_disease.csv
│   └── outputs/
│       ├── cleaned_data_preview.csv
│       └── feature_importance.csv
├── Task_4/
│   └── task4_health_chatbot.py
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer
- pip
- Git (for pushing to GitHub)
- Internet connection for live Yahoo Finance data in Task 2

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Internship_Tasks.git
cd Internship_Tasks
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn yfinance requests
```

### Optional: API Keys for Task 4

For Task 4 (Health Chatbot), you can optionally use your own API keys. Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
HF_API_KEY=your_huggingface_key_here
```

**Important:** Never commit `.env` to GitHub (it's already in `.gitignore`)

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

## Task 3: Heart Disease Prediction

### Objective

Build a binary classification model to predict heart disease risk from patient health attributes.

### What This Task Covers

- Data cleaning for missing placeholders and numeric conversion
- Exploratory Data Analysis (class distribution, heatmap, feature histograms)
- Binary classification using Logistic Regression or Decision Tree
- Model evaluation with Accuracy, ROC-AUC, ROC curve, and confusion matrix
- Feature importance ranking to identify key predictors

### Run With Kaggle or Local CSV

```bash
python Task_3/task3_heart_disease_prediction.py --model logistic --input-csv Task_3/your_heart_dataset.csv
```

Decision Tree option:

```bash
python Task_3/task3_heart_disease_prediction.py --model decision_tree --input-csv Task_3/your_heart_dataset.csv
```

### Run With Fallback Public Source

If no CSV is provided, the script attempts to load UCI Cleveland heart data directly.

```bash
python Task_3/task3_heart_disease_prediction.py --model logistic
```

### Output

Generated files are saved in Task_3/outputs.

## Task 4: General Health Query Chatbot (Prompt Engineering)

### Objective

Build an interactive health chatbot that answers general health questions using an LLM with fallback to local mode. Enforces safety limits on harmful queries.

### What This Task Covers

- Prompt engineering for friendly, clear, non-diagnostic responses
- Multi-backend support (OpenAI, Hugging Face, Local)
- API-based LLM integration with graceful degradation
- Safety filtering for harmful medical misuse queries
- Interactive terminal-based chat interface

### Supported Backends

1. **OpenAI** (requires valid `OPENAI_API_KEY` starting with `sk-`)
2. **Hugging Face** (requires valid `HF_API_KEY`)
3. **Local Mode** (no API required - rule-based responses)

### Setup & Run

#### Windows PowerShell

With OpenAI:
```powershell
$env:OPENAI_API_KEY = "your_openai_key_here"
python Task_4/task4_health_chatbot.py
```

With Hugging Face:
```powershell
$env:HF_API_KEY = "your_huggingface_key_here"
python Task_4/task4_health_chatbot.py
```

Local mode (no API needed):
```powershell
python Task_4/task4_health_chatbot.py
```

#### macOS/Linux

```bash
export OPENAI_API_KEY="your_openai_key_here"
python Task_4/task4_health_chatbot.py
```

### Interactive Chat Example

```
You: Is panadol safe for children?
Bot: Panadol (paracetamol) can be safe for children when used correctly...

You: exit
Bot: Take care. Goodbye!
```

Type `exit`, `quit`, or `bye` to exit.

### Safety Features

- Blocks harmful requests (self-harm, overdose, poison inquiries)
- Blocks emergency queries (chest pain, stroke, severe bleeding)
- Filters responses containing specific drug dosages
- Redirects to healthcare professionals for dangerous situations

## Implementation Notes

- **Task 1:** Uses a non-interactive plotting backend to ensure plots are saved correctly in terminal-only environments.
- **Task 2:** Includes graceful handling for Yahoo Finance connectivity issues and supports a local CSV fallback.
- **Task 3:** Handles missing data (zeros as placeholders) and supports both Logistic Regression and Decision Tree models.
- **Task 4:** Falls back to local rule-based responses if API keys are unavailable or invalid. Always prioritizes safety.

## Pushing to GitHub

### Initial Setup

1. Initialize git (if not already done):
```bash
cd Internship_Tasks
git init
git add .
git commit -m "Initial commit: Internship tasks"
```

2. Create a new empty repository on GitHub (do NOT add README, .gitignore, or license during creation)

3. Add remote and push:
```bash
git remote add origin https://github.com/yourusername/Internship_Tasks.git
git branch -M main
git push -u origin main
```

## Author

Muhammad Usman Nazir
