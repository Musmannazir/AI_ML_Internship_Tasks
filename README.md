# AI and ML Internship Tasks

This repository contains practical internship tasks focused on data exploration, predictive modeling, chatbot development, and an advanced NLP project using Python.

## Project Structure

```text
Internship_Tasks/
|-- Task_1/
|   |-- task1_iris_exploration.py
|   `-- plots/
|-- Task_2/
|   |-- task2_stock_prediction.py
|   |-- sample_stock_data.csv
|   `-- outputs/
|-- Task_3/
|   |-- task3_heart_disease_prediction.py
|   |-- heart_disease_uci.csv
|   |-- sample_heart_disease.csv
|   `-- outputs/
|-- Task_4/
|   `-- task4_health_chatbot.py
|-- Task_5/
|   |-- task5_mental_health_chatbot.py
|   |-- config.ini
|   |-- examples.py
|   |-- README.md
|   |-- Training.png
|   `-- UI_Screen.png
|-- Task_6/
|   |-- task6_house_price_prediction.py
|   |-- Housing.csv
|   `-- outputs/
|-- Task_7/  (advanced NLP project implementation)
|   |-- app.py
|   |-- train.py
|   |-- utils.py
|   `-- requirements.txt
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.10+
- pip
- Git



## Task 1: Iris Dataset Exploration and Visualization

- Dataset inspection, summary statistics, and visual EDA
- Outputs saved to Task_1/plots

Run:

```bash
python Task_1/task1_iris_exploration.py
```

## Task 2: Stock Price Prediction

- Predict next-day close price using Linear Regression or Random Forest
- Uses Yahoo Finance or local CSV fallback
- Outputs saved to Task_2/outputs

Run (live data):

```bash
python Task_2/task2_stock_prediction.py --ticker AAPL --period 2y --model random_forest
```

Run (CSV fallback):

```bash
python Task_2/task2_stock_prediction.py --ticker AAPL --model linear --input-csv Task_2/sample_stock_data.csv
```

## Task 3: Heart Disease Prediction

- Binary classification with Logistic Regression or Decision Tree
- Includes preprocessing, metrics, and feature importance
- Outputs saved to Task_3/outputs

Run:

```bash
python Task_3/task3_heart_disease_prediction.py --model logistic --input-csv Task_3/sample_heart_disease.csv
```

## Task 4: General Health Query Chatbot

- Prompt-engineered chatbot with safety filtering
- Supports OpenAI, Hugging Face, and local fallback mode

Run:

```bash
python Task_4/task4_health_chatbot.py
```

## Task 5: Mental Health Support Chatbot (Fine-Tuned)

- Fine-tuning and inference workflow for empathetic mental health responses
- Includes CLI mode and Streamlit web UI
- Crisis-keyword safety handling included

Run training:

```bash
python Task_5/task5_mental_health_chatbot.py --train
```

Run chat (CLI):

```bash
python Task_5/task5_mental_health_chatbot.py --chat
```

Run web app:

```bash
python -m streamlit run Task_5/task5_mental_health_chatbot.py
```

## Task 6: House Price Prediction

- Uses Kaggle-style housing data (Task_6/Housing.csv)
- Preprocessing + feature engineering + regression pipelines
- Supports Linear Regression and Gradient Boosting
- Outputs saved to Task_6/outputs

Run:

```bash
python Task_6/task6_house_price_prediction.py --input-csv Task_6/Housing.csv --model gradient_boosting
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Task 7: Advanced News Topic Classifier Using BERT

- Fine-tunes `bert-base-uncased` on the AG News dataset from Hugging Face
- Tokenizes and preprocesses news text
- Evaluates with accuracy and macro F1-score
- Provides a Streamlit app for live topic prediction

Dataset:

```python
from datasets import load_dataset

ds = load_dataset("sh0416/ag_news")
```

Run training:

```bash
python Task_7/train.py --do_train --do_eval --max_train_samples 2000 --max_eval_samples 500 --num_train_epochs 1
```

Run the app:

```bash
streamlit run Task_7/app.py
```

The trained model is saved to `models/news-topic-bert` by default.

## Security Notes

- Do not commit .env or API keys
- .gitignore excludes environment, cache, and local artifact files

## Author

Muhammad Usman Nazir
