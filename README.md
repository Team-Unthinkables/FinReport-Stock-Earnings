# FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model

## Overview

FinReport is a research-oriented system designed to forecast stock earnings by analyzing both technical indicators and financial news. It leverages a multi-factor model implemented in PyTorch that integrates three key modules:

- **News Factorization:**  
  Extracts and processes features from financial news—including sentiment analysis using FinBERT and event extraction using AllenNLP (augmented with domain-specific keyword rules). It also supports aggregation of multiple news items (with a basic temporal analysis) to compute an overall news factor.

- **Return Forecasting:**  
  Uses historical stock data and technical indicators (both standard and domain-specific) to predict returns via an LSTM model. Internal proxies (e.g., market value for Size Factor) are used as substitutes for external classical factors like Fama–French.

- **Risk Assessment:**  
  Evaluates risk using advanced metrics including EGARCH-based volatility forecasting, maximum drawdown, and Conditional Value at Risk (CVaR). A simple integrated risk-adjusted ratio (expected return divided by volatility) is also computed.

The final output is a polished HTML report that presents predictions, risk assessments, detailed technical factors (with conditional styling), and source attributions. In addition, the system prints comprehensive performance metrics (both regression and binary classification metrics) in a tabular format on the terminal for further analysis.

## Features

- **Data Integration:**  
  - Combines historical stock data (including numerous technical indicators) with processed news text.  
  - Automatically renames technical indicator columns (e.g., converting tuple-like strings to underscore-separated names) for consistency.

- **Model Training & Evaluation:**  
  - Trains an LSTM-based model in PyTorch using time-series data with a fixed sequence length.  
  - Hyperparameters (such as batch size, sequence length, learning rate, and model architecture) are loaded from a YAML configuration file to ensure consistency between training and evaluation.

- **Enhanced Feature Engineering:**  
  - **Sentiment Analysis:** Uses FinBERT (via Hugging Face) to compute a precise sentiment score from news text.  
  - **Event Extraction:** Employs AllenNLP’s SRL model with domain-specific keyword rules to extract events and compute an event factor.  
  - **Multi-Article Aggregation:** (Optional) Aggregates multiple news items per day using a dedicated module to compute an overall news effect factor.  
  - **Internal Proxies for Classical Factors:** Derives factors (e.g., Size Factor) directly from CSV data such as market value, avoiding external data fetches.

- **Dynamic Report Generation:**  
  - Generates individual HTML reports for each stock and combines them into a single multi-stock HTML page.  
  - Reports are styled with blue bullet labels; key numeric values (e.g., percentage changes) are highlighted in green (if positive) or red (if negative) while the remainder of the text remains black.  
  - Detailed factors (Market, Size, Valuation, Profitability, Investment, News Effect, Event, and technical indicators like RSI/MFI/BIAS) are included.

- **Advanced Risk Assessment:**  
  - **EGARCH Volatility Forecasting:** Computes conditional volatility and an approximate 95% Value at Risk (VaR).  
  - **Additional Risk Metrics:** Calculates maximum drawdown and CVaR (Expected Shortfall) using historical simulation.  
  - **Risk-Adjusted Ratio:** Computes a simple ratio (predicted return divided by volatility) to assess risk-adjusted performance.

- **Performance Metrics Reporting:**  
  - Prints regression metrics (MSE, RMSE, MAE) and binary classification metrics (Accuracy, Precision, Recall, F1-score, AUC, AUPR, Error Rate, and Confusion Matrix components) in a tabular format on the terminal for deeper performance analysis.  
  - Uses a hybrid approach by dynamically binarizing continuous return predictions (using a stock-specific threshold) to compute classification metrics alongside regression metrics.

- **Modular Design:**  
  The code is organized into separate modules for data loading, preprocessing, model definition, training, evaluation, sentiment analysis, extra factor computation, risk modeling, and report generation.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Kanishk1420/FinReport-Explainable-Stock-Earnings-Forecasting-via-News-Factor.git
   cd FinReport-Explainable-Stock-Earnings-Forecasting-via-News-Factor
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7+ installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include: `pandas`, `numpy`, `torch`, `jinja2`, `pyyaml`, `transformers`, `arch`, `allennlp`, `allennlp-models`, and others as listed in `requirements.txt`.

3. **Project Structure:**

   ```
   FinReport/
   ├── README.md                   # Project documentation
   ├── requirements.txt            # List of dependencies
   ├── config.yaml                 # Configuration for hyperparameters and file paths
   ├── data/
   │   └── stock_data.csv          # Historical stock data and news factors
   ├── templates/
   │   ├── report_template.html    # HTML template for individual reports (with conditional styling)
   │   ├── multi_report_template.html  # HTML template for combined reports
   │   └── report_style.css        # (Optional) External CSS for styling
   ├── src/
   │   ├── __init__.py
   │   ├── data_loader.py          # Loads and parses CSV data
   │   ├── preprocessing.py        # Feature extraction, technical column renaming, and normalization
   │   ├── model.py                # PyTorch LSTM model definition
   │   ├── train.py                # Script to train the model
   │   ├── evalute.py              # Script to evaluate the model, compute performance metrics, and generate HTML reports (with risk metrics)
   │   ├── multi_eval.py           # Script to generate a combined HTML report for multiple stocks
   │   ├── report_generator.py     # Generates HTML reports using Jinja2 (handles extra factors and risk metrics)
   │   ├── sentiment.py            # FinBERT-based sentiment analysis module
   │   ├── extra_factors.py        # Computes additional domain-specific technical indicators with split descriptions for styling
   │   ├── advanced_news.py        # SRL-based event extraction and event factor computation using AllenNLP
   │   ├── news_aggregator.py      # (Optional) Aggregates multiple news items for enhanced temporal analysis
   │   ├── risk_model.py           # Advanced risk metrics: EGARCH volatility, maximum drawdown, CVaR, etc.
   │   └── hyperparameter.py       # Grid search-based hyperparameter tuning using TimeSeriesSplit
   └── ...
   ```

## Usage

### 1. Configure Hyperparameters

Edit `config.yaml` to set parameters. For example:

```yaml
data_path: "src/stock_data.csv"
batch_size: 32
seq_len: 10
learning_rate: 0.001
num_epochs: 30
model:
  input_size: 59  # Update this number after renaming technical indicator columns
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
```

### 2. Train the Model

Run the training script:

```bash
python src/train.py
```

This script loads data, applies preprocessing (including renaming technical indicator columns), trains the LSTM model, and saves the model weights.

### 3. Evaluate and Generate Reports

- **Single Report Evaluation:**

  ```bash
  python src/evalute.py
  ```

  This script:
  - Loads the model and data.
  - Processes each stock (logging row counts and market value details, and skipping stocks with insufficient data).
  - Computes news factors (sentiment, events) and extra technical factors.
  - Computes advanced risk metrics (EGARCH-based volatility, maximum drawdown, CVaR, risk-adjusted ratio).
  - Binarizes predicted returns using a dynamic threshold to calculate classification metrics (accuracy, precision, recall, specificity, F1-score, AUC, AUPR, error rate, and confusion matrix components), which are printed in a well-formatted table in the terminal.
  - Generates an HTML report for each stock with detailed, styled output.
  - Aggregates individual reports into a single HTML page.

- **Combined Report Generation:**

  ```bash
  python src/multi_eval.py
  ```

  This script combines individual stock reports into one comprehensive HTML report.

- **Hyperparameter Tuning (Optional):**

  ```bash
  python src/hyperparameter.py
  ```

  This script performs grid search using time-series cross-validation and outputs the best configuration.

## Debugging & Improvements

- **Data Quantity:**  
  Stocks with fewer than the required number of rows (based on `seq_len`) are skipped. You can increase the dataset size or reduce `seq_len` if necessary.

- **Class Imbalance:**  
  Some stocks have imbalanced true label distributions, causing metrics like AUC to be undefined. Consider using a dynamic threshold or resampling techniques.

- **Risk Model Warnings:**  
  You may receive a DataScaleWarning from the EGARCH model. Consider rescaling the returns series (e.g., multiplying by 10) or setting `rescale=False` in the `arch_model` initialization.

- **Logging:**  
  Additional logging (e.g., printing row counts, latest and average market values) is added to help verify data quality and factor calculations.

- **Conditional Styling:**  
  The HTML templates use conditional styling to highlight key numeric values in green (for positive effects) or red (for negative effects).

- **Hybrid Metrics Reporting:**  
  Both regression metrics (MSE, RMSE, MAE) and binary classification metrics (Accuracy, Precision, Recall, F1-score, AUC, AUPR, Error Rate, and confusion matrix components) are printed in the terminal in a tabular format.

## Future Improvements and Next Steps

- **Return Forecasting Enhancements:**  
  Experiment with integrating classical multi-factor models (e.g., Fama–French factors) and alternative architectures (e.g., Transformers for time series).

- **Enhanced Risk Modeling:**  
  Refine risk metrics by experimenting with alternative distributions (e.g., Student’s t) in the EGARCH model or adding additional risk measures such as drawdown-based metrics.

- **Improved News Aggregation:**  
  Enhance the aggregation strategy for multiple news items, possibly incorporating temporal weights for consecutive news events.

- **Interactive Dashboard:**  
  Develop a web-based dashboard (using Flask, Django, or Streamlit) for real-time report generation and interactive visualizations.

- **Automated Data Updates:**  
  Build a pipeline to automatically fetch and preprocess new data so that the model and reports remain current.

## Contact

For any questions or feedback, please contact:  
**Kanishk**  
**Kanishkgupta2003@outlook.com**

---
