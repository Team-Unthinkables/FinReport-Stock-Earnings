# FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model

## Overview

FinReport is a research-oriented system designed to forecast stock earnings by analyzing both technical indicators and financial news. It leverages a multi-factor model implemented in PyTorch that integrates three key modules:

- **News Factorization:** Extracts and processes features from financial news—including sentiment analysis using FinBERT and event extraction using AllenNLP with domain-specific keyword rules. It also supports aggregation of multiple news items (with a basic temporal analysis) to compute an overall news factor.
- **Return Forecasting:** Uses historical stock data and technical indicators (both standard and domain-specific) to predict returns via an LSTM model. External proxies (such as market value for size) are used as internal data substitutes for classical Fama–French factors.
- **Risk Assessment:** Evaluates risk using advanced metrics including EGARCH-based volatility forecasting, maximum drawdown, and CVaR, as well as a combined risk-adjusted ratio (expected return divided by volatility).

The final output is a polished HTML report that presents predictions, risk assessments, detailed technical factors, and source attributions in an easily readable format with enhanced styling.

## Features

- **Data Integration:**  
  Combines historical stock data (with technical indicators) with processed news text. Technical indicator column names are automatically renamed for consistency.
  
- **Model Training & Evaluation:**  
  Trains an LSTM-based model in PyTorch with hyperparameters loaded from a YAML configuration file. The system uses time-series data with a fixed sequence length to ensure consistency between training and evaluation.
  
- **Enhanced Feature Engineering:**  
  - **Sentiment Analysis:** Uses FinBERT (via Hugging Face) to compute a sentiment score from news text.
  - **Event Extraction:** Uses AllenNLP’s SRL model (in `advanced_news.py`) with domain-specific keyword rules to extract events and compute an event factor.
  - **Multi-Article Aggregation:** (Optional) Aggregates multiple news items per day via `news_aggregator.py` to yield an overall news effect.
  - **Internal Proxies for Classical Factors:** Uses existing CSV columns (e.g., market_value) to derive factors such as the Size Factor, avoiding the need for external Fama–French data.
  
- **Dynamic Report Generation:**  
  - Generates individual HTML reports for each stock and combines them into a single multi-stock HTML page.
  - Reports are styled with blue bullet labels, with key numeric phrases (e.g., “0.5%”, “1%”) highlighted in green (if positive) or red (if negative), while the remaining text is in black.
  - The report displays detailed factors including Market, Size, Valuation, Profitability, Investment, News Effect, Event, and (if available) RSI, MFI, and BIAS factors.
  
- **Advanced Risk Assessment:**  
  - **EGARCH Volatility Forecasting:** Computes conditional volatility and approximate VaR.
  - **Additional Risk Metrics:** Calculates maximum drawdown and Conditional Value at Risk (CVaR) using historical simulation.
  - **Risk-Adjusted Ratio:** Computes a simple integrated measure (expected return divided by volatility) to assess risk-adjusted performance.
  
- **Modular Design:**  
  The code is organized into separate modules for data loading, preprocessing, model definition, training, evaluation, sentiment analysis, extra factor computation, risk modeling, and report generation.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Kanishk1420/FinReport-Explainable-Stock-Earnings-Forecasting-via-News-Factor
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
   │   ├── evalute.py              # Script to evaluate the model and generate HTML reports (with risk metrics)
   │   ├── multi_eval.py           # Script to generate a combined HTML report for multiple stocks
   │   ├── report_generator.py     # Generates HTML reports using Jinja2 (handles extra factors and risk metrics)
   │   ├── sentiment.py            # FinBERT-based sentiment analysis module
   │   ├── extra_factors.py        # Computes additional domain-specific technical indicators with split descriptions
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

This script loads data, applies preprocessing (including technical column renaming), trains the LSTM model, and saves the model weights.

### 3. Evaluate and Generate Reports

- **Single Report Evaluation:**

  ```bash
  python src/evalute.py
  ```

  This script:
  - Loads the model and data.
  - Processes each stock (skipping those with insufficient data).
  - Computes news factors (sentiment, events) and extra technical factors.
  - Computes advanced risk metrics (volatility via EGARCH, maximum drawdown, CVaR, risk-adjusted ratio).
  - Generates an HTML report for each stock with detailed, styled output.

  This script aggregates individual stock reports into a single HTML page.

- **Hyperparameter Tuning (Optional):**

  ```bash
  python src/hyperparameter.py
  ```

  This script performs grid search using time-series cross-validation and outputs the best configuration.

## Challenges, Bugs, and Solutions

1. **Insufficient Data for Certain Stocks:**  
   The code checks if a stock’s data has at least `seq_len + 1` rows. Stocks lacking sufficient data are skipped.

2. **Non-Numeric Data Issues:**  
   The `select_features` function excludes non-numeric columns (like dates and announcements) to avoid normalization errors.

3. **Model Architecture Mismatch:**  
   Consistent hyperparameters are loaded from `config.yaml` for both training and evaluation to prevent state dictionary mismatches.

4. **Handling Scalar Predictions:**  
   Model predictions are flattened to ensure they are iterable, avoiding iteration errors with 0-d arrays.

5. **Combining Multiple Reports:**  
   A multi-report HTML template and corresponding generator function allow all individual reports to be combined into one comprehensive HTML page.

6. **Advanced Feature Engineering and Styling:**  
   - Sentiment is computed using FinBERT.
   - SRL-based event extraction (with domain-specific keyword rules) computes an event factor.
   - Extra factors (Market, Size, Valuation, etc.) return dictionaries with split descriptions for conditional styling in the HTML report.
   - The HTML templates highlight key numeric values in green or red while keeping the rest of the text in black.

7. **Advanced Risk Assessment:**  
   EGARCH-based volatility forecasting is combined with additional risk metrics (maximum drawdown, CVaR) to provide a comprehensive risk view.

## Future Improvements and Next Steps

- **Return Forecasting Enhancements:**  
  Experiment with integrating classical multi-factor models (e.g., Fama–French factors) and alternative architectures (e.g., Transformers for time series).

- **Enhanced Risk Modeling:**  
  Refine risk metrics by experimenting with alternative distributions (e.g., Student’s t) in the EGARCH model or adding measures like drawdown-based risk metrics.

- **Improved News Aggregation:**  
  Refine the aggregation strategy for multiple news items, possibly incorporating temporal weights for consecutive negative/positive news.

- **Interactive Dashboard:**  
  Consider developing a web-based dashboard (using Flask, Django, or Streamlit) for real-time report generation and interactive visualizations.

- **Automated Data Updates:**  
  Build a pipeline to automatically fetch and preprocess new data so the system remains current.

## Contact

For any questions or feedback, please contact:  
**Kanishk**  
**Kanishkgupta2003@outlook.com**

---
