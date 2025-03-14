# FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model

## Overview

FinReport is a research-oriented system designed to forecast stock earnings by analyzing both technical indicators and financial news. It leverages a multi-factor model implemented in PyTorch that integrates three key modules:

- **News Factorization:**  
  Extracts and processes features from financial news—including sentiment analysis using FinBERT and event extraction via AllenNLP with domain-specific keyword rules. The system also supports (optional) aggregation of multiple news items over time to yield an overall news factor.

- **Return Forecasting:**  
  Uses historical stock data (with automatically renamed technical indicator columns) and domain-specific proxies (e.g., market value for size) to predict returns via an LSTM model. The model hyperparameters (including sequence length, hidden size, dropout, etc.) are loaded from a YAML configuration file.

- **Risk Assessment:**  
  Evaluates risk using advanced metrics. It computes EGARCH-based volatility estimates, maximum drawdown, and Conditional Value at Risk (CVaR) via historical simulation. A combined risk-adjusted ratio (expected return divided by volatility) is also computed to assess performance from a risk perspective.

Additionally, the system computes both regression metrics (MSE, RMSE, MAE) and binary classification metrics (Accuracy, Precision, Recall, F1-score, AUC, AUPR, Error Rate) using a dynamic threshold (based on the interquartile midpoint) for binarization. Confusion matrices are logged for each stock to better understand misclassifications.

The final output is a polished HTML report that presents predictions, detailed technical factors (with conditional styling for key numeric values), and risk metrics. In parallel, comprehensive performance metrics are printed in the terminal and saved in heatmaps for further analysis.

## Features

- **Data Integration:**  
  Combines historical stock data with technical indicators (whose column names are automatically renamed for consistency) and processed news text.

- **Model Training & Evaluation:**  
  Trains an LSTM-based model using time-series data with a fixed sequence length. Hyperparameters are loaded from a YAML file, and evaluation includes both regression and binary classification metrics.

- **Enhanced Feature Engineering:**  
  - **Sentiment Analysis:** Uses FinBERT (via Hugging Face) to compute a sentiment score.
  - **Event Extraction:** Uses AllenNLP’s SRL model (with domain-specific keyword rules) to compute an event factor.
  - **Extra Factors:** Internal proxies (e.g., market value) are used to compute factors such as Market, Size, Valuation, Profitability, Investment, News Effect, and (if available) RSI, MFI, and BIAS factors.  
  - **Dynamic Binarization:** A dynamic threshold—computed as the midpoint between the 25th and 75th percentiles of predicted returns—is used to binarize outputs for computing classification metrics.

- **Advanced Risk Assessment:**  
  - **EGARCH Volatility Forecasting:** Computes conditional volatility and approximate VaR.
  - **Additional Risk Metrics:** Calculates maximum drawdown and CVaR (Expected Shortfall) using historical simulation.
  - **Risk-Adjusted Ratio:** A simple measure (expected return divided by volatility) is computed.

- **Dynamic Report Generation:**  
  - Generates individual HTML reports for each stock and combines them into a single multi-stock HTML page.
  - Reports use conditional styling: bullet labels are blue, while key numeric values (e.g., “1%”, “0.5%”) are highlighted in green (for positive effects) or red (for negative effects).

- **Logging & Visualization:**  
  - Detailed logging is implemented (row counts, class distributions, raw prediction statistics, confusion matrices) to aid in debugging and performance analysis.
  - Terminal output includes a tabular summary of both regression and classification metrics.
  - (Optional) Heatmaps of metrics (e.g., MSE, RMSE, MAE; AUC, AUPR) can be generated and saved as images for further analysis.

- **Modular Design:**  
  The code is organized into modules for data loading, preprocessing, model definition, training, evaluation, sentiment analysis, extra factor computation, risk modeling, hyperparameter tuning, and report generation.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Kanishk1420/FinReport-Explainable-Stock-Earnings-Forecasting-via-News-Factor
   cd FinReport
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7+ installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include: `pandas`, `numpy`, `torch`, `jinja2`, `pyyaml`, `transformers`, `arch`, `allennlp`, `allennlp-models`, `seaborn`, and others as listed in `requirements.txt`.

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
   │   ├── evalute.py              # Script to evaluate the model and generate HTML reports with detailed metrics and logging
   │   ├── multi_eval.py           # Script to generate a combined HTML report for multiple stocks
   │   ├── report_generator.py     # Generates HTML reports using Jinja2 (handles extra factors, risk metrics, and conditional styling)
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
  input_size: 59  # Update this number if technical indicator columns change
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

  The evaluation script:
  - Loads the model and data.
  - Processes each stock (skipping those with insufficient data).
  - Computes raw predictions, then applies a **dynamic threshold** (the midpoint between the 25th and 75th percentiles) to binarize both predictions and true labels.
  - Logs detailed statistics (row counts, class distributions, raw predictions, confusion matrices, and target return distributions).
  - Computes both regression metrics (MSE, RMSE, MAE) and classification metrics (Accuracy, Precision, Recall, F1-score, AUC, AUPR, Error Rate).
  - Generates individual HTML reports (with styled extra factors and risk metrics) and combines them into one multi-stock HTML page.
  - Saves additional performance visualizations (e.g., heatmaps) if configured.

- **Hyperparameter Tuning:**

  ```bash
  python src/hyperparameter.py
  ```

  This script performs grid search using time-series cross-validation over various hyperparameter combinations (learning rate, hidden size, number of layers, sequence length, dropout) and outputs the best configuration based on average validation loss.

## Performance & Debugging

- **Dynamic Thresholding:**  
  Instead of hardcoding a 0 threshold, the evaluation script calculates a dynamic binarization threshold using the midpoint of the 25th and 75th percentiles of the predicted returns. This helps adjust for class imbalances in each stock.

- **Logging and Metrics:**  
  The evaluation script logs raw prediction distributions, confusion matrices, and detailed descriptive statistics (mean, median, variance) for target returns. These logs help diagnose if the model struggles on certain stocks due to insufficient data or class imbalance.

- **Fallbacks for Undefined Metrics:**  
  For stocks with only one class present, fallback values (e.g., AUC = 0.5) are used to prevent errors during metric computation.

- **Visualization:**  
  (Optional) Heatmaps for regression and classification metrics can be generated and saved as images for visual performance analysis.

## Future Improvements

- **Return Forecasting Enhancements:**  
  Integrate classical multi-factor approaches (e.g., Fama–French factors) or alternative architectures (e.g., Transformers) for improved return predictions.

- **Risk Modeling Refinements:**  
  Experiment with alternative distributions in the EGARCH model, incorporate additional risk metrics, or visualize risk dynamics over time.

- **Improved Class Balancing:**  
  Consider dynamic binarization thresholds, resampling techniques, or a hybrid approach combining regression and classification metrics to address class imbalances.

- **Interactive Dashboard:**  
  Develop a web-based dashboard for real-time analysis and interactive visualization of model performance and risk metrics.

- **Automated Data Pipeline:**  
  Build a pipeline to automatically update and preprocess new stock data to keep the model and reports current.

## Contact

For any questions or feedback, please contact:  
**Kanishk**  
**Kanishkgupta2003@outlook.com**

---
