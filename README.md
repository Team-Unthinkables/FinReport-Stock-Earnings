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
  - **Event Extraction:** Uses AllenNLP's SRL model (with domain-specific keyword rules) to compute an event factor.
  - **Extra Factors:** Internal proxies (e.g., market value) are used to compute factors such as Market, Size, Valuation, Profitability, Investment, News Effect, and (if available) RSI, MFI, and BIAS factors.  
  - **Dynamic Binarization:** A dynamic threshold—computed as the midpoint between the 25th and 75th percentiles of predicted returns—is used to binarize outputs for computing classification metrics.

- **Advanced Risk Assessment:**  
  - **EGARCH Volatility Forecasting:** Computes conditional volatility and approximate VaR.
  - **Additional Risk Metrics:** Calculates maximum drawdown and CVaR (Expected Shortfall) using historical simulation.
  - **Risk-Adjusted Ratio:** A simple measure (expected return divided by volatility) is computed.

- **Dynamic Report Generation:**  
  - Generates individual HTML reports for each stock and combines them into a single multi-stock HTML page.
  - Reports use conditional styling: bullet labels are blue, while key numeric values (e.g., "1%", "0.5%") are highlighted in green (for positive effects) or red (for negative effects).

- **Logging & Visualization:**  
  - Detailed logging is implemented (row counts, class distributions, raw prediction statistics, confusion matrices, and target return distributions) to aid in debugging and performance analysis.
  - Terminal output includes a tabular summary of both regression and classification metrics.
  - Heatmaps of metrics (MSE, RMSE, MAE; AUC, AUPR) are generated and saved as images for visual performance analysis.
  - Prediction distribution visualizations help analyze how well predictions match ground truth labels.

- **Modular Design:**  
  The code is organized into modules for data loading, preprocessing, model definition, training, evaluation, sentiment analysis, extra factor computation, risk modeling, hyperparameter tuning, and report generation.

## Recent Improvements

- **Enhanced LSTM Model Architecture:**
  - Added batch normalization for improved training stability
  - Implemented proper weight initialization techniques
  - Added support for Monte Carlo dropout to estimate prediction uncertainty

- **Advanced Training Process:**
  - Implemented early stopping with validation to prevent overfitting
  - Added learning rate scheduling to optimize convergence
  - Incorporated gradient clipping to prevent exploding gradients
  - Added comprehensive visualizations of training metrics

- **Improved Evaluation:**
  - Implemented robust threshold optimization that works well with limited data
  - Added threshold placement analysis to ensure optimal binary classification
  - Added visualization of prediction distributions to analyze model performance
  - Enhanced handling of class imbalance and limited data scenarios

- **Performance Achievements:**
  - Achieved excellent regression performance with R² scores of 0.997+
  - Perfect classification metrics (AUC/AUPR = 1.0) for multiple stocks
  - Reduced overfitting through proper regularization and early stopping

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
   ├── src/
   │   ├── config.yaml             # Configuration for hyperparameters and file paths
   │   ├── stock_data.csv          # Historical stock data and news factors
   │   ├── __init__.py
   │   ├── data_loader.py          # Loads and parses CSV data
   │   ├── preprocessing.py        # Feature extraction, technical column renaming, and normalization
   │   ├── model.py                # PyTorch LSTM model definition
   │   ├── train.py                # Script to train the model with early stopping and validation
   │   ├── evalute.py              # Script to evaluate the model and generate HTML reports
   │   ├── report_generator.py     # Generates HTML reports using Jinja2
   │   ├── sentiment.py            # FinBERT-based sentiment analysis module
   │   ├── extra_factors.py        # Computes additional domain-specific technical indicators
   │   ├── advanced_news.py        # SRL-based event extraction and event factor computation
   │   ├── news_aggregator.py      # Aggregates multiple news items for enhanced temporal analysis
   │   ├── risk_model.py           # Advanced risk metrics calculation
   │   └── hyperparameter.py       # Grid search-based hyperparameter tuning
   ├── templates/
   │   ├── report_template.html    # HTML template for individual reports
   │   └── multi_report_template.html  # HTML template for combined reports
   ├── models/                     # Directory for saved model weights
   ├── plots/                      # Directory for training and evaluation plots
   └── img/                        # Directory for generated heatmaps and visualizations
   ```

## Usage

### 1. Configure Hyperparameters

Edit `src/config.yaml` to set parameters. For example:

```yaml
data_path: "src/stock_data.csv"
batch_size: 32
seq_len: 10
learning_rate: 0.0010
num_epochs: 50
model:
  input_size: 59
  hidden_size: 128
  num_layers: 3
  dropout: 0.2
```

### 2. Train the Model

Run the training script:

```bash
python src/train.py
```

This script:
- Loads and preprocesses data
- Creates training and validation splits
- Trains the LSTM model with early stopping
- Saves the best model weights to the models/ directory
- Generates training visualization plots

### 3. Evaluate and Generate Reports

```bash
python src/evalute.py
```

The evaluation script:
- Loads the trained model and test data
- Processes each stock and generates predictions
- Applies dynamic thresholding for binary classification
- Calculates comprehensive metrics including confusion matrices
- Generates prediction distribution visualizations
- Creates individual HTML reports for each stock
- Combines reports into a single multi-stock HTML page
- Saves performance metric heatmaps

### 4. Tune Hyperparameters (Optional)

```bash
python src/hyperparameter.py
```

This script performs grid search using time-series cross-validation to find optimal hyperparameters.

## Performance & Debugging

- **Early Stopping and Validation:**  
  The training process monitors validation loss and stops when no further improvement is observed, preventing overfitting and ensuring model generalization.

- **Dynamic Thresholding:**  
  Instead of using a fixed threshold, the evaluation script calculates an optimal threshold for each stock based on F1-score maximization. For stocks with limited data, it uses a robust threshold estimation technique.

- **Prediction Distribution Analysis:**  
  Visualization of prediction distributions helps analyze how well the model differentiates between positive and negative returns, and how appropriate the classification threshold is.

- **Comprehensive Metrics:**  
  Both regression metrics (MSE, RMSE, MAE, R²) and classification metrics (Accuracy, Precision, Recall, F1-score, AUC, AUPR) are reported to provide a holistic view of model performance.

- **Robust Handling of Edge Cases:**  
  Special handling is implemented for stocks with limited data or class imbalance, ensuring reliable metrics even in challenging scenarios.

## Future Improvements

- **Cross-Validation Implementation:**  
  Implement k-fold cross-validation for more robust performance estimation and to reduce sensitivity to specific train-test splits.

- **Return Forecasting Enhancements:**  
  Integrate classical multi-factor approaches (e.g., Fama–French factors) or alternative architectures (e.g., Transformers) for improved return predictions.

- **Risk Modeling Refinements:**  
  Experiment with alternative distributions in the EGARCH model, incorporate additional risk metrics, or visualize risk dynamics over time.

- **Interactive Dashboard:**  
  Develop a web-based dashboard for real-time analysis and interactive visualization of model performance and risk metrics.

- **Automated Data Pipeline:**  
  Build a pipeline to automatically update and preprocess new stock data to keep the model and reports current.

## Contact

For any questions or feedback, please contact:  
**Kanishk**  
**Kanishkgupta2003@outlook.com**