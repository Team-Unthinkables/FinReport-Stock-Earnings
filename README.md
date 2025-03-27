# FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model

## Overview

FinReport is a research-oriented system designed to forecast stock earnings by analyzing both technical indicators and financial news. It leverages a multi-factor model implemented in PyTorch that integrates three key modules:

- **News Factorization:**  
  Extracts and processes features from financial news—including sentiment analysis using FinBERT and event extraction via AllenNLP with domain-specific keyword rules. The system also supports aggregation of multiple news items over time to yield an overall news factor.

- **Return Forecasting:**  
  Uses historical stock data (with automatically renamed technical indicator columns) and domain-specific proxies (e.g., market value for size) to predict returns via an LSTM model. The model hyperparameters (including sequence length, hidden size, dropout, etc.) are loaded from a YAML configuration file.

- **Risk Assessment:**  
  Evaluates risk using advanced metrics. It computes EGARCH-based volatility estimates, maximum drawdown, and Conditional Value at Risk (CVaR) via historical simulation. A combined risk-adjusted ratio (expected return divided by volatility) is also computed to assess performance from a risk perspective.

Additionally, the system computes comprehensive regression metrics (MSE, RMSE, MAE, R²) to evaluate prediction accuracy. The model's performance is visualized through detailed distribution plots and aggregate summaries.

The final output is a polished HTML report that presents predictions, detailed technical factors (with conditional styling for key numeric values), and risk metrics. In parallel, comprehensive performance metrics are printed in the terminal and saved in heatmaps for further analysis.

## Code Structure and Workflow

### Core Components

1. **Data Processing Pipeline:**
   - `data_loader.py`: Loads CSV data and handles train/test splitting
   - `preprocessing.py`: Normalizes features, renames technical columns, and selects relevant features
   - `Tech_Indicators.py`: Generates technical indicators from raw stock price data

2. **News Analysis:**
   - `sentiment.py`: Implements sentiment analysis using FinBERT for financial news texts
   - `advanced_news.py`: Uses AllenNLP's SRL model to extract events from news and compute event factors
   - `news_aggregator.py`: Aggregates sentiment and event factors from multiple news items

3. **Factor Generation:**
   - `extra_factors.py`: Computes domain-specific factors (market, size, valuation, profitability, etc.)
   - Generates numeric factor values and human-readable descriptions for reports

4. **Model Architecture:**
   - `model.py`: Defines the LSTM-based neural network with batch normalization and dropout
   - Supports Monte Carlo dropout for uncertainty estimation
   - Includes proper weight initialization for improved training stability

5. **Training System:**
   - `train.py`: Implements the training loop with validation, early stopping, and learning rate scheduling
   - Includes gradient clipping to prevent exploding gradients
   - Visualizes training progress and model performance

6. **Evaluation Framework:**
   - `evalute.py`: Comprehensive evaluation system with regression metrics and visualizations
   - `improved_metrics.py`: Advanced metrics calculations and visualization utilities
   - Generates individual and aggregate performance visualizations

7. **Risk Modeling:**
   - `risk_model.py`: Implements EGARCH volatility forecasting, maximum drawdown calculation, and CVaR estimation
   - Computes integrated risk metrics to assess risk-adjusted performance

8. **Report Generation:**
   - `report_generator.py`: Creates HTML reports using Jinja2 templates
   - Generates both individual stock reports and combined multi-stock reports
   - Implements dynamic risk assessment text generation

9. **Hyperparameter Optimization:**
   - `hyperparameter.py`: Performs grid search for optimal hyperparameters
   - Uses time-series cross-validation to ensure robust parameter selection

### Configuration

The system is configured via `config.yaml` which specifies:
- Data path
- Batch size
- Sequence length
- Learning rate
- Epoch count
- Model architecture parameters (input size, hidden size, layers, dropout)

## Workflow Process

1. **Data Loading and Preprocessing:**
   - Load raw stock data with `data_loader.py`
   - Generate technical indicators with `Tech_Indicators.py` if not already present
   - Preprocess features with `preprocessing.py` (normalize, rename columns)

2. **Model Training:**
   - Configure hyperparameters in `config.yaml`
   - Run `train.py` which:
     - Loads preprocessed data
     - Splits into training/validation sets
     - Trains the LSTM model with early stopping
     - Applies learning rate scheduling
     - Saves the best model
     - Generates training visualizations

3. **Model Evaluation:**
   - Run `evalute.py` which:
     - Loads the trained model and test data
     - For each stock:
       - Generates predictions
       - Computes factor values from news using `extra_factors.py` and `sentiment.py`
       - Calculates risk metrics with `risk_model.py`
       - Creates visualizations of prediction distributions
       - Generates HTML reports with `report_generator.py`
     - Creates aggregate visualizations and metrics
     - Combines individual reports into a comprehensive multi-stock report

4. **Hyperparameter Tuning (Optional):**
   - Run `hyperparameter.py` to find optimal parameters
   - Results can be used to update `config.yaml`

## Model Performance Results

Based on the regression_evaluation_results.csv, the model demonstrates strong predictive capabilities:

- **Strong Correlation**: Achieved a correlation of r = 0.948 between predictions and true values.

- **Error Distribution**: Errors are well-distributed with most errors falling between -0.5 and 0.5.

- **Key Metrics**:
  - Average RMSE = 0.2546 (reasonably low)
  - Average MAE = 0.2433 (consistent with RMSE)
  - Average R² = 0.5515 (the model explains about 55% of the variance)

- **Stock-Specific Performance**:
  - Top performers: Several stocks show R² values above 0.95, including 000333.SZ (0.994), 002352.SZ (0.990), 601669.SH (0.988), and 600519.SH (0.992)
  - Most stocks show strong predictive performance with positive R² values
  - Only a few stocks show poor performance, primarily 601727.SH with a negative R²

The regression approach has proven to be significantly more effective than classification, providing more nuanced and accurate predictions for stock returns.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Team-Unthinkables/FinReport-Stock-Earnings.git
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
   ├── regression_evaluation_results.csv  # Performance metrics for each stock
   ├── src/
   │   ├── config.yaml             # Configuration for hyperparameters and file paths
   │   ├── stock_data.csv          # Historical stock data and news factors
   │   ├── __init__.py
   │   ├── data_loader.py          # Loads and parses CSV data
   │   ├── preprocessing.py        # Feature extraction, technical column renaming, and normalization
   │   ├── Tech_Indicators.py      # Generates technical indicators from raw price data
   │   ├── model.py                # PyTorch LSTM model definition
   │   ├── train.py                # Script to train the model with early stopping and validation
   │   ├── evalute.py              # Script to evaluate the model and generate HTML reports
   │   ├── improved_metrics.py     # Advanced metrics calculations and visualization utilities
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
   └── img/                        # Directory for generated heatmaps and visualizations
   ```

## Usage

### 1. Configure Hyperparameters

Edit `src/config.yaml` to set parameters. Current optimal settings:

```yaml
data_path: "src/stock_data.csv"
batch_size: 32
seq_len: 10  # Optimal from hyperparameter search
learning_rate: 0.0010  # Optimal from hyperparameter search
num_epochs: 50  # Increased to allow for early stopping
model:
  input_size: 59
  hidden_size: 128  # Optimal from hyperparameter search
  num_layers: 3  # Optimal from hyperparameter search
  dropout: 0.2  # Optimal from hyperparameter search
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
- Calculates comprehensive regression metrics (MSE, RMSE, MAE, R²)
- Generates prediction distribution visualizations
- Creates aggregate visualizations summarizing performance across all stocks
- Creates individual HTML reports for each stock
- Combines reports into a single multi-stock HTML page
- Saves performance metric heatmaps

### 4. Tune Hyperparameters (Optional)

```bash
python src/hyperparameter.py
```

This script performs grid search using time-series cross-validation to find optimal hyperparameters.

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

**MD Azlan**  
**azlan04.md@gmail.com**