import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from model import FinReportModel
from data_loader import load_data, split_data
from preprocessing import select_features, normalize_features, rename_technical_columns
from report_generator import generate_html_finreport, save_html_report
from sentiment import get_sentiment_score
from extra_factors import (compute_market_factor, compute_size_factor, compute_valuation_factor, 
                           compute_profitability_factor, compute_investment_factor, compute_news_effect_factor,
                           compute_rsi_factor, compute_mfi_factor, compute_bias_factor)
from risk_model import compute_max_drawdown, compute_volatility
from advanced_news import compute_event_factor
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, average_precision_score,
                             precision_score, recall_score, confusion_matrix)
from jinja2 import Environment, FileSystemLoader
from news_aggregator import aggregate_news_factors
import matplotlib.pyplot as plt        # NEW CODE: Import for plotting
import seaborn as sns                  # NEW CODE: Import for plotting

# ----- Set Up Logging to File and Terminal -----
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("evalute_output.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# ----- Additional Functions -----
def compute_integrated_risk(predicted_return, volatility):
    return predicted_return / volatility if volatility != 0 else np.nan

def compute_cvar(returns, alpha=0.05):
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    return np.mean(sorted_returns[:index])

# ----- Load Configuration -----
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path   = config['data_path']
batch_size  = config['batch_size']
seq_len     = config['seq_len']
model_config = config['model']
input_size  = model_config['input_size']
hidden_size = model_config['hidden_size']
num_layers  = model_config.get('num_layers', 1)
dropout     = model_config.get('dropout', 0.0)

# ----- Load Data and Rename Columns -----
df = load_data(data_path)
df = rename_technical_columns(df)
logger.info("Columns after renaming:")
logger.info(list(df.columns))

stock_list = df['ts_code'].unique()

# ----- Load Model -----
model = FinReportModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
model.load_state_dict(torch.load('finreport_model.pth'))
model.eval()

# ----- Define Dataset -----
class FinDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.features) - self.seq_len)
    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# ----- Initialize Lists for Metrics and Reports -----
all_metrics = []
all_reports = []

os.makedirs('img', exist_ok=True)

# ----- Process Each Stock -----
for stock in stock_list:
    logger.info(f"Processing stock: {stock}")
    df_stock = df[df['ts_code'] == stock]
    
    row_count = len(df_stock)
    logger.info(f"Stock {stock} has {row_count} rows.")
    if row_count <= seq_len:
        logger.info(f"Not enough data for stock {stock} (requires > {seq_len} rows). Skipping.")
        continue

    latest_val = df_stock['market_value'].iloc[-1]
    avg_val = df_stock['market_value'].mean()
    diff_percent = ((latest_val - avg_val) / avg_val) * 100
    logger.info(f"For stock {stock}: Latest market value = {latest_val}, Average = {avg_val}, Difference = {diff_percent:.1f}%")
    
    _, test_df = split_data(df_stock)
    if len(test_df) <= seq_len:
        logger.info(f"Not enough test data for stock {stock} (requires > {seq_len} rows). Skipping.")
        continue

    test_features, test_labels = select_features(test_df)
    unique, counts = np.unique(test_labels, return_counts=True)
    logger.info("True label distribution: " + str(dict(zip(unique, counts))))
    mean_val = np.mean(test_labels)
    median_val = np.median(test_labels)
    variance_val = np.var(test_labels)
    logger.info(f"Target Return Distribution - Mean: {mean_val:.3f}, Median: {median_val:.3f}, Variance: {variance_val:.3f}")
    logger.info("Shape of features: " + str(test_features.shape))
    logger.info("First row of features: " + str(test_features[0]))
    test_features, scaler = normalize_features(test_features)
    dataset = FinDataset(test_features, test_labels, seq_len)
    if len(dataset) <= 0:
        logger.info(f"Dataset for stock {stock} is empty after processing. Skipping.")
        continue

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    with torch.no_grad():
        for x_batch, _ in loader:
            preds = model(x_batch)
            all_predictions.extend(preds.cpu().numpy().flatten())
    all_predictions = np.array(all_predictions)

    # NEW CODE: Log raw prediction statistics.
    unique_preds, counts_preds = np.unique(all_predictions, return_counts=True)
    logger.info(f"Raw predicted values distribution: {dict(zip(unique_preds, counts_preds))}")
    logger.info(f"Mean predicted value: {np.mean(all_predictions):.3f}, Median: {np.median(all_predictions):.3f}")

    predicted_return = all_predictions[0]

    # --- UPDATED CODE: Compute Classification Metrics using Dynamic Threshold ---
    q25, q75 = np.percentile(all_predictions, [25, 75])
    threshold = (q25 + q75) / 2
    logger.info(f"Dynamic threshold for binarization (midpoint between 25th and 75th percentiles): {threshold:.3f}")
    
    # NEW CODE: Log Raw Predictions and True Labels before binarization
    logger.info(f"First 10 Raw Predictions for {stock}: {all_predictions[:10]}")
    logger.info(f"First 10 True Labels for {stock}: {test_labels[seq_len:seq_len+10]}")
    
    # Use the same threshold for both predictions and true labels
    binary_preds = (all_predictions > threshold).astype(int)
    binary_true = (test_labels[seq_len:] > threshold).astype(int)

    # NEW CODE: Log binary predictions distribution.
    unique_preds, counts_preds = np.unique(binary_preds, return_counts=True)
    logger.info(f"Binary Predictions Distribution for {stock}: {dict(zip(unique_preds, counts_preds))}")

    unique_labels = np.unique(binary_true)  # NEW CODE: Check unique labels
    if len(unique_labels) < 2:
        logger.info(f"Warning: Only one class present in y_true for stock {stock}.")
        auc = 0.5  # fallback value for AUC (random performance)
        aupr = 0.5 # fallback value for AUPR
        tn, fp, fn, tp = 0, 0, 0, 0
    else:
        auc = roc_auc_score(binary_true, all_predictions)
        aupr = average_precision_score(binary_true, all_predictions)
        # Compute confusion matrix
        cm = confusion_matrix(binary_true, binary_preds, labels=[0, 1])
        logger.info(f"Confusion Matrix for stock {stock}:\n{cm}")
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            # Plot and save confusion matrix as an image
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix for stock {stock}")
            image_filename = os.path.join('img', f'confusion_matrix_{stock}.png')
            plt.savefig(image_filename)
            plt.close()
            logger.info(f"Confusion matrix image saved as {image_filename}")
        else:
            logger.warning(f"Confusion Matrix for stock {stock} is not 2x2 due to class imbalance. Using fallback values.")
            tn, fp, fn, tp = 0, 0, 0, 0

    accuracy = accuracy_score(binary_true, binary_preds)
    precision = precision_score(binary_true, binary_preds, zero_division=0)
    recall = recall_score(binary_true, binary_preds, zero_division=0)
    f1 = f1_score(binary_true, binary_preds, zero_division=0)
    error_rate = 1 - accuracy
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("----------------------------------------------------------")
    print(f"{'Metric':<20} | {'Value':>8}")
    print("----------------------------------------------------------")
    print(f"{'Accuracy':<20} | {accuracy:>8.3f}")
    print(f"{'Precision':<20} | {precision:>8.3f}")
    print(f"{'Recall (Sensitivity)':<20} | {recall:>8.3f}")
    print(f"{'Specificity':<20} | {specificity:>8.3f}")
    print(f"{'F1-score':<20} | {f1:>8.3f}")
    print(f"{'AUC':<20} | {auc:>8.3f}")
    print(f"{'AUPR':<20} | {aupr:>8.3f}")
    print(f"{'Error Rate':<20} | {error_rate:>8.3f}")
    print("----------------------------------------------------------")

    # Compute regression metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse_value = mean_squared_error(test_labels[seq_len:], all_predictions)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(test_labels[seq_len:], all_predictions)

    # Store metrics for the current stock
    all_metrics.append({
        'Stock': stock,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-score': f1,
        'AUC': auc,
        'AUPR': aupr,
        'Error Rate': error_rate,
        'MSE': mse_value,   # regression metrics stored here
        'RMSE': rmse_value,
        'MAE': mae_value
    })

    news_summary = str(test_df.iloc[-1]["announcement"])
    sentiment_score = get_sentiment_score(news_summary)
    logger.info(f"Sentiment Score for {stock}: {sentiment_score:.4f}")
    news_source = "财联社"

    market_factor = compute_market_factor(df_stock)
    size_factor = compute_size_factor(df_stock)
    valuation_factor = compute_valuation_factor(df_stock)
    profitability_factor = compute_profitability_factor(df_stock)
    investment_factor = compute_investment_factor(df_stock)
    news_effect_factor = compute_news_effect_factor(sentiment_score)
    event_factor = compute_event_factor(news_summary)
    rsi_factor = compute_rsi_factor(df_stock)
    mfi_factor = compute_mfi_factor(df_stock)
    bias_factor = compute_bias_factor(df_stock)

    returns = df_stock['close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    try:
        vol, var_95 = compute_volatility(returns)
        max_dd = compute_max_drawdown(returns)
        cvar = compute_cvar(returns, alpha=0.05)
        risk_adjusted_ratio = compute_integrated_risk(predicted_return, vol)
        risk_metrics = {
            "volatility": f"{vol:.2f}",
            "var_95": f"{var_95:.2f}",
            "max_drawdown": f"{max_dd:.2f}",
            "cvar": f"{cvar:.2f}",
            "risk_adjusted_ratio": f"{risk_adjusted_ratio:.2f}"
        }
    except ValueError as e:
        risk_metrics = {"error": str(e)}

    risk_assessment = "Low/Moderate Risk" if predicted_return < 2.0 else "High Risk"
    overall_trend = "Positive"
    final_summary = ("Based on current technical indicators and recent news, the outlook for this stock appears positive. "
                     "However, market conditions remain volatile, so caution is advised.")
    date_str = "Today"

    report_html = generate_html_finreport(
        stock_symbol=stock,
        date_str=date_str,
        news_summary=news_summary,
        news_source=news_source,
        market_factor=market_factor,
        size_factor=size_factor,
        valuation_factor=valuation_factor,
        profitability_factor=profitability_factor,
        investment_factor=investment_factor,
        news_effect_factor=news_effect_factor,
        event_factor=event_factor,
        rsi_factor=rsi_factor,
        mfi_factor=mfi_factor,
        bias_factor=bias_factor,
        risk_assessment=risk_assessment,
        overall_trend=overall_trend,
        news_effect_score=news_effect_factor["value"],
        risk_metrics=risk_metrics,
        template_path="templates/report_template.html"
    )
    all_reports.append(report_html)
    logger.info(f"Report for {stock} generated.")
    target_mean = np.mean(all_predictions)
    target_median = np.median(all_predictions)
    target_variance = np.var(all_predictions)
    logger.info(f"Target Return Distribution - Mean: {target_mean:.3f}, Median: {target_median:.3f}, Variance: {target_variance:.3f}")

# ----- After Loop: Print Combined Metrics Table -----
if all_metrics:
    df_metrics = pd.DataFrame(all_metrics)
    logger.info("\nOverall Performance Metrics for All Stocks:")
    logger.info(df_metrics.to_string(index=False))
    
    # ----- Generate Heatmap for Regression Metrics (MSE, RMSE, MAE) -----
    regression_metrics = df_metrics[['Stock', 'MSE', 'RMSE', 'MAE']]
    regression_metrics.set_index('Stock', inplace=True)
    plt.figure(figsize=(10, len(regression_metrics) * 0.5))
    sns.heatmap(regression_metrics, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Regression Metrics (MSE, RMSE, MAE) for Each Stock")
    heatmap_reg_filename = os.path.join('img', "regression_metrics_heatmap.png")
    plt.savefig(heatmap_reg_filename)
    plt.close()
    logger.info(f"Regression metrics heatmap saved as {heatmap_reg_filename}")
    
    # ----- Generate Heatmap for Classification Metrics (AUC, AUPR) -----
    classification_metrics = df_metrics[['Stock', 'AUC', 'AUPR']]
    classification_metrics.set_index('Stock', inplace=True)
    
    plt.figure(figsize=(10, len(classification_metrics) * 0.5))
    sns.heatmap(classification_metrics, annot=True, fmt=".3f", cmap="coolwarm")
    plt.title("Classification Metrics (AUC, AUPR) for Each Stock")
    heatmap_class_filename = os.path.join('img', "classification_metrics_heatmap.png")
    plt.savefig(heatmap_class_filename)
    plt.close()
    logger.info(f"Classification metrics heatmap saved as {heatmap_class_filename}")
else:
    logger.info("No performance metrics were computed.")

if all_reports:
    from report_generator import generate_multi_report_html
    final_html = generate_multi_report_html(all_reports, template_path="templates/multi_report_template.html")
    save_html_report(final_html, output_filename="finreport_combined.html")
else:
    logger.info("No reports were generated due to insufficient data across stocks.")
logger.info("Evaluation completed.")