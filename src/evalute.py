# src/evalute.py
import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import FinReportModel
from data_loader import load_data, split_data
from preprocessing import select_features, normalize_features
from report_generator import generate_html_finreport, save_html_report
from sentiment import get_sentiment_score
from extra_factors import (compute_market_factor, compute_size_factor, compute_valuation_factor, compute_profitability_factor, compute_investment_factor, compute_news_effect_factor,
                           compute_rsi_factor, compute_mfi_factor, compute_bias_factor)
from risk_model import compute_max_drawdown, compute_volatility
from advanced_news import compute_event_factor

def compute_integrated_risk(predicted_return, volatility):
    # Placeholder implementation, replace with actual logic
    return predicted_return / volatility

def compute_cvar(returns, alpha=0.05):
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    return np.mean(sorted_returns[:index])
from news_aggregator import aggregate_news_factors
from jinja2 import Environment, FileSystemLoader

# Load configuration
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

df = load_data(data_path)
stock_list = df['ts_code'].unique()

model = FinReportModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
model.load_state_dict(torch.load('finreport_model.pth'))
model.eval()

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

all_reports = []

for stock in stock_list:
    print(f"Processing stock: {stock}")
    df_stock = df[df['ts_code'] == stock]
    
    # Log the row count for this stock
    row_count = len(df_stock)
    print(f"Stock {stock} has {row_count} rows.")
    
    if row_count <= seq_len:
        print(f"Not enough data for stock {stock} (requires > {seq_len} rows). Skipping.")
        continue
    
    # Log latest and average market values
    latest_val = df_stock['market_value'].iloc[-1]
    avg_val = df_stock['market_value'].mean()
    diff_percent = ((latest_val - avg_val) / avg_val) * 100
    print(f"For stock {stock}: Latest market value = {latest_val}, Average = {avg_val}, Difference = {diff_percent:.1f}%")
    
    _, test_df = split_data(df_stock)
    
    if len(test_df) <= seq_len:
        print(f"Not enough test data for stock {stock} (requires > {seq_len} rows). Skipping.")
        continue
    
    test_features, test_labels = select_features(test_df)
    print("Shape of features:", test_features.shape)
    print("First row of features:", test_features[0])
    test_features, scaler = normalize_features(test_features)
    dataset = FinDataset(test_features, test_labels, seq_len)
    if len(dataset) <= 0:
        print(f"Dataset for stock {stock} is empty after processing. Skipping.")
        continue

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    with torch.no_grad():
        for x_batch, _ in loader:
            preds = model(x_batch)
            all_predictions.extend(preds.cpu().numpy().flatten())
    all_predictions = np.array(all_predictions)
    predicted_return = all_predictions[0]
    
    news_summary = str(test_df.iloc[-1]["announcement"])
    
    sentiment_score = get_sentiment_score(news_summary)
    print(f"Sentiment Score for {stock}: {sentiment_score:.4f}")
    news_source = "财联社"
    
    market_factor = compute_market_factor(df_stock)
    size_factor = compute_size_factor(df_stock)
    
    # Temporary logging to verify size factor calculation
    print("Size Factor:", size_factor)
    
    valuation_factor = compute_valuation_factor(df_stock)
    profitability_factor = compute_profitability_factor(df_stock)
    investment_factor = compute_investment_factor(df_stock)
    news_effect_factor = compute_news_effect_factor(sentiment_score)
    event_factor = compute_event_factor(news_summary)
    
    rsi_factor = compute_rsi_factor(df_stock)
    mfi_factor = compute_mfi_factor(df_stock)
    bias_factor = compute_bias_factor(df_stock)
    
    # Compute returns (assuming 'close' column exists; adjust if needed)
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
        risk_metrics=risk_metrics,   # New parameter here
        template_path="templates/report_template.html"
    )
    all_reports.append(report_html)
    print(f"Report for {stock} generated.")

if all_reports:
    from report_generator import generate_multi_report_html
    final_html = generate_multi_report_html(all_reports, template_path="templates/multi_report_template.html")
    save_html_report(final_html, output_filename="finreport_combined.html")
else:
    print("No reports were generated due to insufficient data across stocks.")
