# src/extra_factors.py
import numpy as np
import pandas as pd

def compute_market_factor(df):
    """
    Compute Market Factor using the volatility indicator from the pct_chg column.
    - If the standard deviation (volatility) exceeds a defined threshold, compute a potential decline.
    - If the volatility is exceptionally low (below a lower threshold), compute a potential benefit.
    - Otherwise, indicate that market conditions appear stable.
    
    Returns a dictionary with keys: value, desc_base, desc_highlight.
    """
    if 'pct_chg' not in df.columns or len(df) == 0:
        return {"value": 0.0, "desc_base": "No market data available.", "desc_highlight": ""}
    
    volatility = df['pct_chg'].std()  # volatility as a decimal (e.g., 0.04 means 4%)
    lower_threshold = 0.02  # exceptionally low volatility (2%)
    threshold = 0.04        # our standard threshold (4%)
    vol_percent = volatility * 100  # convert to percentage
    
    if volatility > threshold:
        # High volatility: compute potential decline proportionally.
        potential_decline = (volatility / threshold) * 2.0  # e.g., if volatility == threshold, then 2%
        return {
            "value": -potential_decline,
            "desc_base": f"High volatility detected <span style='color: rgb(255, 19, 19);'>{vol_percent:.1f}%</span>, potential decline of ",
            "desc_highlight": f"<span style='color: rgb(255, 19, 19);'>{potential_decline:.1f}%</span>."
        }
    elif volatility < lower_threshold:
        # Exceptionally low volatility: assign a potential benefit.
        # For instance, the lower the volatility, the higher the potential benefit up to a cap.
        potential_increase = (lower_threshold / volatility) * 0.5  # adjust scaling as needed
        return {
            "value": potential_increase,
            "desc_base": f"Exceptionally low volatility detected <span style='color: rgb(168, 217, 115);'>{vol_percent:.1f}%</span>, potential benefit of ",
            "desc_highlight": f"<span style='color: rgb(3, 177, 82);'>{potential_increase:.1f}%</span>."
        }
    else:
        return {
            "value": 0.0,
            "desc_base": f"Market conditions appear stable",
            "desc_highlight": f'<span style="color: rgb(3, 177, 82);">{vol_percent:.1f}%</span>.',
        }

def compute_size_factor(df):
    """
    Compute Size Factor using the market_value column.
    Compare the latest market value to the historical average.
    - If the latest market value is more than 10% above the average, return a positive effect.
    - If it is more than 10% below the average, return a negative effect.
    - Otherwise, indicate that the asset size change is moderate.
    
    Returns a dictionary with keys: value, desc_base, desc_highlight.
    """
    if 'market_value' not in df.columns or len(df) == 0:
        return {"value": 0.0, "desc_base": "No market value data available.", "desc_highlight": ""}
    
    latest_val = df['market_value'].iloc[-1]
    avg_val = df['market_value'].mean()
    diff_ratio = (latest_val - avg_val) / avg_val  # e.g., 0.15 means a 15% increase
    diff_percent = diff_ratio * 100  # convert to percentage
    
    # Set threshold at 10%
    if diff_ratio > 0.10:
        return {
            "value": diff_percent,
            "desc_base": "Asset size <span style='color:rgb(3, 177, 82);'>increased </span> significantly, leading to a ",
            "desc_highlight": f"<span style='color:rgb(3, 177, 82);'>{diff_percent:.1f}%</span> price boost expected."
        }
    elif diff_ratio < -0.10:
        return {
            "value": diff_percent,
            "desc_base": "Asset size <span style='color:rgb(255, 19, 19);'> decreased </span> significantly, leading to a ",
            "desc_highlight": f"<span style='color:rgb(255, 19, 19);'>{abs(diff_percent):.1f}%</span> price decrease expected."
        }
    else:
        return {
            "value": 0.0,
            "desc_base": "Asset size change is ",
            "desc_highlight": "moderate."
        }

def compute_valuation_factor(df):
    """
    For demonstration, return a fixed effect.
    """
    return {"value": +0.5, "desc_base": "Enhances valuation, potentially increasing PIE ratio by ", "desc_highlight": "0.5%."}

def compute_profitability_factor(df):
    """
    For demonstration, return a fixed effect.
    """
    return {"value": +0.7, "desc_base": "Earnings growth may rise by ", "desc_highlight": "0.7% due to the announcement."}

def compute_investment_factor(df):
    """
    For demonstration, return a fixed effect.
    """
    return {"value": +1.0, "desc_base": "Increased investment could boost stock price by ", "desc_highlight": "1%."}

def compute_news_effect_factor(sentiment_score):
    """
    Compute News Effect Factor based on the raw sentiment score.
    This version returns the sentiment score directly (e.g., -0.99),
    preserving its precision.
    """
    effect_value = sentiment_score  # No scaling applied
    if effect_value >= 0:
        return {
            "value": effect_value,
            "desc_base": "Positive sentiment, estimated effect ",
            "desc_highlight": f"+{effect_value:.2f}."
        }
    else:
        return {
            "value": effect_value,
            "desc_base": "Negative sentiment, estimated effect ",
            "desc_highlight": f"{effect_value:.2f}."
        }

def compute_rsi_factor(df):
    col = "technical_indicators_overbought_oversold_RSI"
    if col not in df.columns or df[col].isnull().all():
        return {"value": 0.0, "desc_base": "RSI data unavailable.", "desc_highlight": ""}
    rsi_value = df[col].iloc[-1]
    if rsi_value > 70:
        return {"value": -2.0, "desc_base": f"RSI {rsi_value:.1f} indicates overbought (", "desc_highlight": "-2%)."}
    elif rsi_value < 30:
        return {"value": +2.0, "desc_base": f"RSI {rsi_value:.1f} indicates oversold (", "desc_highlight": "+2%)."}
    else:
        return {"value": 0.0, "desc_base": f"RSI {rsi_value:.1f} is in a normal range.", "desc_highlight": ""}

def compute_mfi_factor(df):
    col = "technical_indicators_overbought_oversold_MFI"
    if col not in df.columns or df[col].isnull().all():
        return {"value": 0.0, "desc_base": "MFI data unavailable.", "desc_highlight": ""}
    mfi_value = df[col].iloc[-1]
    if mfi_value > 80:
        return {"value": -1.0, "desc_base": f"MFI {mfi_value:.1f} suggests overbought (", "desc_highlight": "-1%)."}
    elif mfi_value < 20:
        return {"value": +1.0, "desc_base": f"MFI {mfi_value:.1f} suggests oversold (", "desc_highlight": "+1%)."}
    else:
        return {"value": 0.0, "desc_base": f"MFI {mfi_value:.1f} is normal.", "desc_highlight": ""}

def compute_bias_factor(df):
    col = "technical_indicators_overbought_oversold_BIAS"
    if col not in df.columns or df[col].isnull().all():
        return {"value": 0.0, "desc_base": "BIAS data unavailable.", "desc_highlight": ""}
    bias_value = df[col].iloc[-1]
    if bias_value > 5:
        return {"value": -1.0, "desc_base": f"BIAS {bias_value:.1f} indicates overbought (", "desc_highlight": "-1%)."}
    elif bias_value < -5:
        return {"value": +1.0, "desc_base": f"BIAS {bias_value:.1f} indicates oversold (", "desc_highlight": "+1%)."}
    else:
        return {"value": 0.0, "desc_base": f"BIAS {bias_value:.1f} is near neutral.", "desc_highlight": ""}