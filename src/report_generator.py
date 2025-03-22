# src/report_generator.py
import os
from jinja2 import Environment, FileSystemLoader

def generate_report(stock_symbol, predicted_return, risk_score, news_summary):
    risk_label = "High Risk" if risk_score > 0.7 else "Low/Moderate Risk"
    report = f"""
    FinReport for {stock_symbol}:
    -------------------------------
    Predicted Return: {predicted_return:.2f}%
    Risk Assessment: {risk_label} (score={risk_score:.2f})
    News Summary: {news_summary}

    [Disclaimer: This is a research-based forecast and not financial advice.]
    """
    return report

def generate_styled_finreport(
    stock_symbol,
    date_str,
    news_summary,
    return_forecast,
    risk_assessment,
    final_summary
):
    report = f"""
Analyze stock {stock_symbol} {date_str}, the following impacts on the stock's returns are expected based on the news:

────────────────────────────────────────────────────────
 \033[1mFinReport\033[0m
────────────────────────────────────────────────────────

News Factor: 
{news_summary}

────────────────────────────────────────────────────────
Return Forecast:
{return_forecast}

Risk Assessment:
{risk_assessment}

────────────────────────────────────────────────────────
Summary:
{final_summary}

Disclaimer: This is a research-based forecast. Kindly exercise caution and acknowledge personal risks.
────────────────────────────────────────────────────────
"""
    return report

def generate_html_finreport(
    stock_symbol,
    date_str,
    news_summary,
    news_source,
    market_factor,       # dict with keys: value, desc_base, desc_highlight
    size_factor,         # dict with keys: value, desc_base, desc_highlight
    valuation_factor,    # dict with keys: value, desc_base, desc_highlight
    profitability_factor, # dict with keys: value, desc_base, desc_highlight
    investment_factor,   # dict with keys: value, desc_base, desc_highlight
    news_effect_factor,  # dict with keys: value, desc_base, desc_highlight
    event_factor,        # dict with keys: value, desc_base, desc_highlight
    rsi_factor,          # dict with keys: value, desc (or split if desired)
    mfi_factor,          # dict with keys: value, desc
    bias_factor,         # dict with keys: value, desc
    risk_assessment,
    overall_trend,
    news_effect_score,   # numeric value
    risk_metrics=None,   # New optional parameter
    template_path="templates/report_template.html"
):
    template_dir = os.path.dirname(template_path) or '.'
    env = Environment(loader=FileSystemLoader(template_dir))
    template_file = os.path.basename(template_path)
    template = env.get_template(template_file)

    rendered_html = template.render(
        stock_symbol=stock_symbol,
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
        news_effect_score=news_effect_score,
        risk_metrics=risk_metrics  # Pass risk metrics to the template
    )
    return rendered_html

def generate_multi_report_html(reports, template_path="templates/multi_report_template.html"):
    """
    Combines multiple HTML report snippets into a single HTML page using a multi-report template.
    """
    from datetime import datetime
    template_dir = os.path.dirname(template_path) or '.'
    env = Environment(loader=FileSystemLoader(template_dir))
    template_file = os.path.basename(template_path)
    template = env.get_template(template_file)
    
    # Pass the current datetime as 'now'
    now = datetime.now()
    
    rendered_html = template.render(reports=reports, now=now)
    return rendered_html

def save_html_report(html_content, output_filename="finreport.html"):
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report saved to: {output_filename}")

def generate_risk_assessment_text(predicted_return, volatility, max_drawdown):
    """
    Generate a dynamic and varied risk assessment text based on actual volatility and predicted returns.
    
    Args:
        predicted_return: The predicted return from the model (can be positive or negative)
        volatility: The volatility measure (typically between 0.01-0.30, higher = more volatile)
        max_drawdown: The maximum drawdown value (typically negative, lower = worse)
        
    Returns:
        tuple: (risk_assessment, overall_trend, summary_text)
    """
    import random
    
    # Define thresholds
    high_volatility_threshold = 0.08  # 8%
    medium_volatility_threshold = 0.03  # 3%
    high_drawdown_threshold = -0.40  # 40% drawdown
    medium_drawdown_threshold = -0.25  # 25% drawdown
    
    # Determine volatility level
    if volatility > high_volatility_threshold:
        volatility_level = "high"
        risk_level = "substantial"
        max_decline = f"{min(round(volatility * 100 * 1.3), 20)}%"  # Cap at 20%
    elif volatility > medium_volatility_threshold:
        volatility_level = "elevated"
        risk_level = "moderate to high"
        max_decline = f"{round(volatility * 100 * 1.2)}%"
    else:
        volatility_level = "moderate"
        risk_level = "favorable"
        max_decline = f"{max(round(volatility * 100 * 1.5), 2)}%"  # At least 2%
    
    # Determine drawdown impact
    if max_drawdown < high_drawdown_threshold:
        drawdown_level = "severe"
    elif max_drawdown < medium_drawdown_threshold:
        drawdown_level = "significant"
    else:
        drawdown_level = "moderate"
    
    # Calculate weighted risk score (50% volatility, 30% drawdown, 20% predicted return)
    # Normalize all components to roughly 0-10 scale
    volatility_score = min(volatility * 100, 10)  # Cap at 10
    drawdown_score = min(abs(max_drawdown) * 10, 10)  # Cap at 10
    return_score = 5 - min(max(predicted_return * 2, -5), 5)  # Convert to 0-10 (higher = more risk)
    
    weighted_risk_score = (volatility_score * 0.5) + (drawdown_score * 0.3) + (return_score * 0.2)
    
    # Set overall risk level
    if weighted_risk_score > 6.5:
        overall_risk_level = "high"
    elif weighted_risk_score > 4:
        overall_risk_level = "moderate to high"
    else:
        overall_risk_level = "moderate"
    
    # Generate varied risk assessment text
    risk_templates = [
        f"Historical data shows {volatility_level} stock volatility in recent periods, with expected maximum decline around {max_decline}, indicating {risk_level} risk.",
        f"Analysis of market data reveals {volatility_level} price fluctuations and {drawdown_level} historical drawdowns, suggesting {risk_level} risk levels.",
        f"Trading patterns indicate {volatility_level} volatility with potential {max_decline} maximum decline, representing {risk_level} risk.",
        f"Stock exhibits {volatility_level} price swings and {drawdown_level} historical pullbacks, pointing to {risk_level} risk profile.",
        f"Technical indicators show {volatility_level} price movements with potential {max_decline} downside scenarios, reflecting {risk_level} investment risk."
    ]
    
    risk_assessment = random.choice(risk_templates)
    
    # Determine overall trend
    # Sum all factor values to get overall direction
    # We don't have these values here, but we can use the predicted return as a proxy
    if predicted_return >= 0.5:  # Strong positive
        trend_strength = "Strongly Positive"
    elif predicted_return > 0:  # Mild positive
        trend_strength = "Positive"
    elif predicted_return > -0.5:  # Mild negative
        trend_strength = "Negative"
    else:  # Strong negative
        trend_strength = "Strongly Negative"
    
    # For the summary, need to make it consistent with the risk and trend
    if trend_strength.endswith("Positive"):
        # Positive trend case
        if overall_risk_level == "high":
            # Positive trend but high risk
            summary_templates = [
                f"After analyzing the return forecast and risk metrics, we conclude that while the stock shows positive momentum, it carries a {overall_risk_level} level of risk. Returns may grow by {round(max(predicted_return, 0.5), 1)}% to {round(max(predicted_return, 0.5) + 1, 1)}%, but with substantial volatility.",
                f"Our analysis indicates positive growth potential of {round(max(predicted_return, 0.7), 1)}% to {round(max(predicted_return, 0.7) + 1.2, 1)}%, though investors should note the {overall_risk_level} risk profile with potential for significant price fluctuations.",
                f"The stock demonstrates positive return potential estimated at {round(max(predicted_return, 0.6), 1)}% to {round(max(predicted_return, 0.6) + 0.9, 1)}%, but the {overall_risk_level} risk level suggests caution and appropriate position sizing."
            ]
        else:
            # Positive trend with moderate/low risk
            summary_templates = [
                f"After reviewing the financial metrics and risk assessment, we believe the stock has a {overall_risk_level} level of risk, with estimated returns likely to grow by {round(max(predicted_return, 0.8), 1)}% to {round(max(predicted_return, 0.8) + 1.3, 1)}%.",
                f"Our analysis indicates favorable growth prospects with expected returns of {round(max(predicted_return, 1.0), 1)}% to {round(max(predicted_return, 1.0) + 1.5, 1)}%, supported by a {overall_risk_level} risk profile.",
                f"The combination of positive momentum and {overall_risk_level} risk suggests potential returns of {round(max(predicted_return, 0.9), 1)}% to {round(max(predicted_return, 0.9) + 1.4, 1)}% in the near term."
            ]
    else:
        # Negative trend case
        if overall_risk_level == "high":
            # Negative trend and high risk - most concerning
            summary_templates = [
                f"After analyzing the return forecast and risk metrics, we conclude the stock carries a {overall_risk_level} level of risk and is expected to decline by approximately {round(abs(min(predicted_return, -0.5)) + 5, 1)}% to {round(abs(min(predicted_return, -0.5)) + 8, 1)}%.",
                f"Our assessment indicates significant downside risk with an anticipated decline of {round(abs(min(predicted_return, -0.6)) + 6, 1)}% to {round(abs(min(predicted_return, -0.6)) + 9, 1)}%, coupled with a {overall_risk_level} risk profile.",
                f"The combination of negative momentum and {overall_risk_level} risk suggests a potential decline of {round(abs(min(predicted_return, -0.7)) + 7, 1)}% to {round(abs(min(predicted_return, -0.7)) + 10, 1)}% in the near term."
            ]
        else:
            # Negative trend but moderate/low risk
            summary_templates = [
                f"After analyzing the metrics, we conclude the stock has a {overall_risk_level} level of risk but may experience a decline of approximately {round(abs(min(predicted_return, -0.3)) + 2, 1)}% to {round(abs(min(predicted_return, -0.3)) + 4, 1)}% in the near term.",
                f"Our assessment indicates a likely decline of {round(abs(min(predicted_return, -0.4)) + 3, 1)}% to {round(abs(min(predicted_return, -0.4)) + 5, 1)}%, though the {overall_risk_level} risk profile suggests the downside may be limited.",
                f"The stock shows negative momentum with an expected decline of {round(abs(min(predicted_return, -0.2)) + 1.5, 1)}% to {round(abs(min(predicted_return, -0.2)) + 3.5, 1)}%, within the context of a {overall_risk_level} risk environment."
            ]
    
    summary_text = random.choice(summary_templates)
    
    return risk_assessment, trend_strength, summary_text

def calculate_overall_trend(factor_values):
    """
    Calculate overall trend based on weighted factor scores.
    
    Args:
        factor_values: Dictionary of factor values
        
    Returns:
        str: Overall trend assessment
    """
    # Define factor weights
    weights = {
        'market_factor': 0.20,
        'size_factor': 0.15,
        'valuation_factor': 0.10,
        'profitability_factor': 0.20,
        'investment_factor': 0.15,
        'news_effect_factor': 0.15,
        'event_factor': 0.05
    }
    
    # Calculate weighted sum
    weighted_sum = 0
    for factor, weight in weights.items():
        if factor in factor_values and factor_values[factor] is not None:
            weighted_sum += factor_values[factor] * weight
    
    # Determine trend category
    if weighted_sum >= 0.7:
        return "Strongly Positive"
    elif weighted_sum >= 0.2:
        return "Positive"
    elif weighted_sum >= -0.2:
        return "Neutral"
    elif weighted_sum >= -0.7:
        return "Negative"
    else:
        return "Strongly Negative"

def enhanced_generate_html_finreport(
    stock_symbol,
    date_str,
    news_summary,
    news_source,
    market_factor,
    size_factor,
    valuation_factor,
    profitability_factor,
    investment_factor,
    news_effect_factor,
    event_factor,
    rsi_factor,
    mfi_factor,
    bias_factor,
    risk_assessment,
    overall_trend,
    news_effect_score,
    risk_metrics=None,
    overall_trend_text=None,
    summary_text=None,
    template_path="templates/report_template.html"
):
    """Enhanced version of generate_html_finreport with consistent risk assessment and summary."""
    import os
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = os.path.dirname(template_path) or '.'
    env = Environment(loader=FileSystemLoader(template_dir))
    template_file = os.path.basename(template_path)
    template = env.get_template(template_file)
    
    # Gather all factor values to calculate overall trend
    factor_values = {
        'market_factor': market_factor['value'],
        'size_factor': size_factor['value'],
        'valuation_factor': valuation_factor['value'],
        'profitability_factor': profitability_factor['value'],
        'investment_factor': investment_factor['value'],
        'news_effect_factor': news_effect_factor['value'],
        'event_factor': event_factor['value']
    }
    
    # Calculate overall trend if not provided
    if overall_trend_text is None:
        overall_trend_text = calculate_overall_trend(factor_values)
    
    # Default values for risk metrics
    volatility = 0.03  # Default volatility
    max_drawdown = -0.30  # Default drawdown
    predicted_return = 0.5  # Default return
    
    # Extract real values from risk_metrics if available
    if risk_metrics:
        try:
            # Extract values from string representations
            volatility = float(risk_metrics.get("volatility", "0.03").strip())
            max_drawdown = float(risk_metrics.get("max_drawdown", "-0.30").strip())
            
            # Calculate predicted return from factors
            predicted_return = sum(factor_values.values()) / len(factor_values)
            
            # If overall trend text is negative, adjust predicted return
            if "Negative" in overall_trend_text:
                predicted_return = -abs(predicted_return) - 0.5
            
            # Re-generate consistent risk assessment and summary
            risk_assessment, trend_override, summary = generate_risk_assessment_text(
                predicted_return, volatility, max_drawdown)
            
            # Only override trend if not manually specified
            if overall_trend_text is None:
                overall_trend_text = trend_override
                
            # Only override summary if not manually specified
            if summary_text is None:
                summary_text = summary
        except (ValueError, AttributeError, ZeroDivisionError) as e:
            # Fallback if there's any error in the calculations
            if summary_text is None:
                if "Positive" in overall_trend_text:
                    summary_text = "After reviewing the financial metrics and risk assessment, we believe the stock has a moderate level of risk, with estimated returns likely to grow by 1.0% to 2.0%."
                else:
                    summary_text = "After analyzing the return forecast and risk metrics, we conclude the stock carries a high level of risk and is expected to decline by approximately 6.0% to 8.0%."
    elif summary_text is None:
        # Default summary text if risk metrics not available
        if "Positive" in overall_trend_text:
            summary_text = "After reviewing the financial metrics and risk assessment, we believe the stock has a moderate level of risk, with estimated returns likely to grow by 1.0% to 2.0%."
        else:
            summary_text = "After analyzing the return forecast and risk metrics, we conclude the stock carries a high level of risk and is expected to decline by approximately 6.0% to 8.0%."
    
    rendered_html = template.render(
        stock_symbol=stock_symbol,
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
        overall_trend=overall_trend_text,
        news_effect_score=news_effect_score,
        risk_metrics=risk_metrics,
        summary_text=summary_text
    )
    return rendered_html
