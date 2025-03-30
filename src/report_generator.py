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
    high_volatility_threshold = 0.12    # Increase from 0.08
    medium_volatility_threshold = 0.05  # Increase from 0.03
    high_drawdown_threshold = -0.45     # Adjust from -0.40
    medium_drawdown_threshold = -0.30   # Adjust from -0.25
    
    # Determine volatility level
    if volatility > high_volatility_threshold:
        volatility_level = "high"
        risk_level = "substantial"
        max_decline = f"{min(round(volatility * 100 * 1.3), 20)}%"  # Cap at 20%
    elif volatility > medium_volatility_threshold:
        volatility_level = "elevated"
        risk_level = "moderate to high"
        max_decline = f"{round(volatility * 100 * 1.2)}%"
    elif volatility > 0.03:  # Add another tier
        volatility_level = "moderate"
        risk_level = "moderate"
        max_decline = f"{round(volatility * 100 * 1.5)}%"
    else:
        volatility_level = "low"
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
        f"Historical data shows <span class='{'negative' if volatility_level in ['high', 'elevated'] else 'positive'}'>{volatility_level}</span> stock volatility in recent periods, with expected maximum decline around <span class='{'negative' if float(max_decline.strip('%')) > 5 else 'positive'}'>{max_decline}</span>, indicating <span class='{'negative' if 'substantial' in risk_level or 'high' in risk_level else 'positive'}'>{risk_level} risk</span>.",
        f"Analysis of market data reveals <span class='{'negative' if volatility_level in ['high', 'elevated'] else 'positive'}'>{volatility_level}</span> price fluctuations and <span class='{'negative' if drawdown_level in ['severe', 'significant'] else 'positive'}'>{drawdown_level}</span> historical drawdowns, suggesting <span class='{'negative' if 'substantial' in risk_level or 'high' in risk_level else 'positive'}'>{risk_level} risk levels</span>.",
        f"Trading patterns indicate <span class='{'negative' if volatility_level in ['high', 'elevated'] else 'positive'}'>{volatility_level}</span> volatility with potential <span class='{'negative' if float(max_decline.strip('%')) > 5 else 'positive'}'>{max_decline}</span> maximum decline, representing <span class='{'negative' if 'substantial' in risk_level or 'high' in risk_level else 'positive'}'>{risk_level} risk</span>.",
        f"Stock exhibits <span class='{'negative' if volatility_level in ['high', 'elevated'] else 'positive'}'>{volatility_level}</span> price swings and <span class='{'negative' if drawdown_level in ['severe', 'significant'] else 'positive'}'>{drawdown_level}</span> historical pullbacks, pointing to <span class='{'negative' if 'substantial' in risk_level or 'high' in risk_level else 'positive'}'>{risk_level} risk profile</span>.",
        f"Technical indicators show <span class='{'negative' if volatility_level in ['high', 'elevated'] else 'positive'}'>{volatility_level}</span> price movements with potential <span class='{'negative' if float(max_decline.strip('%')) > 5 else 'positive'}'>{max_decline}</span> downside scenarios, reflecting <span class='{'negative' if 'substantial' in risk_level or 'high' in risk_level else 'positive'}'>{risk_level} investment risk</span>."
    ]
    
    risk_assessment = random.choice(risk_templates)
    
    # Determine overall trend
    if predicted_return >= 0.5:  # Strong positive
        trend_strength = "Strongly Positive"
    elif predicted_return > 0:  # Mild positive
        trend_strength = "Positive"
    elif predicted_return > -0.5:  # Mild negative
        trend_strength = "Negative"
    else:  # Strong negative
        trend_strength = "Strongly Negative"
    
    # Generate summary text based on trend and overall risk level
    if trend_strength.endswith("Positive"):
        if overall_risk_level == "high":
            # Positive trend with high risk
            summary_templates = [
                f"After analyzing the metrics, we conclude that while the stock shows <span class='positive'>positive momentum</span>, it carries a <span class='negative'>{overall_risk_level} level of risk</span>. Returns may grow by <span class='positive'>{round(max(predicted_return, 0.5), 1)}% to {round(max(predicted_return, 0.5) + 1, 1)}%</span>, but with substantial volatility.",
                f"Our analysis indicates <span class='positive'>positive growth potential of {round(max(predicted_return, 0.7), 1)}% to {round(max(predicted_return, 0.7) + 1.2, 1)}%</span>, though investors should note the <span class='negative'>{overall_risk_level} risk profile</span> requires caution.",
                f"While showing <span class='positive'>positive trends with projected returns of {round(max(predicted_return, 0.6), 1)}% to {round(max(predicted_return, 0.6) + 0.9, 1)}%</span>, the <span class='negative'>{overall_risk_level} risk level</span> suggests using appropriate position sizing."
            ]
        else:
            # Positive trend with moderate/low risk
            summary_templates = [
                f"After reviewing the financial metrics and risk assessment, we believe the stock has a <span class='positive'>{overall_risk_level} level of risk</span>, with estimated returns likely to <span class='positive'>grow by {round(max(predicted_return, 0.8), 1)}% to {round(max(predicted_return, 0.8) + 1.3, 1)}%</span>.",
                f"Our analysis indicates <span class='positive'>favorable growth prospects with expected returns of {round(max(predicted_return, 1.0), 1)}% to {round(max(predicted_return, 1.0) + 1.5, 1)}%</span>, supported by a <span class='positive'>{overall_risk_level} risk profile</span>.",
                f"The combination of <span class='positive'>positive momentum</span> and <span class='positive'>{overall_risk_level} risk</span> suggests <span class='positive'>potential returns of {round(max(predicted_return, 0.9), 1)}% to {round(max(predicted_return, 0.9) + 1.4, 1)}%</span> in the near term.",
                f"Technical indicators and financial metrics point to a <span class='positive'>promising outlook with projected growth of {round(max(predicted_return, 0.7), 1)}% to {round(max(predicted_return, 0.7) + 1.2, 1)}%</span> alongside <span class='positive'>{overall_risk_level} risk levels</span>.",
                f"With a <span class='positive'>{overall_risk_level} risk profile</span>, we expect the stock to deliver <span class='positive'>returns of {round(max(predicted_return, 0.6), 1)}% to {round(max(predicted_return, 0.6) + 1.1, 1)}%</span> based on current market conditions."
            ]
    else:
        if overall_risk_level == "high":
            # Negative trend and high risk
            summary_templates = [
                f"After analyzing the return forecast and risk metrics, we conclude the stock carries a <span class='negative'>{overall_risk_level} level of risk</span> and is <span class='negative'>expected to decline by approximately {round(abs(min(predicted_return, -0.5)) + 5, 1)}% to {round(abs(min(predicted_return, -0.5)) + 8, 1)}%</span>.",
                f"Our assessment indicates <span class='negative'>significant downside risk with an anticipated decline of {round(abs(min(predicted_return, -0.6)) + 6, 1)}% to {round(abs(min(predicted_return, -0.6)) + 9, 1)}%</span>, coupled with a <span class='negative'>{overall_risk_level} risk profile</span>.",
                f"The combination of <span class='negative'>negative momentum</span> and <span class='negative'>{overall_risk_level} risk</span> suggests a <span class='negative'>potential decline of {round(abs(min(predicted_return, -0.7)) + 7, 1)}% to {round(abs(min(predicted_return, -0.7)) + 10, 1)}%</span> in the near term.",
                f"Market indicators point to <span class='negative'>substantial headwinds with a projected decrease of {round(abs(min(predicted_return, -0.5)) + 4, 1)}% to {round(abs(min(predicted_return, -0.5)) + 7, 1)}%</span>, amid a backdrop of <span class='negative'>{overall_risk_level} volatility</span>.",
                f"Technical analysis reveals <span class='negative'>concerning trends with an expected decline of {round(abs(min(predicted_return, -0.4)) + 5.5, 1)}% to {round(abs(min(predicted_return, -0.4)) + 8.5, 1)}%</span>, compounded by <span class='negative'>{overall_risk_level} risk factors</span>."
            ]
        else:
            # Negative trend but moderate/low risk
            summary_templates = [
                f"After analyzing the metrics, we conclude the stock has a <span class='positive'>{overall_risk_level} level of risk</span> but may experience a <span class='negative'>decline of approximately {round(abs(min(predicted_return, -0.3)) + 2, 1)}% to {round(abs(min(predicted_return, -0.3)) + 4, 1)}%</span> in the near term.",
                f"Our assessment indicates a likely <span class='negative'>decline of {round(abs(min(predicted_return, -0.4)) + 3, 1)}% to {round(abs(min(predicted_return, -0.4)) + 5, 1)}%</span>, though the <span class='positive'>{overall_risk_level} risk profile</span> suggests the downside may be limited.",
                f"The stock shows <span class='negative'>negative momentum with an expected decline of {round(abs(min(predicted_return, -0.2)) + 1.5, 1)}% to {round(abs(min(predicted_return, -0.2)) + 3.5, 1)}%</span>, within the context of a <span class='positive'>{overall_risk_level} risk environment</span>.",
                f"Despite maintaining a <span class='positive'>{overall_risk_level} risk assessment</span>, current trends point to a <span class='negative'>potential decrease of {round(abs(min(predicted_return, -0.3)) + 2.5, 1)}% to {round(abs(min(predicted_return, -0.3)) + 4.5, 1)}%</span> in the coming period.",
                f"Financial indicators suggest a temporary setback with a <span class='negative'>projected decline of {round(abs(min(predicted_return, -0.25)) + 1.8, 1)}% to {round(abs(min(predicted_return, -0.25)) + 3.8, 1)}%</span>, though the <span class='positive'>{overall_risk_level} risk level</span> may limit further downside."
            ]
    
    summary_text = random.choice(summary_templates)
    
    return risk_assessment, trend_strength, summary_text

def calculate_overall_trend(factor_values):
    """
    Calculate overall trend based on weighted factor scores with improved balance.
    """
    weights = {
        'market_factor': 0.15,        # Reduced from 0.20
        'size_factor': 0.15,
        'valuation_factor': 0.10,
        'profitability_factor': 0.15, # Reduced from 0.20
        'investment_factor': 0.20,    # Increased from 0.15
        'news_effect_factor': 0.10,   # Reduced from 0.15
        'event_factor': 0.15          # Increased from 0.05
    }
    
    weighted_sum = 0
    sum_of_weights = 0
    
    for factor, weight in weights.items():
        if factor in factor_values and factor_values[factor] is not None:
            weighted_sum += factor_values[factor] * weight
            sum_of_weights += weight
    
    if sum_of_weights > 0 and sum_of_weights < 1.0:
        weighted_sum = weighted_sum / sum_of_weights
    
    weighted_sum += 0.15
    
    if weighted_sum >= 0.6:
        return "Strongly Positive"
    elif weighted_sum >= 0.15:
        return "Positive"
    elif weighted_sum >= -0.15:
        return "Neutral"
    elif weighted_sum >= -0.6:
        return "Negative"
    else:
        return "Strongly Negative"

# New formatting functions

def format_factor_for_display(factor):
    """
    Formats factor output into a clean format for display in the report template.
    Ensures all percentage values are properly rounded and handles zero values appropriately.
    """
    value = factor.get('value', 0.0)
    
    # Round the value to 1 decimal place to avoid floating point issues
    value_rounded = round(float(value), 1)
    
    if 'desc_base' in factor and 'desc_highlight' in factor:
        desc_parts = factor['desc_base'].split(',')
        if len(desc_parts) > 0:
            desc_text = desc_parts[0].strip()
        else:
            desc_text = factor['desc_base']
        
        # Even if value is zero, still show a directional impact rather than "neutral impact"
        if abs(value_rounded) < 0.1:
            # For very small values (including zero), show as a minimal impact with direction
            if value_rounded >= 0:
                impact_text = f"causing a 0.1% stock price increase."
            else:
                impact_text = f"causing a 0.1% stock decline."
        else:
            if value_rounded >= 0:
                impact_text = f"causing a {abs(value_rounded):.1f}% stock price increase."
            else:
                impact_text = f"causing a {abs(value_rounded):.1f}% stock decline."
    else:
        desc_text = "Factor analysis indicates"
        
        # Same handling for fallback case
        if abs(value_rounded) < 0.1:
            if value_rounded >= 0:
                impact_text = f"a minor 0.1% positive impact."
            else:
                impact_text = f"a minor 0.1% negative impact."
        else:
            impact_text = f"a {abs(value_rounded):.1f}% {'positive' if value_rounded >= 0 else 'negative'} impact."
    
    return {
        'value': value_rounded,
        'desc_text': desc_text,
        'impact_text': impact_text
    }

def format_risk_assessment(risk_metrics, predicted_return):
    """
    Format risk assessment into components needed for template.
    """
    volatility = float(risk_metrics.get("volatility", "0.03").strip())
    
    if volatility > 0.12:
        volatility_text = "heightened stock volatility in 30 days"
        risk_level = "substantial risk"
        max_decline = min(round(volatility * 100 * 1.3), 20)
    elif volatility > 0.08:
        volatility_text = "elevated stock volatility in 30 days"
        risk_level = "moderate to high risk"
        max_decline = round(volatility * 100 * 1.2)
    elif volatility > 0.05:
        volatility_text = "moderate stock fluctuations in 30 days"
        risk_level = "moderate risk"
        max_decline = round(volatility * 100 * 1.5)
    else:
        volatility_text = "low stock fluctuations in 30 days"
        risk_level = "favorable risk"
        max_decline = max(round(volatility * 100 * 1.5), 2)
    
    if volatility > 0.08:
        volatility_text += ", indicating potential for abnormal fluctuations"
    else:
        volatility_text += ", no abnormal swings"
    
    risk_assessment_text = f"Historical data reveals {volatility_text}."
    
    return {
        'risk_assessment_text': risk_assessment_text,
        'max_decline': max_decline,
        'risk_level': risk_level
    }

def format_summary_text(overall_trend, predicted_return, risk_level):
    """
    Generate a consistent summary text based on trend and risk.
    Ensures all percentage values are properly rounded to 1 decimal place.
    
    Args:
        overall_trend: String indicating overall trend assessment
        predicted_return: Float with predicted return value
        risk_level: String with risk level assessment
        
    Returns:
        Summary text appropriate for the trend and risk level
    """
    predicted_return_rounded = round(float(predicted_return), 1)
    
    is_positive = 'Positive' in overall_trend
    is_high_risk = 'substantial' in risk_level or 'high' in risk_level
    
    if is_positive:
        if is_high_risk:
            return f"the stock has a relatively high level of risk, and the estimated rate of return is expected to grow by {abs(predicted_return_rounded):.1f}% to {abs(predicted_return_rounded) + 1.0:.1f}%."
        else:
            return f"the stock has a relatively low level of risk, and the estimated rate of return is expected to grow by {abs(predicted_return_rounded):.1f}% to {abs(predicted_return_rounded) + 1.0:.1f}%."
    else:
        if is_high_risk:
            return f"the stock carries an extremely high level of risk, and the anticipated rate of return decline is expected to be above {abs(predicted_return_rounded) + 6.0:.1f}%."
        else:
            return f"the stock carries an elevated risk profile, and the anticipated rate of return decline is expected to be {abs(predicted_return_rounded) + 3.0:.1f}% to {abs(predicted_return_rounded) + 5.0:.1f}%."

# Replaced enhanced_generate_html_finreport function

def enhanced_generate_html_finreport(
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
    risk_metrics=None,   # Optional parameter
    overall_trend_text=None,
    summary_text=None,
    template_path="templates/report_template.html"
):
    """Enhanced version of generate_html_finreport with cleaner, formatted display."""
    import os
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = os.path.dirname(template_path) or '.'
    env = Environment(loader=FileSystemLoader(template_dir))
    template_file = os.path.basename(template_path)
    template = env.get_template(template_file)
    
    factor_values = {
        'market_factor': market_factor['value'],
        'size_factor': size_factor['value'],
        'valuation_factor': valuation_factor['value'],
        'profitability_factor': profitability_factor['value'],
        'investment_factor': investment_factor['value'],
        'news_effect_factor': news_effect_factor['value'],
        'event_factor': event_factor['value']
    }
    
    if overall_trend_text is None:
        overall_trend_text = calculate_overall_trend(factor_values)
    
    volatility = 0.03
    max_drawdown = -0.30
    
    predicted_return = (
        factor_values.get('market_factor', 0) * 0.10 +
        factor_values.get('size_factor', 0) * 0.15 +
        factor_values.get('valuation_factor', 0) * 0.10 +
        factor_values.get('profitability_factor', 0) * 0.10 +
        factor_values.get('investment_factor', 0) * 0.20 +
        factor_values.get('news_effect_factor', 0) * 0.10 +
        factor_values.get('event_factor', 0) * 0.25
    )
    predicted_return = round(predicted_return + 0.6, 1)
    
    formatted_market_factor = format_factor_for_display(market_factor)
    formatted_size_factor = format_factor_for_display(size_factor)
    formatted_valuation_factor = format_factor_for_display(valuation_factor)
    formatted_profitability_factor = format_factor_for_display(profitability_factor)
    formatted_investment_factor = format_factor_for_display(investment_factor)
    formatted_news_effect_factor = format_factor_for_display(news_effect_factor)
    
    risk_info = {"risk_assessment_text": "Historical data shows moderate volatility.", 
                 "max_decline": 5, 
                 "risk_level": "moderate risk"}
    
    if risk_metrics:
        try:
            volatility = float(risk_metrics.get("volatility", "0.03").strip())
            max_drawdown = float(risk_metrics.get("max_drawdown", "-0.30").strip())
            
            risk_info = format_risk_assessment(risk_metrics, predicted_return)
            
            if "Negative" in overall_trend_text and predicted_return > 0:
                predicted_return = round(-abs(predicted_return) * 0.5, 1)
            elif "Positive" in overall_trend_text and predicted_return < 0:
                predicted_return = round(abs(predicted_return) * 0.5, 1)
        except (ValueError, AttributeError, ZeroDivisionError):
            pass
    
    if summary_text is None:
        summary_text = format_summary_text(overall_trend_text, predicted_return, risk_info['risk_level'])
    
    rendered_html = template.render(
        stock_symbol=stock_symbol,
        date_str=date_str,
        news_summary=news_summary,
        news_source=news_source,
        market_factor=formatted_market_factor,
        size_factor=formatted_size_factor,
        valuation_factor=formatted_valuation_factor,
        profitability_factor=formatted_profitability_factor,
        investment_factor=formatted_investment_factor,
        news_effect_factor=formatted_news_effect_factor,
        risk_assessment_text=risk_info['risk_assessment_text'],
        max_decline=risk_info['max_decline'],
        risk_level=risk_info['risk_level'],
        overall_trend=overall_trend_text,
        summary_text=summary_text,
        risk_metrics=risk_metrics,
        risk_assessment=risk_assessment
    )
    return rendered_html
