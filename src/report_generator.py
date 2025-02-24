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
    template_dir = os.path.dirname(template_path) or '.'
    env = Environment(loader=FileSystemLoader(template_dir))
    template_file = os.path.basename(template_path)
    template = env.get_template(template_file)
    rendered_html = template.render(reports=reports)
    return rendered_html

def save_html_report(html_content, output_filename="finreport.html"):
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report saved to: {output_filename}")
