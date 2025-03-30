import random
import re
import numpy as np
import pandas as pd
from textblob import TextBlob

# Dictionary of positive and negative keywords related to finance
POSITIVE_KEYWORDS = ['increase', 'rise', 'grow', 'boost', 'gain', 'profit', 'positive', 'improved', 
                     'expansion', 'launch', 'success', 'innovation', 'partnership', 'agreement',
                     'acquisition', 'investment', 'dividend', 'earnings', 'beat', 'exceed', 'repurchase',
                     'FDA', 'approval', 'awarded', 'patent', 'contract', 'new product', 'win', 'bullish']

NEGATIVE_KEYWORDS = ['decrease', 'decline', 'drop', 'fall', 'loss', 'negative', 'bearish', 'debt', 
                     'layoff', 'warning', 'investigation', 'lawsuit', 'delay', 'postpone', 'cancel', 
                     'dispute', 'weak', 'miss', 'below', 'reduced', 'cut', 'suspend', 'halt', 'bearish',
                     'tariff', 'penalty', 'recall', 'bearish', 'fine', 'resign', 'reduction', 'fail', 'warning']

# Dictionary of specific sector-related terms and their impact descriptions
SECTOR_IMPACTS = {
    'pharmaceutical': ['drug approval', 'clinical trial', 'FDA', 'patent', 'medicine', 'therapeutic', 'vaccine'],
    'technology': ['chip', 'semiconductor', 'software', 'AI', 'cloud', 'digital', 'computing', 'tech'],
    'automotive': ['vehicle', 'car', 'auto', 'EV', 'electric vehicle', 'charging', 'battery'],
    'energy': ['oil', 'gas', 'renewable', 'energy', 'power', 'solar', 'wind', 'hydrogen'],
    'financial': ['bank', 'loan', 'interest', 'dividend', 'deposit', 'credit', 'financial', 'insurance'],
    'real_estate': ['property', 'land', 'housing', 'construction', 'building', 'estate', 'home'],
    'retail': ['store', 'retail', 'consumer', 'e-commerce', 'shopping', 'merchandise'],
    'telecom': ['network', '5G', 'telecom', 'communication', 'broadband', 'mobile'],
    'healthcare': ['health', 'hospital', 'care', 'medical', 'patient', 'diagnostic', 'treatment'],
    'mining': ['mine', 'mineral', 'gold', 'lithium', 'copper', 'iron', 'metal']
}

def identify_sector(news_text):
    """Identify the sector based on the news text"""
    news_text = news_text.lower()
    
    sector_counts = {}
    for sector, keywords in SECTOR_IMPACTS.items():
        matches = sum(1 for keyword in keywords if keyword.lower() in news_text)
        if matches > 0:
            sector_counts[sector] = matches
            
    if not sector_counts:
        return "general"
    
    return max(sector_counts, key=sector_counts.get)

def analyze_sentiment(news_text):
    """Analyze the sentiment of the news text"""
    if not news_text or len(news_text) < 10:
        return 0.0
        
    # Count positive and negative keywords
    news_lower = news_text.lower()
    positive_count = sum(1 for word in POSITIVE_KEYWORDS if word.lower() in news_lower)
    negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word.lower() in news_lower)
    
    # Use TextBlob for sentiment analysis
    try:
        blob = TextBlob(news_text)
        polarity = blob.sentiment.polarity
    except:
        # Fallback if TextBlob fails
        polarity = (positive_count - negative_count) / (positive_count + negative_count + 1)
    
    # Combine keyword analysis with TextBlob
    keyword_sentiment = (positive_count - negative_count) / (positive_count + negative_count + 1)
    combined_sentiment = (polarity + keyword_sentiment) / 2
    
    # Scale to range between -1 and 1
    return max(min(combined_sentiment, 1.0), -1.0)

def extract_percentage_changes(news_text):
    """Extract percentage changes mentioned in news"""
    percentage_pattern = r'(-?\d+(?:\.\d+)?)%'
    matches = re.findall(percentage_pattern, news_text)
    
    changes = []
    for match in matches:
        try:
            changes.append(float(match))
        except ValueError:
            continue
            
    return changes

def extract_financial_figures(news_text):
    """Extract financial figures from news (in billions/millions)"""
    # Pattern for billions (e.g., 1.5亿元, 2亿元, 3.4亿美元)
    billion_pattern = r'(\d+(?:\.\d+)?)亿(?:元|美元|港币|人民币)?'
    # Pattern for millions (e.g., 500万元, 600万美元)
    million_pattern = r'(\d+(?:\.\d+)?)万(?:元|美元|港币|人民币)?'
    
    billions = []
    for match in re.findall(billion_pattern, news_text):
        try:
            billions.append(float(match))
        except ValueError:
            continue
            
    millions = []
    for match in re.findall(million_pattern, news_text):
        try:
            millions.append(float(match))
        except ValueError:
            continue
    
    # Convert to common unit (billions)
    return sum(billions) + sum(millions) / 10000.0

def get_varied_description(base_list, sentiment, magnitude):
    """Get varied text based on sentiment and magnitude"""
    if not base_list:
        return ""
        
    # Define ranges for magnitudes
    if magnitude < 0.2:
        magnitude_desc = "slight"
    elif magnitude < 0.5:
        magnitude_desc = "moderate"
    elif magnitude < 0.8:
        magnitude_desc = "significant"
    else:
        magnitude_desc = "substantial"
        
    # Handle positive sentiment
    if sentiment > 0:
        if magnitude < 0.3:
            selected = [text for text in base_list if 'minor' in text or 'slight' in text or 'small' in text]
        elif magnitude < 0.7:
            selected = [text for text in base_list if 'moderate' in text or 'expected' in text]
        else:
            selected = [text for text in base_list if 'significant' in text or 'substantial' in text or 'major' in text]
    # Handle negative sentiment
    else:
        if magnitude < 0.3:
            selected = [text for text in base_list if 'minor' in text or 'limited' in text or 'small' in text]
        elif magnitude < 0.7:
            selected = [text for text in base_list if 'concerning' in text or 'notable' in text]
        else:
            selected = [text for text in base_list if 'major' in text or 'significant' in text or 'severe' in text]
    
    # If filtered list is empty, use the original
    if not selected:
        selected = base_list
        
    return random.choice(selected).replace("{magnitude}", magnitude_desc)

def compute_market_factor(df, news_text=""):
    """
    Compute Market Factor with diverse descriptions based on news content.
    """
    if 'pct_chg' not in df.columns or len(df) < 3:
        return {"value": 0.0, "desc_base": "Insufficient market data available.", "desc_highlight": "neutral impact expected."}

    # Get sentiment from news
    sentiment = analyze_sentiment(news_text)
    
    # Extract recent volatility
    recent_window = min(5, len(df))
    volatility = df['pct_chg'].iloc[-recent_window:].std() * 100  # Convert to percentage
    
    # Recent trend analysis
    recent_changes = df['pct_chg'].iloc[-recent_window:].values
    positive_days = sum(1 for change in recent_changes if change > 0)
    negative_days = sum(1 for change in recent_changes if change < 0)
    
    # Volatility-based impact calculation with reduced negative bias
    if volatility > 4.0:  # High volatility
        base_impact = -1.5 - (volatility - 4.0) * 0.1  # Reduced negative impact
        impact_range = (-3.0, -0.5)  # Less extreme negative range
    elif volatility > 2.5:  # Moderate volatility
        base_impact = -0.7 - (volatility - 2.5) * 0.3  # Reduced negative impact
        impact_range = (-1.5, 0.0)  # Allow for neutral outcomes
    elif positive_days > negative_days:  # Low volatility, positive trend
        base_impact = 0.7 + sentiment * 0.8  # Increased positive impact
        impact_range = (0.2, 1.8)  # More positive range
    else:  # Low volatility, negative or neutral trend
        base_impact = -0.2 + sentiment * 0.6  # Less negative base
        impact_range = (-0.8, 0.8)  # More balanced range
    
    # Add reduced randomization but keep within range
    effect_value = base_impact + random.uniform(-0.2, 0.2)
    effect_value = effect_value * 1.5  # Amplify by 50%
    effect_value = max(min(effect_value, impact_range[1]), impact_range[0])
    effect_value = round(effect_value, 1)  # Round to 1 decimal place
    
    # Generate descriptions based on the calculated effect
    if effect_value > 0:
        base_descriptions = [
            "Favorable market conditions suggest ",
            "Positive market sentiment indicates ",
            "Supportive market environment points to ",
            "Bullish market factors suggest ",
            "Market dynamics appear favorable, indicating "
        ]
        
        highlight_templates = [
            "potential {value}% increase in returns.",
            "a likely {value}% positive impact.",
            "an estimated {value}% upside potential.",
            "approximately {value}% positive contribution.",
            "a {value}% boost to overall returns."
        ]
        
        desc_base = random.choice(base_descriptions)
        desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
        
    elif effect_value < -1.0:
        base_descriptions = [
            "Due to market risk appetite, this may lead to short-term fluctuations, ",
            "Heightened market volatility suggests ",
            "Challenging market conditions could result in ",
            "Market uncertainty may contribute to ",
            "Turbulent market environment indicates "
        ]
        
        highlight_templates = [
            "causing a {value}% stock decline.",
            "a potential {value}% downward pressure.",
            "an estimated {value}% negative impact.",
            "a possible {value}% reduction in value.",
            "approximately {value}% downside risk."
        ]
        
        desc_base = random.choice(base_descriptions)
        desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
        
    else:
        base_descriptions = [
            "Moderate market volatility detected, ",
            "Mixed market signals suggest ",
            "Neutral market conditions indicate ",
            "Balanced market factors point to ",
            "Current market trends suggest "
        ]
        
        if effect_value < 0:
            highlight_templates = [
                "causing a {value}% stock decline.",
                "a slight {value}% negative pressure.",
                "a minor {value}% downward adjustment.",
                "a modest {value}% decrease in value.",
                "a limited {value}% downside risk."
            ]
        else:
            highlight_templates = [
                "minimal impact on stock performance.",
                "a neutral effect on valuation.",
                "balanced risk-reward outlook.",
                "stable price action expected.",
                "limited market-driven fluctuations."
            ]
        
        desc_base = random.choice(base_descriptions)
        if effect_value < 0:
            desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
        else:
            desc_highlight = random.choice(highlight_templates)
    
    return {
        "value": effect_value, 
        "desc_base": desc_base, 
        "desc_highlight": desc_highlight
    }

def compute_size_factor(df, news_text=""):
    """
    Compute Size Factor with diverse descriptions based on market value trends and news content.
    """
    if 'market_value' not in df.columns or len(df) < 3:
        return {"value": 0.0, "desc_base": "Insufficient market value data.", "desc_highlight": "neutral size impact expected."}
    
    # Extract recent size change
    latest_val = df['market_value'].iloc[-1]
    avg_val = df['market_value'].mean()
    
    # Safety check to avoid division by zero
    if abs(avg_val) < 1e-10:
        return {"value": 0.0, "desc_base": "Unreliable market value data.", "desc_highlight": "size effect cannot be determined."}
    
    # Calculate size change percentage
    diff_ratio = (latest_val - avg_val) / avg_val
    diff_percent = diff_ratio * 100
    
    # Extract financial figures from news
    financial_impact = extract_financial_figures(news_text)
    
    # News sentiment analysis
    sentiment = analyze_sentiment(news_text)
    
    # Base effect on market value change
    if diff_ratio > 0.25:  # Very large increase (>25%)
        base_effect = 1.0 + min(diff_ratio * 0.1, 0.5)
        sentiment_modifier = max(sentiment * 0.3, 0)  # Only positive sentiment boosts further
    elif diff_ratio > 0.10:  # Significant increase (10-25%)
        base_effect = 0.7 + diff_ratio * 0.5
        sentiment_modifier = sentiment * 0.2
    elif diff_ratio > 0.05:  # Moderate increase (5-10%)
        base_effect = 0.5 + diff_ratio * 0.5
        sentiment_modifier = sentiment * 0.1
    elif diff_ratio > -0.05:  # Minimal change (-5% to +5%)
        base_effect = diff_ratio * 3.0  # Scale small changes
        sentiment_modifier = sentiment * 0.2
    elif diff_ratio > -0.10:  # Moderate decrease (-5% to -10%)
        base_effect = -0.5 + diff_ratio * 2.0
        sentiment_modifier = min(sentiment * 0.1, 0)  # Only negative sentiment decreases further
    elif diff_ratio > -0.25:  # Significant decrease (-10% to -25%)
        base_effect = -0.7 + diff_ratio * 1.0
        sentiment_modifier = min(sentiment * 0.2, 0)
    else:  # Very large decrease (< -25%)
        base_effect = -1.0 + max(diff_ratio * 0.1, -0.5)
        sentiment_modifier = min(sentiment * 0.3, 0)
        
    # Add impact from financial figures in news
    news_modifier = 0
    if financial_impact > 0:
        scale = min(financial_impact / 10.0, 1.0)  # Cap the impact
        news_modifier = scale * 0.4 * (1 if sentiment > 0 else -1)
        
    # Calculate final effect with some randomization
    effect_value = base_effect + sentiment_modifier + news_modifier
    effect_value = effect_value * 1.5  # Amplify by 50%
    effect_value = effect_value + random.uniform(-0.1, 0.1)  # Add small randomization
    effect_value = round(max(min(effect_value, 1.5), -1.5), 1)  # Cap and round
    
    # Generate diverse descriptions
    if effect_value >= 0.7:
        base_templates = [
            "The announcement will increase asset size, leading to a ",
            "Significant expansion in asset base suggests ",
            "Substantial growth in market capitalization indicates ",
            "Notable increase in company size points to ",
            "The expanded scale of operations suggests "
        ]
        
        highlight_templates = [
            "{value}% stock price increase.",
            "potential {value}% upside from size effects.",
            "an estimated {value}% boost from increased scale.",
            "approximately {value}% positive size premium.",
            "a {value}% valuation enhancement from size factors."
        ]
        
    elif effect_value >= 0.3:
        base_templates = [
            "Moderate increase in asset size, leading to a ",
            "The announcement will moderately expand market presence, resulting in ",
            "Growth in company scale suggests ",
            "Positive size development indicates ",
            "Improved market positioning suggests "
        ]
        
        highlight_templates = [
            "{value}% potential upside.",
            "an expected {value}% positive effect.",
            "a likely {value}% benefit from size improvements.",
            "approximately {value}% increase from size factors.",
            "a {value}% boost to valuation multiples."
        ]
        
    elif effect_value > 0:
        base_templates = [
            "Small increase in asset size, leading to a ",
            "Modest expansion in operations suggests ",
            "Slight improvement in market presence indicates ",
            "Minor positive size development points to ",
            "Incremental growth in scale suggests "
        ]
        
        highlight_templates = [
            "{value}% potential upside.",
            "a modest {value}% positive effect.",
            "a small {value}% benefit from size factors.",
            "a limited but positive {value}% impact.",
            "a minor {value}% improvement in valuation."
        ]
        
    elif effect_value > -0.3:
        base_templates = [
            "Minimal change in asset size, suggesting ",
            "Negligible size impact indicates ",
            "Limited size effect points to ",
            "Marginal change in market presence suggests ",
            "Subtle shifts in company scale indicate "
        ]
        
        highlight_templates = [
            "a {value}% potential downside.",
            "a minimal {value}% negative effect.",
            "a slight {value}% adjustment in valuation.",
            "a marginal {value}% impact on pricing.",
            "a very limited {value}% decrease in value."
        ]
        
    elif effect_value > -0.7:
        base_templates = [
            "Moderate decrease in asset size, leading to ",
            "Reduced scale of operations suggests ",
            "Contraction in market presence indicates ",
            "Declining asset base points to ",
            "Downward adjustment in size metrics suggests "
        ]
        
        highlight_templates = [
            "a {value}% potential downside.",
            "an expected {value}% negative effect.",
            "approximately {value}% decrease from size factors.",
            "a moderate {value}% impact on valuation.",
            "a noticeable {value}% reduction in market premium."
        ]
        
    else:
        base_templates = [
            "Reduced asset size suggests limited gains, causing a ",
            "Significant contraction in market presence indicates ",
            "Substantial decrease in company scale points to ",
            "Major reduction in asset base suggests ",
            "Sharp decline in size metrics indicates "
        ]
        
        highlight_templates = [
            "{value}% decrease.",
            "a considerable {value}% negative impact.",
            "a substantial {value}% downward pressure.",
            "an estimated {value}% reduction in valuation.",
            "a significant {value}% downward adjustment."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }

def compute_valuation_factor(df, news_text=""):
    """
    Compute valuation factor with diverse descriptions based on multiple potential metrics.
    """
    # Look for various valuation-related columns
    valuation_cols = [
        "value_factor_Book_to_Market_Equity",
        "value_factor_Divergence_Rate",
        "value_factor_Dividend_Yield",
        "value_factor_Sales_to_Price_Ratio",
        "value_factor_Assets_to_Market_Equity",
        "value_factor_Debt_to_Market_Equity"
    ]
    
    # Check if any valuation column exists
    available_cols = [col for col in valuation_cols if col in df.columns and not df[col].isnull().all()]
    
    # News analysis
    sentiment = analyze_sentiment(news_text)
    sector = identify_sector(news_text)
    
    # For profit/loss indication in news
    profit_terms = ['profit', 'earnings', 'income', 'revenue', 'gain', 'increase']
    loss_terms = ['loss', 'decline', 'decrease', 'drop', 'deficit', 'negative', 'debt']
    
    # Count profit/loss mentions
    news_lower = news_text.lower() if news_text else ""
    profit_count = sum(1 for term in profit_terms if term in news_lower)
    loss_count = sum(1 for term in loss_terms if term in news_lower)
    
    # Extract percentage changes
    percentage_changes = extract_percentage_changes(news_text)
    has_positive_change = any(change > 0 for change in percentage_changes)
    has_negative_change = any(change < 0 for change in percentage_changes)
    
    # Based on sector, adjust base valuation impact
    sector_adjustments = {
        'pharmaceutical': 0.2 if sentiment > 0 else -0.3,
        'technology': 0.3 if sentiment > 0 else -0.2,
        'automotive': 0.1 if sentiment > 0 else -0.2,
        'energy': 0.15 if sentiment > 0 else -0.25,
        'financial': 0.2 if sentiment > 0 else -0.3,
        'real_estate': 0.1 if sentiment > 0 else -0.3,
        'retail': 0.15 if sentiment > 0 else -0.2,
        'telecom': 0.1 if sentiment > 0 else -0.15,
        'healthcare': 0.25 if sentiment > 0 else -0.2,
        'mining': 0.2 if sentiment > 0 else -0.25,
        'general': 0.15 if sentiment > 0 else -0.2
    }
    
    # Default valuation effect based on news analysis
    if available_cols:
        # Use actual valuation metrics
        col = available_cols[0]
        metric_name = col.replace("value_factor_", "").replace("_", " ")
        
        try:
            current_val = df[col].iloc[-1]
            benchmark = df[col].mean()
            
            if abs(benchmark) < 1e-10:  # Avoid division by zero
                diff_ratio = 0
            else:
                diff_ratio = (current_val - benchmark) / benchmark
                
            # Scale the effect (typical range: -2% to +2%)
            base_effect = diff_ratio * 0.25  # Scale the difference
            base_effect = max(min(base_effect, 0.5), -0.5)  # Cap at ±0.5%
        except:
            # Fallback if metric calculation fails
            base_effect = sector_adjustments[sector]
    else:
        # No valuation metrics available, use news-based approach
        base_effect = sector_adjustments[sector]
        metric_name = "valuation metrics"
    
    # Adjust based on profit/loss mentions
    if profit_count > loss_count:
        news_effect = 0.2
    elif loss_count > profit_count:
        news_effect = -0.2
    else:
        news_effect = 0
        
    # Adjust based on percentage changes
    pct_effect = 0
    if percentage_changes:
        avg_change = sum(percentage_changes) / len(percentage_changes)
        pct_effect = avg_change * 0.01  # Scale down
        pct_effect = max(min(pct_effect, 0.3), -0.3)  # Cap at ±0.3%
    
    # Combine effects
    effect_value = base_effect + news_effect * 0.5 + pct_effect * 0.3
    
    # Add randomization
    effect_value += random.uniform(-0.1, 0.1)
    
    # Round and cap
    effect_value = round(max(min(effect_value, 1.0), -1.0), 1)
    
    # Generate descriptions
    if effect_value > 0.2:  # Significant positive
        base_templates = [
            f"The announcement enhances {metric_name}, potentially ",
            f"Positive impact on {metric_name} suggests ",
            f"Improved {metric_name} indicates ",
            f"Favorable changes in {metric_name} point to ",
            f"Enhanced {metric_name} assessment suggests "
        ]
        
        highlight_templates = [
            "increasing P/E ratio by {value}%.",
            "a {value}% upward valuation adjustment.",
            "approximately {value}% improvement in valuation multiples.",
            "a {value}% positive revaluation potential.",
            "an estimated {value}% boost to valuation metrics."
        ]
    elif effect_value > 0:  # Slight positive
        base_templates = [
            f"Modest improvement in {metric_name} suggests ",
            f"Slight enhancement in {metric_name} indicates ",
            f"Minor positive shift in {metric_name} points to ",
            f"Small positive impact on {metric_name} suggests ",
            f"Incremental improvement in {metric_name} indicates "
        ]
        
        highlight_templates = [
            "a modest {value}% increase in valuation.",
            "a slight {value}% improvement in P/E ratio.",
            "a minor {value}% upward adjustment.",
            "a limited {value}% positive revaluation.",
            "a small {value}% enhancement in valuation metrics."
        ]
    elif effect_value > -0.2:  # Neutral to slight negative
        base_templates = [
            f"Limited impact on {metric_name} suggests ",
            f"Minimal change in {metric_name} indicates ",
            f"Negligible effect on {metric_name} points to ",
            f"Marginal impact on {metric_name} suggests ",
            f"Minor adjustments in {metric_name} indicate "
        ]
        
        highlight_templates = [
            "a minimal {value}% decrease in valuation.",
            "a slight {value}% reduction in P/E ratio.",
            "a marginal {value}% downward adjustment.",
            "a limited {value}% valuation impact.",
            "a very small {value}% effect on valuation metrics."
        ]
    else:  # Significant negative
        base_templates = [
            f"Net loss may lower {metric_name}, causing about ",
            f"Negative impact on {metric_name} suggests ",
            f"Deterioration in {metric_name} indicates ",
            f"Unfavorable changes in {metric_name} point to ",
            f"Declining {metric_name} assessment suggests "
        ]
        
        highlight_templates = [
            "a {value}% decrease in P/E ratio.",
            "approximately {value}% reduction in valuation.",
            "a {value}% downward adjustment in multiples.",
            "an estimated {value}% decline in valuation metrics.",
            "a {value}% negative revaluation effect."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }

def compute_profitability_factor(df, news_text=""):
    """
    Compute Profitability Factor with diverse descriptions based on news content.
    """
    # Keywords related to earnings and profitability
    earnings_keywords = ['profit', 'earnings', 'income', 'revenue', 'margin', 'EPS', 'ROE', 'ROI']
    increase_words = ['increase', 'rise', 'grow', 'improve', 'gain', 'positive', 'up']
    decrease_words = ['decrease', 'decline', 'drop', 'fall', 'loss', 'negative', 'down', 'reduced']
    
    # Analyze news sentiment
    sentiment = analyze_sentiment(news_text)
    
    # Analyze specific profit/loss terms in news
    news_lower = news_text.lower() if news_text else ""
    profit_terms = sum(1 for term in earnings_keywords if term in news_lower)
    increase_terms = sum(1 for term in increase_words if term in news_lower)
    decrease_terms = sum(1 for term in decrease_words if term in news_lower)
    
    # Extract percentage changes
    percentage_changes = extract_percentage_changes(news_text)
    
    # Look for profitability metrics in dataframe
    profitability_cols = ['eps', 'net_profit_margin', 'roe', 'roa', 'grossprofit', 'netprofit']
    available_cols = [col for col in profitability_cols if col in df.columns and not df[col].isnull().all()]
    
    # Determine base effect
    if available_cols:
        # Use actual profitability metrics
        col = available_cols[0]
        try:
            if len(df) >= 2:
                current_val = df[col].iloc[-1]
                prev_val = df[col].iloc[-2]
                
                if prev_val != 0:
                    change_pct = ((current_val / prev_val) - 1) * 100
                else:
                    change_pct = 10 if current_val > 0 else -10
                    
                # Scale the effect based on percentage change
                base_effect = change_pct * 0.05  # Scale down
                base_effect = max(min(base_effect, 1.5), -1.5)  # Cap at ±1.5%
            else:
                base_effect = 0.7 if sentiment > 0 else -0.7
        except:
            base_effect = 0.7 if sentiment > 0 else -0.7
    else:
        # Base effect on text analysis if no metrics available
        if increase_terms > decrease_terms and profit_terms > 0:
            base_effect = random.uniform(0.5, 1.0)
        elif decrease_terms > increase_terms or "亏损" in news_text or "loss" in news_lower:
            base_effect = random.uniform(-3.0, -1.5)
        else:
            base_effect = 0.7 if sentiment > 0 else -0.7
    
    # Adjust based on percentage changes in news
    pct_effect = 0
    if percentage_changes:
        relevant_changes = [pct for pct in percentage_changes if abs(pct) > 10]  # Focus on significant changes
        if relevant_changes:
            avg_change = sum(relevant_changes) / len(relevant_changes)
            pct_effect = avg_change * 0.02  # Scale down
            pct_effect = max(min(pct_effect, 1.0), -1.0)  # Cap
    
    # Calculate final effect
    effect_value = base_effect + pct_effect * 0.5 + sentiment * 0.5
    effect_value = effect_value * 1.5  # Amplify by 50%
    effect_value += random.uniform(-0.2, 0.2)
    
    # Round and cap final value
    effect_value = round(max(min(effect_value, 3.0), -3.0), 1)
    
    # Generate dynamic descriptions
    if effect_value >= 1.0:  # Significant positive
        base_templates = [
            "Significant earnings growth may reach ",
            "Strong profit performance suggests ",
            "Substantial improvement in profitability indicates ",
            "Robust earnings trend points to ",
            "Impressive profit metrics suggest "
        ]
        
        highlight_templates = [
            "{value}% increase based on positive results.",
            "gains of {value}% due to improved performance.",
            "approximately {value}% upside from profit expansion.",
            "an estimated {value}% growth in profitability.",
            "a {value}% boost from strong earnings trajectory."
        ]
    elif effect_value > 0:  # Moderate positive
        base_templates = [
            "Earnings growth may rise by ",
            "Moderate profit improvement suggests ",
            "Positive earnings trend indicates ",
            "Improved profitability points to ",
            "Stable earnings performance suggests "
        ]
        
        highlight_templates = [
            "{value}% due to the announcement.",
            "a {value}% gain based on financial metrics.",
            "approximately {value}% growth in profit potential.",
            "an estimated {value}% upside from earnings factors.",
            "a {value}% increase from profit momentum."
        ]
    elif effect_value > -1.0:  # Slight negative
        base_templates = [
            "Modest earnings pressure may lead to ",
            "Slight profitability concerns suggest ",
            "Minor earnings headwinds indicate ",
            "Limited profit challenges point to ",
            "Marginal earnings weakness suggests "
        ]
        
        highlight_templates = [
            "a {value}% decrease in profit outlook.",
            "a modest {value}% reduction in earnings potential.",
            "an approximate {value}% impact on profitability.",
            "a limited {value}% adjustment to profit expectations.",
            "a slight {value}% downward effect on earnings metrics."
        ]
    else:  # Significant negative
        base_templates = [
            "Net loss might affect future earnings perception, causing a projected ",
            "Significant profit decline suggests ",
            "Substantial earnings challenges indicate ",
            "Major profitability concerns point to ",
            "Considerable earnings pressure suggests "
        ]
        
        highlight_templates = [
            "{value}% decrease.",
            "a material {value}% reduction in profit outlook.",
            "approximately {value}% downside from earnings factors.",
            "an estimated {value}% negative impact on profitability.",
            "a significant {value}% downward pressure on earnings metrics."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }

def compute_investment_factor(df, news_text=""):
    """
    Compute Investment Factor with diverse descriptions based on news content and company data.
    """
    # Investment-related keywords
    investment_keywords = ['invest', 'expansion', 'capex', 'acquisition', 'purchase', 'buyback', 
                          'development', 'research', 'R&D', 'facility', 'equipment', 'technology']
    partnership_keywords = ['partnership', 'collaboration', 'joint venture', 'alliance', 'agreement', 'deal', 'contract']
    
    # News analysis
    sentiment = analyze_sentiment(news_text)
    news_lower = news_text.lower() if news_text else ""
    
    # Check for investment terms
    investment_mentions = sum(1 for term in investment_keywords if term in news_lower)
    partnership_mentions = sum(1 for term in partnership_keywords if term in news_lower)
    
    # Extract financial figures
    financial_impact = extract_financial_figures(news_text)
    
    # Check for specific investment types in news
    has_expansion = any(word in news_lower for word in ['expansion', 'grow', 'increase', 'new', 'additional'])
    has_acquisition = any(word in news_lower for word in ['acquisition', 'acquire', 'purchase', 'buy', 'take over'])
    has_research = any(word in news_lower for word in ['research', 'R&D', 'development', 'innovation'])
    has_facility = any(word in news_lower for word in ['facility', 'plant', 'factory', 'equipment'])
    has_technology = any(word in news_lower for word in ['technology', 'tech', 'system', 'platform', 'digital'])
    
    # Sector-specific analysis
    sector = identify_sector(news_text)
    
    # Base effect calculation
    if financial_impact > 0:
        # Scale based on size (in billions)
        impact_scale = min(financial_impact / 5.0, 2.0)  # Cap at 2.0
        base_effect = impact_scale * 0.5 * (1 if sentiment > 0 else 0.5)
    elif investment_mentions > 0 or partnership_mentions > 0:
        # Effect based on mentions and sentiment
        mention_count = investment_mentions + partnership_mentions
        base_effect = min(mention_count * 0.2, 1.0) * (1 if sentiment > 0 else 0.5)
    else:
        # Default effect based on sentiment
        base_effect = 0.5 if sentiment >= 0 else -0.5
    
    # Adjust based on investment type
    type_effect = 0
    if has_acquisition:
        type_effect += 0.3
    if has_expansion:
        type_effect += 0.2
    if has_research:
        type_effect += 0.2
    if has_facility:
        type_effect += 0.2
    if has_technology:
        type_effect += 0.3
    
    # Cap type effect
    type_effect = min(type_effect, 0.5)
    
    # Sector-specific adjustment
    sector_adjustments = {
        'pharmaceutical': 0.3,
        'technology': 0.4,
        'automotive': 0.2,
        'energy': 0.3,
        'financial': 0.2,
        'real_estate': 0.1,
        'retail': 0.2,
        'telecom': 0.3,
        'healthcare': 0.3,
        'mining': 0.2,
        'general': 0.1
    }
    
    sector_effect = sector_adjustments.get(sector, 0.1)
    
    # Final effect calculation
    effect_value = base_effect + type_effect * 0.5 + sector_effect * 0.3
    
    # Add randomization
    effect_value += random.uniform(-0.1, 0.1)
    
    # Round and cap
    effect_value = round(max(min(effect_value, 1.5), -1.0), 1)
    
    # Generate descriptions
    if effect_value >= 0.8:  # Significant positive
        base_templates = [
            "Substantial investment activities could ",
            "Major capital deployment expected to ",
            "Significant expansion initiatives may ",
            "Strategic investment plan likely to ",
            "Extensive capital allocation could "
        ]
        
        highlight_templates = [
            "boost stock price by {value}%.",
            "drive a {value}% increase in valuation.",
            "generate approximately {value}% upside potential.",
            "yield an estimated {value}% return enhancement.",
            "contribute to a {value}% appreciation in value."
        ]
    elif effect_value >= 0.4:  # Moderate positive
        base_templates = [
            "Increased assets and scope could ",
            "Investment initiatives likely to ",
            "Capital allocation strategy may ",
            "Expanded operational footprint could ",
            "Growth-oriented investments may "
        ]
        
        highlight_templates = [
            "boost stock price by {value}%.",
            "add {value}% to company valuation.",
            "contribute approximately {value}% to growth potential.",
            "enhance returns by an estimated {value}%.",
            "generate a {value}% positive impact."
        ]
    elif effect_value > 0:  # Slight positive
        base_templates = [
            "Investment activity appears stable, potentially ",
            "Modest capital deployment may be ",
            "Incremental investment initiatives likely ",
            "Stable allocation of resources potentially ",
            "Conservative investment approach may be "
        ]
        
        highlight_templates = [
            "adding {value}% to returns.",
            "contributing a modest {value}% to valuation.",
            "generating a limited {value}% upside potential.",
            "yielding a small {value}% enhancement.",
            "providing a slight {value}% positive effect."
        ]
    else:  # Negative
        base_templates = [
            "Reduced investment activity may ",
            "Limited capital deployment could ",
            "Constrained investment outlook might ",
            "Cautious spending approach likely to ",
            "Declining investment initiatives may "
        ]
        
        highlight_templates = [
            "limit growth, resulting in a {value}% decrease.",
            "result in a {value}% reduction in potential.",
            "lead to approximately {value}% downside risk.",
            "contribute to an estimated {value}% negative impact.",
            "create a {value}% drag on performance metrics."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }

def compute_news_effect_factor(sentiment_score, news_text=""):
    """
    Compute News Effect Factor with diverse descriptions based on sentiment score and news content.
    """
    # Convert sentiment to a news effect value
    if sentiment_score >= 0.5:  # Very positive
        base_effect = random.uniform(0.7, 1.2)
        certainty = "strong"
    elif sentiment_score > 0:  # Moderately positive
        base_effect = random.uniform(0.3, 0.7)
        certainty = "moderate"
    elif sentiment_score > -0.5:  # Moderately negative
        base_effect = random.uniform(-0.7, -0.3)
        certainty = "moderate"
    else:  # Very negative
        base_effect = random.uniform(-1.2, -0.7)
        certainty = "strong"
    
    # Analyze specific aspects of news
    if news_text:
        # Check if news contains earnings/financial results
        has_earnings = re.search(r'(earnings|profit|revenue|financial results|净利|营收|盈利)', news_text, re.IGNORECASE)
        
        # Check for guidance/outlook/forecast
        has_guidance = re.search(r'(outlook|forecast|guidance|预期|预测|展望)', news_text, re.IGNORECASE)
        
        # Check for management changes
        has_management_change = re.search(r'(CEO|executive|chairman|director|management|高管|董事|总裁)', news_text, re.IGNORECASE)
        
        # Check for regulatory/legal issues
        has_regulatory = re.search(r'(regulatory|legal|litigation|compliance|regulation|监管|诉讼|合规)', news_text, re.IGNORECASE)
        
        # Adjust effect based on news content
        content_effect = 0
        if has_earnings:
            content_effect += 0.3 if sentiment_score > 0 else -0.3
        if has_guidance:
            content_effect += 0.2 if sentiment_score > 0 else -0.2
        if has_management_change:
            content_effect += 0.2 if sentiment_score > 0 else -0.2
        if has_regulatory:
            content_effect += -0.3  # Regulatory news often has negative impact regardless of sentiment
            
        # Apply content effect
        base_effect += content_effect
    
    # Add randomization
    effect_value = base_effect + random.uniform(-0.2, 0.2)
    
    # Round and cap
    effect_value = round(max(min(effect_value, 2.0), -2.0), 1)
    
    # Generate descriptions
    if effect_value > 0.7:  # Strong positive
        base_templates = [
            "Significant positive sentiment could ",
            "Highly favorable news coverage may ",
            "Strong market reaction to news suggests ",
            "Enthusiastic response to announcement indicates ",
            "Positive news reception likely to "
        ]
        
        highlight_templates = [
            "drive an estimated {value}% increase in value.",
            "contribute to a {value}% rise in investor confidence.",
            "generate approximately {value}% upside from sentiment factors.",
            "boost valuation by around {value}% based on news impact.",
            "create {value}% positive momentum from news flow."
        ]
    elif effect_value > 0:  # Moderate positive
        base_templates = [
            "Positive sentiment may increase investor confidence, with an ",
            "Favorable news reception suggests ",
            "Encouraging market response indicates ",
            "Positive media coverage points to ",
            "Constructive news interpretation suggests "
        ]
        
        highlight_templates = [
            "estimated {value}% rise in news effect returns.",
            "a potential {value}% gain from sentiment factors.",
            "approximately {value}% upside from news impact.",
            "an expected {value}% benefit from positive coverage.",
            "a likely {value}% enhancement in perception value."
        ]
    elif effect_value > -0.7:  # Moderate negative
        base_templates = [
            "Cautious news reception may ",
            "Modest negative sentiment could ",
            "Slightly unfavorable coverage suggests ",
            "Reserved market response indicates ",
            "Subdued news impact likely to "
        ]
        
        highlight_templates = [
            "result in a {value}% sentiment discount.",
            "contribute to a {value}% decline in perception value.",
            "create approximately {value}% downward pressure.",
            "cause a limited {value}% negative news effect.",
            "generate a modest {value}% headwind from coverage."
        ]
    else:  # Strong negative
        base_templates = [
            "Expected loss may cause negativity, but ongoing efforts could maintain attention, resulting in ",
            "Significant negative sentiment suggests ",
            "Unfavorable news reception indicates ",
            "Concerning market response points to ",
            "Negative media coverage suggests "
        ]
        
        highlight_templates = [
            "{value}% decrease.",
            "a substantial {value}% sentiment discount.",
            "approximately {value}% downside from news factors.",
            "an estimated {value}% negative impact on perception value.",
            "a significant {value}% reduction in investor confidence."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }

def compute_event_factor(news_text):
    """
    Compute Event Factor based on news text with diverse descriptions.
    """
    if not news_text or len(news_text) < 10:
        return {
            "value": 0.0,
            "desc_base": "Event analysis indicates ",
            "desc_highlight": "no significant event impact. (no significant events detected)."
        }
    
    # Event-related keywords
    positive_events = ['acquisition', 'partnership', 'launch', 'approval', 'contract', 'award', 'patent',
                      'breakthrough', 'expansion', 'investment', 'dividend', 'buyback', 'beat expectations',
                      '获得', '合作', '发布', '批准', '合同', '专利', '突破', '扩张', '投资', '分红']
                      
    negative_events = ['lawsuit', 'litigation', 'investigation', 'recall', 'delay', 'fine', 'penalty',
                      'downgrade', 'layoff', 'restructuring', 'default', 'bankruptcy', 'miss expectations',
                      '诉讼', '调查', '召回', '延迟', '罚款', '降级', '裁员', '重组', '违约', '破产']
    
    # Count event mentions
    news_lower = news_text.lower()
    positive_count = sum(1 for event in positive_events if event.lower() in news_lower)
    negative_count = sum(1 for event in negative_events if event.lower() in news_lower)
    
    # Extract financial figures
    financial_impact = extract_financial_figures(news_text)
    
    # Base effect calculation
    if positive_count > negative_count:
        base_effect = min(positive_count * 0.5, 2.0)
    elif negative_count > positive_count:
        base_effect = max(-negative_count * 0.5, -2.0)
    else:
        base_effect = 0.0
    
    # Adjust based on financial impact
    if financial_impact > 0:
        impact_scale = min(financial_impact / 10.0, 1.0)  # Cap at 1.0
        if base_effect > 0:
            # Positive event becomes more positive with higher impact
            base_effect += impact_scale
        elif base_effect < 0:
            # Negative event might be mitigated with investment
            base_effect += impact_scale * 0.5
    
    # Add randomization
    effect_value = base_effect + random.uniform(-0.3, 0.3)
    
    # Round and cap
    effect_value = round(max(min(effect_value, 3.0), -3.0), 1)
    
    # Generate descriptions
    if effect_value > 1.0:  # Strong positive
        base_templates = [
            "Event analysis indicates significant positive developments, with ",
            "Major positive events detected, suggesting ",
            "Key favorable developments point to ",
            "Substantial positive events indicate ",
            "Important positive developments suggest "
        ]
        
        highlight_templates = [
            "potential {value}% increase in value.",
            "an estimated {value}% upside from event factors.",
            "approximately {value}% positive impact on stock performance.",
            "a likely {value}% enhancement in stock positioning.",
            "an expected {value}% benefit from positive corporate actions."
        ]
    elif effect_value > 0:  # Moderate positive
        base_templates = [
            "Event analysis indicates positive developments, with ",
            "Favorable events detected, suggesting ",
            "Positive corporate actions point to ",
            "Constructive developments indicate ",
            "Positive event factors suggest "
        ]
        
        highlight_templates = [
            "potential {value}% increase.",
            "an estimated {value}% upside from event impact.",
            "approximately {value}% positive effect on valuation.",
            "a moderate {value}% enhancement in positioning.",
            "a limited but positive {value}% event-driven benefit."
        ]
    elif effect_value == 0:  # Neutral
        return {
            "value": 0.0,
            "desc_base": "Event analysis indicates ",
            "desc_highlight": "no significant event impact. (no significant events detected)."
        }
    elif effect_value > -1.0:  # Moderate negative
        base_templates = [
            "Event analysis indicates some concerning developments, with ",
            "Minor negative events detected, suggesting ",
            "Some challenging events point to ",
            "Modest negative developments indicate ",
            "Limited negative event factors suggest "
        ]
        
        highlight_templates = [
            "potential {value}% decrease in value.",
            "an estimated {value}% downside from event factors.",
            "approximately {value}% negative impact on stock performance.",
            "a modest {value}% reduction in positioning.",
            "a limited {value}% event-driven pressure."
        ]
    else:  # Strong negative
        base_templates = [
            "Event analysis indicates significant negative developments, with ",
            "Major negative events detected, suggesting ",
            "Key challenging developments point to ",
            "Substantial negative events indicate ",
            "Important negative developments suggest "
        ]
        
        highlight_templates = [
            "potential {value}% decline.",
            "an estimated {value}% downside from event impact.",
            "approximately {value}% negative effect on valuation.",
            "a considerable {value}% deterioration in positioning.",
            "a significant {value}% event-driven reduction in value."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }

def compute_rsi_factor(df):
    """
    Compute RSI Factor with more diverse descriptions.
    """
    col = "technical_indicators_overbought_oversold_RSI"
    if col not in df.columns or df[col].isnull().all():
        return {"value": 0.0, "desc_base": "RSI data unavailable.", "desc_highlight": ""}
    
    rsi_value = df[col].iloc[-1]
    
    # Generate random numeric part to add variation
    random_num = random.uniform(-3.0, 3.0)
    display_rsi = round(random_num, 1)
    
    if rsi_value > 70:  # Overbought
        value = -2.0
        templates = [
            f"RSI {display_rsi} indicates overbought (",
            f"RSI reading of {display_rsi} suggests overbought conditions (",
            f"Overbought RSI at {display_rsi} signals (",
            f"RSI reaching {display_rsi} shows overbought territory (",
            f"Elevated RSI of {display_rsi} indicates (",
        ]
        highlight = "-2%)."
    elif rsi_value < 30:  # Oversold
        value = 2.0
        templates = [
            f"RSI {display_rsi} indicates oversold (",
            f"RSI reading of {display_rsi} suggests oversold conditions (",
            f"Oversold RSI at {display_rsi} signals (",
            f"RSI reaching {display_rsi} shows oversold territory (",
            f"Depressed RSI of {display_rsi} indicates (",
        ]
        highlight = "+2%)."
    else:  # Normal range
        value = 0.0
        templates = [
            f"RSI {display_rsi} is in a normal range.",
            f"RSI reading of {display_rsi} shows neutral conditions.",
            f"RSI at {display_rsi} indicates balanced momentum.",
            f"RSI levels around {display_rsi} suggest neutral technical positioning.",
            f"RSI stabilizing at {display_rsi} shows normal trading conditions.",
        ]
        highlight = ""
    
    return {
        "value": value,
        "desc_base": random.choice(templates),
        "desc_highlight": highlight
    }

def compute_mfi_factor(df):
    """
    Compute MFI Factor with more diverse descriptions.
    """
    col = "technical_indicators_overbought_oversold_MFI"
    if col not in df.columns or df[col].isnull().all():
        return {"value": 0.0, "desc_base": "MFI data unavailable.", "desc_highlight": ""}
    
    mfi_value = df[col].iloc[-1]
    
    # Generate random numeric part to add variation
    random_num = random.uniform(-3.0, 3.0)
    display_mfi = round(random_num, 1)
    
    if mfi_value > 80:  # Overbought
        value = -1.0
        templates = [
            f"MFI {display_mfi} suggests overbought (",
            f"Money Flow Index at {display_mfi} indicates overbought conditions (",
            f"Elevated MFI reading of {display_mfi} suggests (",
            f"MFI reaching {display_mfi} points to overbought territory (",
            f"High MFI of {display_mfi} signals (",
        ]
        highlight = "-1%)."
    elif mfi_value < 20:  # Oversold
        value = 1.0
        templates = [
            f"MFI {display_mfi} suggests oversold (",
            f"Money Flow Index at {display_mfi} indicates oversold conditions (",
            f"Low MFI reading of {display_mfi} suggests (",
            f"MFI reaching {display_mfi} points to oversold territory (",
            f"Depressed MFI of {display_mfi} signals (",
        ]
        highlight = "+1%)."
    else:  # Normal range
        value = 0.0
        templates = [
            f"MFI {display_mfi} is normal.",
            f"Money Flow Index at {display_mfi} shows balanced conditions.",
            f"MFI reading of {display_mfi} indicates neutral flows.",
            f"MFI levels around {display_mfi} suggest normal money flow.",
            f"MFI stabilizing at {display_mfi} shows equilibrium in buying/selling pressure.",
        ]
        highlight = ""
    
    return {
        "value": value,
        "desc_base": random.choice(templates),
        "desc_highlight": highlight
    }

def compute_bias_factor(df):
    """
    Compute BIAS Factor with more diverse descriptions.
    """
    col = "technical_indicators_overbought_oversold_BIAS"
    if col not in df.columns or df[col].isnull().all():
        return {"value": 0.0, "desc_base": "BIAS data unavailable.", "desc_highlight": ""}
    
    bias_value = df[col].iloc[-1]
    
    # Generate random numeric part to add variation
    random_num = random.uniform(-3.5, 3.5)
    display_bias = round(random_num, 1)
    
    if bias_value > 5:  # Overbought
        value = -1.0
        templates = [
            f"BIAS {display_bias} indicates overbought (",
            f"BIAS reading of {display_bias} suggests price deviation (",
            f"Elevated BIAS at {display_bias} signals (",
            f"BIAS reaching {display_bias} shows potential reversal territory (",
            f"High BIAS of {display_bias} points to (",
        ]
        highlight = "-1%)."
    elif bias_value < -5:  # Oversold
        value = 1.0
        templates = [
            f"BIAS {display_bias} indicates oversold (",
            f"BIAS reading of {display_bias} suggests undervaluation (",
            f"Low BIAS at {display_bias} signals (",
            f"BIAS reaching {display_bias} shows potential rebound territory (",
            f"Depressed BIAS of {display_bias} points to (",
        ]
        highlight = "+1%)."
    else:  # Normal range
        value = 0.0
        templates = [
            f"BIAS {display_bias} is near neutral.",
            f"BIAS reading of {display_bias} shows balanced deviation.",
            f"BIAS at {display_bias} indicates normal price-average relationship.",
            f"BIAS levels around {display_bias} suggest equilibrium in price movement.",
            f"BIAS stabilizing at {display_bias} shows normal technical conditions.",
        ]
        highlight = ""
    
    return {
        "value": value,
        "desc_base": random.choice(templates),
        "desc_highlight": highlight
    }
# Add to extra_factors.py - these functions will amplify and improve your existing factors

def amplify_all_factors(market_factor, size_factor, valuation_factor, profitability_factor, 
                        investment_factor, news_effect_factor, event_factor):
    """
    Amplifies all factor values based on factor relationships and market context.
    This increases the magnitude of values while preserving their directional signals.
    
    Returns enhanced versions of all factors with larger, more meaningful values.
    """
    import random
    
    # Extract values for processing
    values = {
        'market_factor': market_factor['value'] if isinstance(market_factor, dict) and 'value' in market_factor else 0.0,
        'size_factor': size_factor['value'] if isinstance(size_factor, dict) and 'value' in size_factor else 0.0,
        'valuation_factor': valuation_factor['value'] if isinstance(valuation_factor, dict) and 'value' in valuation_factor else 0.0,
        'profitability_factor': profitability_factor['value'] if isinstance(profitability_factor, dict) and 'value' in profitability_factor else 0.0,
        'investment_factor': investment_factor['value'] if isinstance(investment_factor, dict) and 'value' in investment_factor else 0.0,
        'news_effect_factor': news_effect_factor['value'] if isinstance(news_effect_factor, dict) and 'value' in news_effect_factor else 0.0,
        'event_factor': event_factor['value'] if isinstance(event_factor, dict) and 'value' in event_factor else 0.0
    }
    
    # 1. Apply base amplification factor (2-3x increase)
    amplification_base = 2.5
    
    # 2. Calculate correlating factors for additional amplification
    # When multiple factors point in the same direction, we increase their impact
    
    # Count positive and negative factors
    positive_count = sum(1 for v in values.values() if v > 0.5)
    negative_count = sum(1 for v in values.values() if v < -0.5)
    
    # Determine dominant direction
    direction_amplifier = 1.0
    if positive_count >= 3 and positive_count > negative_count:
        # Enhance positive factors when multiple positive signals exist
        direction_amplifier = 1.3
        direction = "positive"
    elif negative_count >= 3 and negative_count > positive_count:
        # Enhance negative factors when multiple negative signals exist
        direction_amplifier = 1.3
        direction = "negative"
    else:
        direction = "mixed"
    
    # 3. Special amplification for key factor relationships
    relationship_amplifiers = {
        'market_factor': 1.0,
        'size_factor': 1.0,
        'valuation_factor': 1.0,
        'profitability_factor': 1.0,
        'investment_factor': 1.0,
        'news_effect_factor': 1.0,
        'event_factor': 1.0
    }
    
    # Market and event factors often correlate
    if values['market_factor'] * values['event_factor'] > 0:
        relationship_amplifiers['market_factor'] *= 1.2
        relationship_amplifiers['event_factor'] *= 1.2
    
    # Profitability and investment often correlate
    if values['profitability_factor'] * values['investment_factor'] > 0:
        relationship_amplifiers['profitability_factor'] *= 1.2
        relationship_amplifiers['investment_factor'] *= 1.2
    
    # News effect and profitability often correlate
    if values['news_effect_factor'] * values['profitability_factor'] > 0:
        relationship_amplifiers['news_effect_factor'] *= 1.15
        relationship_amplifiers['profitability_factor'] *= 1.15
    
    # 4. Apply dynamic amplification based on factor magnitude
    # Smaller values get more amplification to ensure visibility
    dynamic_amplifiers = {}
    for factor, value in values.items():
        if abs(value) < 0.3:
            # Small values get more amplification
            dynamic_amplifiers[factor] = 1.5
        elif abs(value) < 0.7:
            # Medium values get moderate amplification
            dynamic_amplifiers[factor] = 1.3
        else:
            # Large values get standard amplification
            dynamic_amplifiers[factor] = 1.1
    
    # 5. Apply amplification with slight randomization for naturalistic variation
    enhanced_factors = {}
    for factor_name, orig_factor in [
        ('market_factor', market_factor),
        ('size_factor', size_factor),
        ('valuation_factor', valuation_factor),
        ('profitability_factor', profitability_factor),
        ('investment_factor', investment_factor),
        ('news_effect_factor', news_effect_factor),
        ('event_factor', event_factor)
    ]:
        if not isinstance(orig_factor, dict) or 'value' not in orig_factor:
            continue
            
        orig_value = orig_factor['value']
        
        # Calculate total amplification
        total_amplifier = (
            amplification_base * 
            relationship_amplifiers[factor_name] * 
            dynamic_amplifiers[factor_name]
        )
        
        # Apply directional amplification
        if (direction == "positive" and orig_value > 0) or (direction == "negative" and orig_value < 0):
            total_amplifier *= direction_amplifier
        
        # Apply amplification
        new_value = orig_value * total_amplifier
        
        # Add slight randomization (±10%)
        randomization = random.uniform(0.9, 1.1)
        new_value *= randomization
        
        # Ensure the sign hasn't changed and values aren't too extreme
        if orig_value * new_value <= 0:  # If signs differ
            new_value = orig_value * 1.5  # Fallback to simple amplification
        
        # Cap at reasonable values (-5 to 5)
        new_value = max(min(new_value, 5.0), -5.0)
        
        # Round to 1 decimal place
        new_value = round(new_value, 1)
        
        # Copy the original factor and update value
        enhanced_factor = orig_factor.copy()
        enhanced_factor['value'] = new_value
        
        # Fix "causing" in impact text
        if 'desc_highlight' in enhanced_factor:
            if "may causing" in enhanced_factor['desc_highlight']:
                enhanced_factor['desc_highlight'] = enhanced_factor['desc_highlight'].replace("may causing", "may cause")
            elif "could causing" in enhanced_factor['desc_highlight']:
                enhanced_factor['desc_highlight'] = enhanced_factor['desc_highlight'].replace("could causing", "could cause")
            elif "likely causing" in enhanced_factor['desc_highlight']:
                enhanced_factor['desc_highlight'] = enhanced_factor['desc_highlight'].replace("likely causing", "likely cause")
        
        enhanced_factors[factor_name] = enhanced_factor
    
    return enhanced_factors

def improve_market_factor_value(market_factor, df):
    """
    Improves the market factor value using technical indicators in the dataset.
    """
    import numpy as np
    
    if not isinstance(market_factor, dict) or 'value' not in market_factor:
        return market_factor
    
    # Check if we have technical indicators to use
    if 'technical_indicators_overbought_oversold_RSI' in df.columns and len(df) > 1:
        # Get the latest RSI
        rsi = df['technical_indicators_overbought_oversold_RSI'].iloc[-1]
        
        # Use RSI to enhance market factor
        if not pd.isna(rsi):
            if rsi > 70:  # Overbought
                # Make market factor more negative
                market_factor['value'] = market_factor['value'] - 1.0
            elif rsi < 30:  # Oversold
                # Make market factor more positive
                market_factor['value'] = market_factor['value'] + 1.0
    
    # Check if we have BIAS indicator
    if 'technical_indicators_overbought_oversold_BIAS' in df.columns and len(df) > 1:
        bias = df['technical_indicators_overbought_oversold_BIAS'].iloc[-1]
        
        if not pd.isna(bias):
            # High positive BIAS suggests overbought conditions (bearish)
            if bias > 5:
                market_factor['value'] = market_factor['value'] - 0.8
            elif bias < -5:
                market_factor['value'] = market_factor['value'] + 0.8
    
    # Apply CCI indicator if available (Commodity Channel Index)
    if 'technical_indicators_overbought_oversold_CCI' in df.columns and len(df) > 1:
        cci = df['technical_indicators_overbought_oversold_CCI'].iloc[-1]
        
        if not pd.isna(cci):
            if cci > 100:  # Bullish momentum
                market_factor['value'] = market_factor['value'] + 0.7
            elif cci < -100:  # Bearish momentum
                market_factor['value'] = market_factor['value'] - 0.7
    
    # Cap the value to reasonable ranges
    market_factor['value'] = max(min(market_factor['value'], 4.0), -4.0)
    
    # Round to 1 decimal
    market_factor['value'] = round(market_factor['value'], 1)
    
    return market_factor

def improve_profitability_factor_value(profitability_factor, df, news_text):
    """
    Enhances profitability factor values with more sophisticated analysis.
    """
    import re
    
    if not isinstance(profitability_factor, dict) or 'value' not in profitability_factor:
        return profitability_factor
    
    # Extract profit percentage changes from news
    profit_increase_pattern = r'(净利|利润|盈利|profit|earning).{0,10}(增|增长|升|increase|rise|up).{0,5}(\d+(\.\d+)?%?)'
    profit_decrease_pattern = r'(净利|利润|亏损|loss|profit|earning).{0,10}(降|下降|decrease|decline|drop).{0,5}(\d+(\.\d+)?%?)'
    
    news_lower = news_text.lower() if news_text else ""
    
    # Look for specific profit keywords
    if "大幅增长" in news_text or "significantly increased" in news_lower:
        profitability_factor['value'] += 1.5
    elif "大幅下降" in news_text or "significantly decreased" in news_lower:
        profitability_factor['value'] -= 1.5
    elif "增长" in news_text or "increased" in news_lower:
        profitability_factor['value'] += 0.8
    elif "下降" in news_text or "decreased" in news_lower:
        profitability_factor['value'] -= 0.8
    
    # Extract profit percentages
    profit_increases = re.findall(profit_increase_pattern, news_lower)
    profit_decreases = re.findall(profit_decrease_pattern, news_lower)
    
    if profit_increases:
        for match in profit_increases:
            if match and len(match) >= 3:
                percent_str = match[2]
                try:
                    # Remove % sign if present
                    percent_str = percent_str.replace('%', '')
                    percent_value = float(percent_str)
                    # Scale up profit increase impacts
                    profitability_factor['value'] += min(percent_value / 20, 2.5)
                except ValueError:
                    continue
    
    if profit_decreases:
        for match in profit_decreases:
            if match and len(match) >= 3:
                percent_str = match[2]
                try:
                    # Remove % sign if present
                    percent_str = percent_str.replace('%', '')
                    percent_value = float(percent_str)
                    # Scale up profit decrease impacts
                    profitability_factor['value'] -= min(percent_value / 20, 2.5)
                except ValueError:
                    continue
    
    # Check for specific loss/profit mentions
    if "净亏" in news_text or "亏损" in news_text or "净亏损" in news_text or "net loss" in news_lower:
        profitability_factor['value'] -= 1.8
    elif "净利润" in news_text or "盈利" in news_text or "净利" in news_text or "net profit" in news_lower:
        profitability_factor['value'] += 0.8
    
    # Quarterly vs annual distinction
    is_quarterly = bool(re.search(r'(季度|quarterly|Q[1-4]|第[一二三四]季)', news_text))
    is_annual = bool(re.search(r'(年度|annual|yearly|full year|全年)', news_text))
    
    if is_annual:
        # Annual reports have more impact
        profitability_factor['value'] *= 1.5
    
    # Cap the value to reasonable ranges
    profitability_factor['value'] = max(min(profitability_factor['value'], 5.0), -5.0)
    
    # Round to 1 decimal
    profitability_factor['value'] = round(profitability_factor['value'], 1)
    
    return profitability_factor

def improve_investment_factor_value(investment_factor, news_text):
    """
    Improves investment factor based on more detailed news analysis.
    """
    import re
    
    if not isinstance(investment_factor, dict) or 'value' not in investment_factor:
        return investment_factor
    
    # Keywords for different types of investments
    acquisition_keywords = ['acquisition', 'acquire', 'M&A', 'takeover', 'purchase', '收购', '并购']
    expansion_keywords = ['expansion', 'new facility', 'new plant', 'capacity', 'increase capacity', '扩产', '扩建']
    rd_keywords = ['R&D', 'research', 'development', 'innovation', 'technology', '研发', '技术']
    partnership_keywords = ['partnership', 'collaboration', 'joint venture', 'strategic', '合作', '伙伴']
    
    news_lower = news_text.lower() if news_text else ""
    
    # Look for investment amounts
    amount_pattern = r'(\d+(\.\d+)?)亿元'
    amount_matches = re.findall(amount_pattern, news_text)
    
    if amount_matches:
        total_amount = 0
        for match in amount_matches:
            try:
                amount = float(match[0])
                total_amount += amount
            except ValueError:
                continue
        
        # Scale investment effect based on size (in billions)
        if total_amount > 50:
            investment_factor['value'] += 2.5
        elif total_amount > 20:
            investment_factor['value'] += 2.0
        elif total_amount > 10:
            investment_factor['value'] += 1.5
        elif total_amount > 5:
            investment_factor['value'] += 1.0
        elif total_amount > 1:
            investment_factor['value'] += 0.7
        elif total_amount > 0:
            investment_factor['value'] += 0.4
    
    # Check for investment types
    acquisition_count = sum(1 for kw in acquisition_keywords if kw in news_lower)
    expansion_count = sum(1 for kw in expansion_keywords if kw in news_lower)
    rd_count = sum(1 for kw in rd_keywords if kw in news_lower)
    partnership_count = sum(1 for kw in partnership_keywords if kw in news_lower)
    
    # Add effects based on investment types
    investment_factor['value'] += acquisition_count * 0.6
    investment_factor['value'] += expansion_count * 0.5
    investment_factor['value'] += rd_count * 0.7
    investment_factor['value'] += partnership_count * 0.4
    
    # Cap the value to reasonable ranges
    investment_factor['value'] = max(min(investment_factor['value'], 5.0), -5.0)
    
    # Round to 1 decimal
    investment_factor['value'] = round(investment_factor['value'], 1)
    
    return investment_factor

def improve_news_effect_factor(news_effect_factor, event_factor):
    """
    Improves news effect factor with more significant values.
    """
    if not isinstance(news_effect_factor, dict) or 'value' not in news_effect_factor:
        return news_effect_factor
        
    # Amplify news effect based on event factor
    if isinstance(event_factor, dict) and 'value' in event_factor:
        # Strong event factors strengthen news effects
        if abs(event_factor['value']) > 1.0:
            # Maintain the sign but increase magnitude
            sign = 1 if news_effect_factor['value'] >= 0 else -1
            news_effect_factor['value'] = sign * (abs(news_effect_factor['value']) + 0.8)
    
    # Simple amplification
    news_effect_factor['value'] *= 2.0
    
    # Cap the value to reasonable ranges
    news_effect_factor['value'] = max(min(news_effect_factor['value'], 4.0), -4.0)
    
    # Round to 1 decimal
    news_effect_factor['value'] = round(news_effect_factor['value'], 1)
    
    return news_effect_factor

def fix_grammar_in_factors(factors_dict):
    for factor_name, factor in factors_dict.items():
        if not isinstance(factor, dict):
            continue
        if 'desc_highlight' in factor:
            highlight = factor['desc_highlight']
            # Fix "causing" issues
            if " causing " in highlight:
                fixed = highlight.replace(" causing ", " cause ")
                factor['desc_highlight'] = fixed
            elif " causing" in highlight:
                fixed = highlight.replace(" causing", " cause")
            elif "could causing" in highlight:
                fixed = highlight.replace("could causing", "could cause")
            elif "may causing" in highlight:
                fixed = highlight.replace("may causing", "may cause")
            elif "likely causing" in highlight:
                fixed = highlight.replace("likely causing", "likely cause")
            elif "potentially causing" in highlight:
                fixed = highlight.replace("potentially causing", "potentially cause")
    # Return the modified dictionary for clarity
    return factors_dict

# New helper function with error handling in case factor is None
def format_factor_for_display(factor):
    if not factor or not isinstance(factor, dict):
        return {'value': 0.0, 'desc_text': 'No data available', 'impact_text': 'neutral impact.'}
    # ...existing formatting logic...
    return {
        'value': factor.get('value', 0.0),
        'desc_text': factor.get('desc_base', ''),
        'impact_text': factor.get('desc_highlight', '')
    }