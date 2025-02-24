# src/news_aggregator.py
import numpy as np
from advanced_news import compute_event_factor
from sentiment import get_sentiment_score

def aggregate_news_factors(news_texts):
    """
    Given a list of news texts, compute the aggregated news effect and event factors.
    Returns a dictionary containing:
      - aggregated_sentiment: average sentiment score
      - aggregated_event: dict with keys "value", "desc_base", "desc_highlight"
      - news_effect: dict representing overall news effect factor based on aggregated sentiment.
    """
    if not news_texts:
        return None

    sentiment_scores = []
    event_factors = []
    
    for text in news_texts:
        sentiment = get_sentiment_score(text)
        sentiment_scores.append(sentiment)
        event_factor = compute_event_factor(text)
        event_factors.append(event_factor)
    
    aggregated_sentiment = np.mean(sentiment_scores)
    
    # Average numeric values from event factors and concatenate the highlighted parts.
    event_values = [ef["value"] for ef in event_factors]
    aggregated_event_value = np.mean(event_values) if event_values else 0.0
    aggregated_event_descs = "; ".join([ef["desc_highlight"] for ef in event_factors if ef["desc_highlight"]])
    aggregated_event = {
        "value": aggregated_event_value,
        "desc_base": "Event analysis indicates ",
        "desc_highlight": aggregated_event_descs if aggregated_event_descs else "no significant event impact."
    }
    
    # Compute overall news effect factor based on aggregated sentiment.
    if aggregated_sentiment >= 0:
        news_effect = {
            "value": +0.5,
            "desc_base": "Overall news effect shows ",
            "desc_highlight": "positive sentiment, estimated effect +0.5%."
        }
    else:
        news_effect = {
            "value": -0.5,
            "desc_base": "Overall news effect shows ",
            "desc_highlight": "negative sentiment, estimated effect -0.5%."
        }
    
    return {
        "aggregated_sentiment": aggregated_sentiment,
        "aggregated_event": aggregated_event,
        "news_effect": news_effect
    }
