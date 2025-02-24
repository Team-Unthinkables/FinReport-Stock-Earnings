# src/advanced_news.py
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction  # ensure models are imported

# Load the pre-trained SRL model from AllenNLP
PREDICTOR = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

def extract_events(news_text):
    """
    Uses AllenNLP's SRL predictor to extract events from news text.
    Returns a list of event dictionaries.
    """
    prediction = PREDICTOR.predict(sentence=news_text)
    events = []
    # Iterate over each verb info, which contains 'verb' and 'tags'
    for verb_info in prediction["verbs"]:
        predicate = verb_info["verb"]
        tags = verb_info["tags"]
        arguments = {}
        current_arg = None
        arg_tokens = []
        for token, tag in zip(prediction["words"], tags):
            if tag.startswith("B-"):
                if current_arg is not None:
                    arguments[current_arg] = " ".join(arg_tokens)
                current_arg = tag[2:]
                arg_tokens = [token]
            elif tag.startswith("I-") and current_arg is not None:
                arg_tokens.append(token)
            else:
                if current_arg is not None:
                    arguments[current_arg] = " ".join(arg_tokens)
                    current_arg = None
                    arg_tokens = []
        if current_arg is not None:
            arguments[current_arg] = " ".join(arg_tokens)
        events.append({"predicate": predicate, "arguments": arguments})
    return events

def compute_event_factor(news_text):
    """
    Process the news text to extract events and compute a factor.
    For demonstration, if an event contains keywords like 'loss', 'decline', 'warning'
    we assign a negative impact, and if it contains 'acquire', 'growth', 'increase', we assign a positive impact.
    Returns a dictionary with 'value', 'desc_base', and 'desc_highlight'.
    """
    events = extract_events(news_text)
    score = 0.0
    descriptions = []
    
    # Define simple keyword-based rules (expand these as needed)
    negative_keywords = ["loss", "decline", "warning", "drop", "decrease", "profit warning"]
    positive_keywords = ["acquire", "growth", "increase", "boost", "gain", "share repurchase"]
    
    for event in events:
        predicate = event["predicate"].lower()
        negative_hits = sum(1 for word in negative_keywords if word in predicate)
        for arg in event["arguments"].values():
            if any(word in arg.lower() for word in negative_keywords):
                negative_hits += 1
        positive_hits = sum(1 for word in positive_keywords if word in predicate)
        for arg in event["arguments"].values():
            if any(word in arg.lower() for word in positive_keywords):
                positive_hits += 1

        score += (positive_hits - negative_hits)
        if positive_hits > negative_hits:
            descriptions.append(f"Detected positive event: '{event['predicate']}'")
        elif negative_hits > positive_hits:
            descriptions.append(f"Detected negative event: '{event['predicate']}'")
    
    if score < 0:
        factor_value = -2.0
        highlight = "potential 2% decline."
    elif score > 0:
        factor_value = +2.0
        highlight = "potential 2% increase."
    else:
        factor_value = 0.0
        highlight = "no significant event impact."
    
    base_text = "Event analysis indicates "
    event_summary = "; ".join(descriptions) if descriptions else "no significant events detected"
    return {
        "value": factor_value,
        "desc_base": base_text,
        "desc_highlight": f"{highlight} ({event_summary})."
    }
