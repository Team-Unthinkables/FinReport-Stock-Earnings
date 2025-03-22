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
    Compute Event Factor based on news text with improved Chinese text handling.
    """
    if not news_text or len(news_text) < 10:
        return {
            "value": 0.0,
            "desc_base": "Event analysis indicates ",
            "desc_highlight": "no significant event impact."
        }
    
    # Event-related keywords in both English and Chinese
    positive_events = [
        'acquisition', 'partnership', 'launch', 'approval', 'contract', 'award', 'patent',
        'breakthrough', 'expansion', 'investment', 'dividend', 'buyback', 'profit', 'increase',
        'growth', 'revenue', 'earnings', 'win', 'positive', 'cooperation', 'FDA', 'success',
        # Chinese keywords
        '增长', '获得', '合作', '发布', '批准', '合同', '专利', '突破', '扩张', '投资', '分红',
        '回购', '净利润', '收入', '盈利', '中标', '收购', '成功', '提高', '上升'
    ]
    
    negative_events = [
        'lawsuit', 'litigation', 'investigation', 'recall', 'delay', 'fine', 'penalty',
        'downgrade', 'layoff', 'restructuring', 'default', 'bankruptcy', 'decline', 'decrease',
        'loss', 'debt', 'missed', 'lower', 'fall', 'drop', 'negative',
        # Chinese keywords
        '诉讼', '调查', '召回', '延迟', '罚款', '降级', '裁员', '重组', '违约', '破产',
        '亏损', '下降', '减少', '下跌', '跌幅', '债务', '负债', '减持', '不能如期'
    ]
    
    news_lower = news_text.lower()
    positive_count = sum(3 if event in news_lower else 0 for event in positive_events)
    negative_count = sum(3 if event in news_lower else 0 for event in negative_events)
    
    import re
    percentage_pattern = r'(\d+(?:\.\d+)?)%'
    percentage_matches = re.findall(percentage_pattern, news_text)
    percentages = [float(match) for match in percentage_matches if float(match) > 10]
    
    amount_pattern = r'(\d+(?:\.\d+)?)亿元'
    amount_matches = re.findall(amount_pattern, news_text)
    amounts = [float(match) for match in amount_matches]
    
    if positive_count > negative_count * 1.5:  # Significantly positive
        event_type = "positive"
        base_effect = min((positive_count - negative_count) * 0.1, 1.5)
        if percentages and max(percentages) > 50:
            base_effect += 0.5
        if amounts and sum(amounts) > 10:
            base_effect += 0.3
    elif negative_count > positive_count * 1.5:  # Significantly negative
        event_type = "negative"
        base_effect = max((negative_count - positive_count) * -0.1, -1.5)
        if percentages and max(percentages) > 50:
            base_effect -= 0.5
        if amounts and sum(amounts) > 50:
            base_effect -= 0.7
    else:
        if "中标" in news_text or "获得" in news_text or "合作" in news_text:
            event_type = "positive"
            base_effect = 0.5
        elif "亏损" in news_text or "减持" in news_text or "债务" in news_text:
            event_type = "negative"
            base_effect = -0.5
        else:
            event_type = "neutral"
            base_effect = 0.0
    
    effect_value = round(max(min(base_effect, 2.0), -2.0), 1)
    
    import random
    if effect_value > 0.3:
        base_templates = [
            "Event analysis reveals significant positive developments, with ",
            "Major positive corporate events detected, indicating ",
            "Key business developments suggest ",
            "Significant corporate actions point to ",
            "Important positive business events indicate "
        ]
        highlight_templates = [
            "potential {value}% increase from event factors.",
            "an estimated {value}% positive event impact.",
            "approximately {value}% upside from these developments.",
            "a likely {value}% enhancement from these events.",
            "a projected {value}% gain from identified positive events."
        ]
    elif effect_value < -0.3:
        base_templates = [
            "Event analysis identifies concerning developments, with ",
            "Negative corporate events detected, suggesting ",
            "Challenging business developments indicate ",
            "Adverse corporate actions point to ",
            "Significant negative events suggest "
        ]
        highlight_templates = [
            "potential {value}% downside from event factors.",
            "an estimated {value}% negative event impact.",
            "approximately {value}% downward pressure from these developments.",
            "a likely {value}% decline from these events.",
            "a projected {value}% decrease from identified negative events."
        ]
    else:
        base_templates = [
            "Event analysis shows limited impact developments, with ",
            "Minor corporate events detected, suggesting ",
            "Business developments indicate ",
            "Current corporate actions point to ",
            "The identified events suggest "
        ]
        highlight_templates = [
            "minimal {value}% impact from event factors.",
            "a marginal {value}% event effect.",
            "approximately {value}% influence from these developments.",
            "a limited {value}% impact from these events.",
            "a small {value}% change from identified events."
        ]
    
    desc_base = random.choice(base_templates)
    desc_highlight = random.choice(highlight_templates).format(value=abs(effect_value))
    
    return {
        "value": effect_value,
        "desc_base": desc_base,
        "desc_highlight": desc_highlight
    }
