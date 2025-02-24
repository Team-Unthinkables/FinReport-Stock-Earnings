from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the FinBERT model and tokenizer from Hugging Face
# Make sure you have installed the 'transformers' library.
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def get_sentiment_score(text):
    # Tokenize the input text with truncation (max_length=128)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    logits = outputs.logits[0].cpu().numpy()
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    raw_score = probs[2] - probs[0]
    # Apply a tanh transformation to compress extreme values:
    scale = 2  # Adjust scale as needed; here, a score of 1 becomes tanh(2)=0.964
    score = np.tanh(raw_score * scale)
    return score

