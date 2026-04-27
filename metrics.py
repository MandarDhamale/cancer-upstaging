import numpy as np
import string
import collections
import re
from sklearn.metrics import f1_score
from bert_score import score as bert_score

def calculate_macro_f1(y_true, y_pred):
    """
    Calculate the macro-averaged F1 score.
    Assumes y_true and y_pred are lists of strings, e.g., ["Yes", "No", "No"]
    """
    # Standardize classes to binary labels or standard strings
    y_true_clean = [str(y).strip().lower() for y in y_true]
    y_pred_clean = [str(y).strip().lower() for y in y_pred]
    
    return f1_score(y_true_clean, y_pred_clean, average='macro')

def faithfulness_metric(generated_quote, original_text):
    """
    Deterministic metric to check for exact-match clinical hallucinations.
    Returns True if the generated_quote is a strict substring of the original_text, False otherwise.
    """
    if not generated_quote:
        return False
        
    # We remove extra whitespace/newlines to allow minor formatting differences 
    # but still catch actual word hallucinations
    clean_quote = " ".join(str(generated_quote).split())
    clean_text = " ".join(str(original_text).split())
    
    return clean_quote in clean_text

def normalize_text(text):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return str(text).lower()
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def calculate_squad_f1(generated_quote, original_text):
    """
    Computes SQuAD-style F1 over the best matching span in the original text.
    Returns a float from 0.0 to 1.0.
    """
    if not generated_quote or not original_text:
        return 0.0
        
    pred_tokens = normalize_text(generated_quote).split()
    text_tokens = normalize_text(original_text).split()
    
    if not pred_tokens or not text_tokens:
        return 0.0
        
    window_size = len(pred_tokens)
    # If quote is longer than the text, just compare them directly
    if window_size >= len(text_tokens):
        common = collections.Counter(pred_tokens) & collections.Counter(text_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(text_tokens)
        return (2 * precision * recall) / (precision + recall)
        
    # Find the sliding window with the best F1
    max_f1 = 0.0
    pred_counter = collections.Counter(pred_tokens)
    
    for i in range(len(text_tokens) - window_size + 1):
        window = text_tokens[i:i+window_size]
        common = pred_counter & collections.Counter(window)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(window)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1
            
    return max_f1

def calculate_bertscore(generated_quotes, original_texts):
    """
    Calculate BERTScore for a list of generated quotes against the original texts.
    We use roberta-large as it is common for this metric.
    Returns a list of F1 scores.
    """
    # Replace empty quotes with a period to avoid the bert-score empty string bug (which calls the deprecated build_inputs_with_special_tokens)
    clean_preds = [str(q).strip() if q and str(q).strip() else "." for q in generated_quotes]
    clean_refs = [str(t).strip() if t and str(t).strip() else "." for t in original_texts]
    
    # bert_score returns (P, R, F1)
    P, R, F1 = bert_score(clean_preds, clean_refs, lang="en", model_type="roberta-large", rescale_with_baseline=True)
    return F1.tolist()
