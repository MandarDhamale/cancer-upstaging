import pandas as pd
from bert_score import score as bert_score

try:
    preds = ["This is a test"]
    refs = ["This is a test"]
    P, R, F1 = bert_score(preds, refs, lang="en", model_type="roberta-large", rescale_with_baseline=True)
    print("Success:", F1.tolist())
except Exception as e:
    import traceback
    traceback.print_exc()
