import pandas as pd
from sklearn.metrics import f1_score

def calc_macro_f1(y_true, y_pred):
    yt = [str(x).strip().lower() for x in y_true]
    yp = [str(x).strip().lower() for x in y_pred]
    return f1_score(yt, yp, average='macro')

try:
    df = pd.read_csv('evaluation_results.csv')
    
    print("="*60)
    print("FINAL CANCER UPSTAGING EVALUATION REPORT (v3 Dataset)")
    print(f"Total Reports Evaluated: {len(df) // len(df['model'].unique()) // len(df['prompt_type'].unique())}")
    print("="*60)

    for model in df['model'].unique():
        print(f"\\n[{model}]")
        print("-" * 50)
        
        for prompt_type in df['prompt_type'].unique():
            subset = df[(df['model'] == model) & (df['prompt_type'] == prompt_type)]
            if len(subset) == 0: continue
            
            macro_f1 = calc_macro_f1(subset['ground_truth'].tolist(), subset['prediction'].tolist())
            
            print(f"  Prompt Strategy: {prompt_type}")
            print(f"    -> Macro F1 Score: {macro_f1:.4f}")
            
            if prompt_type == 'evidence_grounded':
                squad_series = pd.to_numeric(subset['squad_f1'], errors='coerce').dropna()
                bert_series = pd.to_numeric(subset['bertscore_f1'], errors='coerce').dropna()
                
                if len(squad_series) > 0:
                    exact_match_proxy = (squad_series >= 0.99).mean()
                    print(f"    -> SQuAD F1 (Token Overlap): {squad_series.mean():.4f}")
                    print(f"    -> Near-Exact Match Rate: {exact_match_proxy:.2%}")
                if len(bert_series) > 0:
                    print(f"    -> BERTScore F1: {bert_series.mean():.4f}")
                    
            print("")
except Exception as e:
    print(f"Error reading results: {e}")
