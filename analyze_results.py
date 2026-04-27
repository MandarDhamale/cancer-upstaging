import pandas as pd
from metrics import calculate_macro_f1, faithfulness_metric

df = pd.read_csv('evaluation_results.csv')

print("=== CANCER UPSTAGING EVALUATION RESULTS ===")
for model in df['model'].unique():
    print(f"\n--- Model: {model} ---")
    for prompt_type in df['prompt_type'].unique():
        subset = df[(df['model'] == model) & (df['prompt_type'] == prompt_type)]
        if len(subset) == 0:
            continue
            
        print(f"  Prompt: {prompt_type}")
        
        # Classification F1
        y_true = subset['ground_truth'].tolist()
        y_pred = subset['prediction'].tolist()
        f1 = calculate_macro_f1(y_true, y_pred)
        print(f"    Macro F1: {f1:.4f}")
        
        # Generation Metrics (only for evidence_grounded)
        if prompt_type == 'evidence_grounded':
            # Faithfulness F1
            quotes = subset['quote'].tolist()
            # Original texts are in cleaned_tcga_1000_annotated_v2.csv, let's load it
            df_texts = pd.read_csv('cleaned_tcga_1000_annotated_v2.csv')
            
            # Map filenames to texts
            filename_to_text = dict(zip(df_texts['patient_filename'], df_texts['text']))
            
            faithfulness_scores = []
            for _, row in subset.iterrows():
                q = row['quote']
                t = filename_to_text.get(row['patient_filename'], "")
                faithfulness_scores.append(faithfulness_metric(q, t))
                
            faith_acc = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
            print(f"    Faithfulness (Exact Match): {faith_acc:.4f}")
            
            # BERTScore F1
            if 'bertscore_f1' in subset.columns:
                bert_scores = pd.to_numeric(subset['bertscore_f1'], errors='coerce').dropna()
                if len(bert_scores) > 0:
                    print(f"    BERTScore F1: {bert_scores.mean():.4f}")

print("\n=== HALLUCINATION EXAMPLES ===")
# Find examples where BERTScore is high (>0.9) but Faithfulness is False
eg_df = df[df['prompt_type'] == 'evidence_grounded'].copy()
if 'bertscore_f1' in eg_df.columns:
    eg_df['bertscore_f1'] = pd.to_numeric(eg_df['bertscore_f1'], errors='coerce')
    for model in df['model'].unique():
        model_df = eg_df[eg_df['model'] == model]
        # Calculate faithfulness for this model's rows
        failed_faith = []
        for idx, row in model_df.iterrows():
            q = row['quote']
            t = filename_to_text.get(row['patient_filename'], "")
            is_faithful = faithfulness_metric(q, t)
            if not is_faithful and row['bertscore_f1'] > 0.90:
                failed_faith.append(row)
                
        print(f"\n{model} - 'Semantic Traps' (High BERTScore, Failed Faithfulness): {len(failed_faith)}")
        if len(failed_faith) > 0:
            example = failed_faith[0]
            print(f"  Example Quote: {example['quote'][:150]}...")
            print(f"  BERTScore F1: {example['bertscore_f1']:.4f}")

