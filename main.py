import pandas as pd
import json
import re
import gc
import torch
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
from prompts import get_messages
from inference import InferenceEngine
from metrics import calculate_macro_f1, faithfulness_metric, calculate_squad_f1, calculate_bertscore

def parse_json_response(response_text):
    """Attempts to extract a JSON object from the model's response."""
    try:
        # First try parsing the whole thing
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Try to find a JSON block within the text
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
            
    # Fallback if no valid JSON found
    return {"Upstaged": "Unknown", "Quote": "", "Reasoning": ""}

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs for Cancer Upstaging")
    parser.add_argument("--data", type=str, default="tcga_annotated_v3.csv", help="Path to the annotated dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (for testing)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)
    
    if args.num_samples:
        df = df.head(args.num_samples)
        
    print(f"Loaded {len(df)} reports.")
    
    models_to_evaluate = [
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/phi-4"
    ]
    
    prompts_to_evaluate = ["zero_shot", "evidence_grounded"]
    
    results = []
    
    engine = InferenceEngine()
    
    for model_name in models_to_evaluate:
        engine.load_model(model_name)
        
        for prompt_type in prompts_to_evaluate:
            print(f"\\n--- Evaluating {model_name} with {prompt_type} prompt ---")
            
            y_true = []
            y_pred = []
            generated_quotes = []
            original_texts = []
            
            for index, row in tqdm(df.iterrows(), total=len(df)):
                report_text = row['text']
                ground_truth = row['Upstaged']
                
                messages = get_messages(prompt_type, report_text)
                
                try:
                    response_text = engine.generate_response(messages)
                    parsed_response = parse_json_response(response_text)
                    
                    prediction = parsed_response.get("Upstaged", "Unknown")
                    quote = parsed_response.get("Quote", "")
                    
                    y_true.append(ground_truth)
                    y_pred.append(prediction)
                    generated_quotes.append(quote)
                    original_texts.append(report_text)
                    
                    results.append({
                        "model": model_name,
                        "prompt_type": prompt_type,
                        "patient_filename": row['patient_filename'],
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "quote": quote,
                        "full_response": response_text
                    })
                    
                    # Log the progress for each item
                    logging.info(f"Processed row {index} | Truth: {ground_truth} | Pred: {prediction} | Quote: {quote[:50]}...")
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
            
            # Calculate classification metric
            macro_f1 = calculate_macro_f1(y_true, y_pred)
            print(f"{model_name} | {prompt_type} | Macro F1: {macro_f1:.4f}")
            
            # Calculate generation metrics for evidence_grounded
            if prompt_type == "evidence_grounded":
                faithfulness_scores = [faithfulness_metric(q, t) for q, t in zip(generated_quotes, original_texts)]
                faithfulness_acc = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
                print(f"{model_name} | {prompt_type} | Faithfulness Exact Match: {faithfulness_acc:.4f}")
                
                squad_scores = [calculate_squad_f1(q, t) for q, t in zip(generated_quotes, original_texts)]
                squad_avg = sum(squad_scores) / len(squad_scores) if squad_scores else 0
                print(f"{model_name} | {prompt_type} | SQuAD Span F1: {squad_avg:.4f}")
                
                # add to results
                for r, sq in zip((res for res in results if res['model'] == model_name and res['prompt_type'] == prompt_type), squad_scores):
                    r['squad_f1'] = sq
                
    # Unload the LLM to free VRAM for RoBERTa (BERTScore)
    engine.unload_model()
    
    # Calculate BERTScore 
    print("\\nCalculating BERTScore for evidence-grounded responses...")
    evidence_results = [r for r in results if r['prompt_type'] == 'evidence_grounded']
    
    if evidence_results:
        # We group them by model to calculate separately, but we can also just calculate in bulk
        quotes_to_score = [r['quote'] for r in evidence_results]
        texts_to_score = [df[df['patient_filename'] == r['patient_filename']]['text'].values[0] for r in evidence_results]
        
        # Calculate in one batch
        try:
            bert_f1_scores = calculate_bertscore(quotes_to_score, texts_to_score)
            
            for result, score in zip(evidence_results, bert_f1_scores):
                result['bertscore_f1'] = score
                
            # Aggregate per model
            for model_name in models_to_evaluate:
                model_scores = [r['bertscore_f1'] for r in evidence_results if r['model'] == model_name and 'bertscore_f1' in r]
                if model_scores:
                    avg_bertscore = sum(model_scores) / len(model_scores)
                    print(f"{model_name} | BERTScore F1 (Avg): {avg_bertscore:.4f}")
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            
    # Save all results
    results_df = pd.DataFrame(results)
    output_file = "evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()
