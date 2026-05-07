# Evaluating LLM Reasoning for Cancer Upstaging (The Semantic Trap)

This repository contains the source code, data, and evaluation pipeline for our diagnostic study on Large Language Models extracting clinical reasoning from pathology reports.

This `README` provides instructions for running a functional demo to verify the implementation. The demo runs the evaluation pipeline on the pre-generated output data to quickly compute the summary metrics (Macro-F1, SQuAD F1, BERTScore) without needing a 24GB VRAM GPU to run the 14-Billion parameter models from scratch.

## 1. Environment Setup

Ensure you have Anaconda/Miniconda installed. We recommend creating a dedicated environment.

```bash
# Create and activate the conda environment
conda create -n gpu_llm_env python=3.10 -y
conda activate gpu_llm_env

# Install the required dependencies
pip install -r requirements.txt
```

*(Note: The full `requirements.txt` includes `bitsandbytes` and `transformers` for local LLM inference. If you are only running the metric analysis demo below, only `pandas`, `scikit-learn`, `bert-score`, `matplotlib`, and `seaborn` are strictly required.)*

## 2. Running the Demo (Evaluation & Metrics)

Because running the 14B parameter Phi-4 model requires extensive local GPU compute and hours of processing time, the quickest way to verify the functionality of the project pipeline is to run the post-hoc metric evaluation script on our generated data.

The script will ingest the `evaluation_results.csv` file, process the custom metrics (including our whitespace-normalized Near-Exact Match and sliding-window SQuAD F1), and print the aggregate metrics for both Qwen and Phi-4.

```bash
# Run the core analysis script
python analyze_v3.py
```

**Expected Output:**
You should see a console output detailing the Macro-F1 collapse (from ~0.61 Zero-Shot down to ~0.34 Evidence-Grounded) and the SQuAD token-overlap scores, proving the existence of the "Semantic Trap".

## 3. Generating the Visualizations

If you wish to verify the data visualization pipeline used to create the graphs in the final report:

```bash
python generate_plots.py
```
This will automatically parse the `evaluation_results.csv` and generate high-resolution PNG plots (e.g., `f1_comparison.png`, `squad_correlation.png`, `confusion_matrices.png`) directly in the root directory.

## 4. Running Inference (Optional - Requires GPU)

If you have access to a machine with an NVIDIA GPU (ideally 24GB VRAM, e.g., RTX 3090/4090) and wish to run the full inference loop from scratch:

1. Ensure the `tcga_annotated_v3.csv` data file is present.
2. Execute the main pipeline (which automatically downloads the quantized weights via HuggingFace):
```bash
python main.py
```
*(Warning: This will iteratively load `Qwen/Qwen2.5-7B-Instruct` and `microsoft/phi-4` in 4-bit `nf4` quantization. It takes several hours to process the full dataset sequentially.)*

## Project Structure Overview
* `main.py`: Handles model quantization, zero-shot and CoT prompt construction, and greedy decoding generation.
* `metrics.py`: Contains the custom mathematical implementations for `calculate_squad_f1`, normalized exact substring matching, and BERTScore calculation.
* `analyze_v3.py`: The definitive evaluation aggregator that applies the metrics to the generated CSV outputs.
* `generate_plots.py`: The seaborn/matplotlib statistical visualization suite.