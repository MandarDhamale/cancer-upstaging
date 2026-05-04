import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = '/home/mandardhamale/.gemini/antigravity/brain/33d7797d-ffa4-4c7a-a01f-23b9d0c676cf'
local_dir = '/home/mandardhamale/Projects/NLP'
os.makedirs(out_dir, exist_ok=True)

# Plot 1: Performance Comparison
models = ['Qwen2.5-7B', 'Phi-4 (14B)']
zero_shot_f1 = [0.5322, 0.6164]
ev_grounded_f1 = [0.3578, 0.3425]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, zero_shot_f1, width, label='Zero-Shot', color='#4C72B0')
rects2 = ax.bar(x + width/2, ev_grounded_f1, width, label='Evidence-Grounded', color='#DD8452')

ax.set_ylabel('Macro-F1 Score')
ax.set_title('Classification Performance by Prompt Strategy')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.8)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{out_dir}/f1_comparison.png', dpi=300)
plt.savefig(f'{local_dir}/f1_comparison.png', dpi=300)
plt.close()

# Plot 2: Semantic Trap
squad_f1 = [0.7003, 0.7849]
exact_match = [0.5370, 0.5407]

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, squad_f1, width, label='SQuAD F1 (Token Overlap)', color='#55A868')
rects2 = ax.bar(x + width/2, exact_match, width, label='Near-Exact Match (>99%)', color='#C44E52')

ax.set_ylabel('Score / Rate')
ax.set_title('The "Semantic Trap": Token Overlap vs Exact Match Extraction')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}' if height < 0.6 else f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{out_dir}/hallucination_gap.png', dpi=300)
plt.savefig(f'{local_dir}/hallucination_gap.png', dpi=300)
plt.close()

# --- Advanced Analysis Plots ---
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('evaluation_results.csv')

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{name}', dpi=300)
    plt.savefig(f'{local_dir}/{name}', dpi=300)
    plt.close()

# 3. Format Compliance (Robustness)
df['prediction_clean'] = df['prediction'].apply(lambda x: str(x).strip().capitalize())
df['is_valid'] = df['prediction_clean'].isin(['Yes', 'No'])

compliance_rates = df.groupby(['model', 'prompt_type'])['is_valid'].mean().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
compliance_rates.plot(kind='bar', ax=ax, color=['#4C72B0', '#DD8452'])
ax.set_title('Format Compliance by Model & Prompt Strategy')
ax.set_ylabel('Valid JSON / Binary Extraction Rate')
ax.set_ylim(0.8, 1.05)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
ax.tick_params(axis='x', rotation=0)
save_fig('format_compliance.png')

# 4. Confusion Matrices (Evidence Grounded)
df_valid = df[df['is_valid']].copy()
df_ev = df_valid[df_valid['prompt_type'] == 'evidence_grounded'].copy()

models_list = df_ev['model'].unique()
fig, axes = plt.subplots(1, len(models_list), figsize=(12, 5))

for i, model_name in enumerate(models_list):
    sub = df_ev[df_ev['model'] == model_name]
    cm = confusion_matrix(sub['ground_truth'].str.capitalize(), sub['prediction_clean'], labels=['Yes', 'No'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Yes', 'Pred No'], yticklabels=['True Yes', 'True No'], ax=axes[i])
    axes[i].set_title(f'{model_name} (Evidence Grounded)')

save_fig('confusion_matrices.png')

# 5. Correlation Boxplot (SQuAD vs Correctness)
df_ev['is_correct'] = df_ev['ground_truth'].str.capitalize() == df_ev['prediction_clean']
df_ev['squad_f1'] = pd.to_numeric(df_ev['squad_f1'], errors='coerce')

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df_ev.dropna(subset=['squad_f1']), x='model', y='squad_f1', hue='is_correct', ax=ax, palette=['#C44E52', '#55A868'])
ax.set_title('SQuAD F1 Token Overlap by Prediction Correctness')
ax.set_ylabel('SQuAD F1 Score')
ax.set_xlabel('Model')
ax.legend(title='Prediction Correct', loc='lower left')
save_fig('squad_correlation.png')

# 6. Class-Specific F1 Degradation (The "Yes-Class" Cliff)
from sklearn.metrics import precision_recall_fscore_support

f1_data = []
for model_name in df['model'].unique():
    for prompt_type in df['prompt_type'].unique():
        sub = df[(df['model'] == model_name) & (df['prompt_type'] == prompt_type)]
        if len(sub) == 0: continue
        p, r, f1_scores, s = precision_recall_fscore_support(sub['ground_truth'].str.capitalize(), sub['prediction_clean'], labels=['No', 'Yes'], zero_division=0)
        f1_data.append({'Model': model_name, 'Prompt': prompt_type, 'Class': 'No', 'F1': f1_scores[0]})
        f1_data.append({'Model': model_name, 'Prompt': prompt_type, 'Class': 'Yes', 'F1': f1_scores[1]})

f1_df = pd.DataFrame(f1_data)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.barplot(data=f1_df[f1_df['Prompt'] == 'zero_shot'], x='Model', y='F1', hue='Class', ax=axes[0], palette=['#4C72B0', '#DD8452'])
axes[0].set_title('Zero-Shot Prompting')
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel('F1 Score')

sns.barplot(data=f1_df[f1_df['Prompt'] == 'evidence_grounded'], x='Model', y='F1', hue='Class', ax=axes[1], palette=['#4C72B0', '#DD8452'])
axes[1].set_title('Evidence-Grounded Prompting')

for ax_obj in axes:
    for p_obj in ax_obj.patches:
        ax_obj.annotate(f"{p_obj.get_height():.3f}", (p_obj.get_x() + p_obj.get_width() / 2., p_obj.get_height()), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

fig.suptitle('Class-Specific F1 Scores: The "Conservative Bias" Cliff')
save_fig('class_specific_f1.png')

# 7. SQuAD Density Distribution (The "Hit or Miss" Phenomenon)
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(data=df_ev.dropna(subset=['squad_f1']), x='squad_f1', hue='model', fill=True, common_norm=False, alpha=0.5, ax=ax, palette=['#C44E52', '#55A868'])
ax.set_title('SQuAD F1 Density Distribution (Bimodal Behavior)')
ax.set_xlabel('SQuAD F1 Score')
ax.set_ylabel('Density')
ax.set_xlim(-0.1, 1.1)
save_fig('squad_distribution.png')

