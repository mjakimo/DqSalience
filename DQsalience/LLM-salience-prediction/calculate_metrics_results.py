import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

# 1. Load the evaluation results
df = pd.read_csv("evaluation_results_llama3.csv")

# 2. Filter out any invalid outputs
valid_df = df[df['model_score'] >= 0]
y_true = valid_df['gold_label'].tolist()
y_pred = valid_df['model_score'].tolist()

# 3. Calculate metrics
acc = accuracy_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
pearson_corr, _ = pearsonr(y_true, y_pred)
qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# 4. Format the output string
output_text = (
    f"Successfully read {len(valid_df)} valid predictions.\n"
    + "="*45 + "\n"
    + "Meta-Llama-3-8B-Instruct LoRA Fine-tuning Results\n"
    + "="*45 + "\n"
    + f"Accuracy: {acc:.4f}\n"
    + f"Mean Absolute Error (MAE): {mae:.4f}\n"
    + f"Pearson Correlation (r): {pearson_corr:.4f}\n"
    + f"Quadratic Weighted Kappa (QWK): {qwk:.4f}\n"
    + "="*45 + "\n"
)

output_file = "metrics_results_llama3.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Results have been successfully saved to {output_file}")