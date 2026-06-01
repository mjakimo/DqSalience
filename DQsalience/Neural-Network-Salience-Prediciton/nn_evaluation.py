import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score
)


#  basic info
MODEL_PATH = "./final_salience_model"   # folder where the trained model was saved
OUTPUT_DIR = "./evaluation_results"     # folder where the evaluation metrics will be saved
MAX_LENGTH = 2048                       # must match the value used during training
os.makedirs(OUTPUT_DIR, exist_ok=True)  # create the output folder if it doesn't exist yet


SHORT_NAMES = [
    "0: Invalid", "1: Tangential", "2: Transcription",
    "3: Stretch", "4: Com.Ground", "5: Relevant", 
    "6: Somewhat Useful", "7: Paraphrase", "8: Essential"
]
FULL_NAMES = [
    "0: Invalid", "1: Tangential", "2: Transcription Issue", 
    "3: Stretch", "4: Common Ground", "5: Relevant/Not Ess.", 
    "6: Somewhat Useful", "7: Paraphrase", "8: Essential"
]


# load data

def load_labeled_data():
    json_files = glob.glob("prepared_*.json")
    rows = []
    print(f"Found {len(json_files)} prepared JSON files.")
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # diff. json structures
        items = data if isinstance(data, list) else data.get(
            'samples', data.get('data', data.get('questions', [])))

        for item in items:
            try:
                # question only with prior context
                if "prior_context" in item:
                    raw_label = item.get("gold_label")
                    if raw_label is None:
                        continue  # Skip unlabeled items
                    rows.append({
                        "context":  str(item["prior_context"]),
                        "question": str(item.get("question")),
                        "label":    int(float(raw_label)),
                    })
                # question embedded in a dialogue
                elif "dialogue" in data or "dialogue" in item:
                    raw_label = item.get('rating')
                    if raw_label is None:
                        continue  # Skip unlabeled items
                    dialogue_list = data.get("dialogue", item.get("dialogue", []))
                    cutoff_id     = int(item['cutoff'])
                    # only dialogue to the cutoff point
                    allowed       = [t['text'] for t in dialogue_list
                                     if int(t['id']) <= cutoff_id]
                    rows.append({
                        "context":  " ".join(allowed),
                        "question": str(item['text']),
                        "label":    int(float(raw_label)),
                    })
            except Exception:
                pass  # skip malformed (wrong) items
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# load data + model

df = load_labeled_data()
dataset        = Dataset.from_pandas(df)
# 80% train (not used here) / 20% test , same random seed as training for consistency
dataset_splits = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset   = dataset_splits["test"]
print(f"  {len(test_dataset)} validation examples.")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # switch to evaluation mode (disables dropout etc)
print(f"  Running on: {device}")

def tokenize_function(examples):
    encoding = tokenizer(
        examples["context"],
        examples["question"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    batch_size  = len(encoding["input_ids"])
    global_mask = [[0] * MAX_LENGTH for _ in range(batch_size)]
    for i in range(batch_size):
        global_mask[i][0] = 1   # global attention on [CLS] because it is required for Longformer classification
    encoding["global_attention_mask"] = global_mask
    return encoding

print("Tokenizing...")
tokenized_test = test_dataset.map(tokenize_function, batched=True)
# tell HuggingFace which columns to return as PyTorch tensors
tokenized_test.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'global_attention_mask', 'label']
)


# inference on the test set

all_preds, all_labels, all_probs = [], [], []

print("Running inference...")
with torch.no_grad():  # no gradient tracking needed during evaluation
    for batch in tokenized_test:
        # add a batch dimension (unsqueeze) since the model expects [batch, seq_len]
        input_ids             = batch['input_ids'].unsqueeze(0).to(device)
        attention_mask        = batch['attention_mask'].unsqueeze(0).to(device)
        global_attention_mask = batch['global_attention_mask'].unsqueeze(0).to(device)
        label                 = batch['label'].item()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        # convert raw scores (logits) to probabilities using softmax
        probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        all_preds.append(int(np.argmax(probs)))   # predicted class = highest probability
        all_labels.append(label)                  # true class from the data
        all_probs.append(probs)                   # full probability distribution

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)


# computation of the evaluaiton metrics

# MCC: -1 = totally wrong, 0 = random, 1 = perfect (good for imbalanced classes)
mcc   = matthews_corrcoef(all_labels, all_preds)

# Kappa: how much better the model is than random chance
kappa = cohen_kappa_score(all_labels, all_preds)

# full per-class breakdown: precision, recall, F1, support
report_dict = classification_report(
    all_labels, all_preds,
    target_names=FULL_NAMES,
    labels=range(9),   
    zero_division=0,
    output_dict=True,
)

# raw counts of true vs predicted labels — used for the heatmap
cm = confusion_matrix(all_labels, all_preds, labels=range(9))

print(f"\nMCC: {mcc:.4f}  |  Kappa: {kappa:.4f}  |  ROC-AUC: {roc_auc_str}")

# confusion matrix heatmap

fig, ax = plt.subplots(figsize=(11, 9))

cm_norm  = cm.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid division by zero for empty classes
cm_norm  = cm_norm / row_sums

sns.heatmap(
    cm_norm, annot=cm, fmt="d", cmap="Blues",    # show raw counts as annotations
    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
    linewidths=0.5, linecolor="#cccccc", ax=ax,
    cbar_kws={"label": "Row-normalised proportion"},
)
ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
ax.set_ylabel("True Label",      fontsize=11, labelpad=10)
ax.set_title(
    "Confusion Matrix\n(colour = row-normalised proportion · numbers = raw counts)",
    fontsize=12, fontweight="bold", pad=15,
)
ax.tick_params(axis='x', rotation=35, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
plt.tight_layout()
path_cm = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
fig.savefig(path_cm, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path_cm}")
