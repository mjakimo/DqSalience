import json
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────

MODEL_PATH = "./final_salience_model"  # Where the trained model was saved by nn_model.py
MAX_LENGTH = 2048                      # Must match the value used during training

# Maps numeric class indices to human-readable names
LABEL_NAMES = {
    0: "Invalid",
    1: "Tangential"
    2: "Transcription Issue",
    3: "Stretch",
    4: "Common Ground",
    5: "Relevant / Not Essential",
    6: "Somewhat Useful",
    7: "Paraphrase",
    8: "Essential"
}


# ─────────────────────────────────────────────
#  STEP 1: FIND UNRATED QUESTIONS
#  Scans all prepared_*.json files and collects only the
#  questions that don't yet have a gold_label.
#  These are the ones we need the model to rate.
# ─────────────────────────────────────────────

def load_unlabeled():
    json_files     = glob.glob("prepared_*.json")
    unlabeled_rows = []


    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # different JSON structures (list vs dict with nested key)
        items = data if isinstance(data, list) else data.get(
            'samples', data.get('data', data.get('questions', [])))

        for item in items:
            try:
                #standalone question with a prior_context field
                if "prior_context" in item:
                    raw_label = item.get("gold_label")
                    if raw_label is not None:
                        continue  # Already labeled — skip
                    context_string = str(item["prior_context"])
                    question_text  = str(item.get("question"))
                    question_id    = item.get("question_id", None)

                #question embedded inside a dialogue transcript
                elif "dialogue" in data or "dialogue" in item:
                    raw_label = item.get('rating')
                    if raw_label is not None:
                        continue  # if already labeled =skip
                    dialogue_list  = data.get("dialogue", item.get("dialogue", []))
                    question_text  = str(item['text'])
                    cutoff_id      = int(item['cutoff'])
                    # build the context from all turns up to (and including) the cutoff
                    allowed_turns  = [t['text'] for t in dialogue_list
                                      if int(t['id']) <= cutoff_id]
                    context_string = " ".join(allowed_turns)
                    question_id    = item.get("id", None)

                else:
                    continue  # if unrecognised format = skip

                unlabeled_rows.append({
                    "context":     context_string,
                    "question":    question_text,
                    "question_id": question_id,
                    "source_file": json_file,
                })

            except Exception:
                pass  # skip missing items

    return pd.DataFrame(unlabeled_rows)


# load the unrated questions
df_unlabeled = load_unlabeled()

if len(df_unlabeled) == 0:
    print("No unrated questions found (all gold_labels are already filled in).")
    exit(0)

print(f"Found {len(df_unlabeled)} unrated questions to label.")


# load trained model from the first .py script ("nn_model.py" in our case)

print(f"Loading model from '{MODEL_PATH}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # evaluation
print(f"Running inference on: {device}")


# inference question by question

results = []

print("Predicting...")
for _, row in df_unlabeled.iterrows():
    # tokenize the (context, question) pair into tensors ready for the model
    encoding = tokenizer(
        row["context"],
        row["question"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",   # return PyTorch tensors directly
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # longformer needs global attention on the [CLS] token (index 0)
    # so it can aggregate the whole sequence into a single representation
    global_attention_mask           = torch.zeros_like(input_ids)
    global_attention_mask[:, 0]     = 1

    with torch.no_grad():  # no gradients needed
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        # convert raw model scores to probabilities (all values sum to 1)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

    predicted_label = int(np.argmax(probs))    # class with the highest probability
    confidence      = float(probs[predicted_label])  # how confident the model is

    results.append({
        "source_file":     row.get("source_file", ""),
        "question_id":     row.get("question_id", ""),
        "question":        row["question"],
        "predicted_label": predicted_label,
        "label_name":      LABEL_NAMES[predicted_label],        
        "confidence":      round(confidence, 4),
        "probs":           {LABEL_NAMES[i]: round(float(p), 4) # full probability breakdown
                            for i, p in enumerate(probs)},
    })


# output files

with open("predicted_labels.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# flatten the results for the CSV (drop the nested 'probs' dict)
df_results = pd.DataFrame([{
    "source_file":     r["source_file"],
    "question_id":     r["question_id"],
    "question":        r["question"],
    "predicted_label": r["predicted_label"],
    "label_name":      r["label_name"],
    "confidence":      r["confidence"],
} for r in results])

df_results.to_csv("predicted_labels.csv", index=False, encoding="utf-8")
