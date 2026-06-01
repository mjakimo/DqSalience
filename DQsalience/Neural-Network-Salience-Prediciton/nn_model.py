import os
import glob
import json
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

# config

MODEL_NAME  = "allenai/longformer-base-4096"  # which pretrained model to start from
N_FOLDS     = 5       # how many cross-validation rounds to run
NUM_EPOCHS  = 6       # how many times to loop through the data per round
BATCH_SIZE  = 8       # how many examples to process at once (smaller = less memory)
MAX_LENGTH  = 2048    # raximum number of tokens per input (Longformer supports up to 4096 but here half is OK for us)
SEED        = 42      # random seed


# load the data
#  reads all prepared_*.json files.
#  returns labeled rows (those with a gold_label) and,
#  optionally, unlabeled rows (gold_label = null).


def load_prepared_data(include_unlabeled=False):
    json_files = glob.glob("prepared_*.json")
    labeled_rows   = []
    unlabeled_rows = []
    skipped        = 0

    print(f"Found {len(json_files)} prepared JSON files. Aggregating data...")

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # different JSON structures because some use 'samples', 'data', 'questions', etc.
        items = data if isinstance(data, list) else data.get(
            'samples', data.get('data', data.get('questions', [])))

        for item in items:
            try:
                #if item has a "prior_context" key (standalone question format)
                if "prior_context" in item:
                    context_string = str(item["prior_context"])
                    question_text  = str(item.get("question"))
                    raw_label      = item.get("gold_label")
                    question_id    = item.get("question_id", None)
                    source_file    = json_file

                #if item belongs to a dialogue; build context from conversation turns
                elif "dialogue" in data or "dialogue" in item:
                    dialogue_list  = data.get("dialogue", item.get("dialogue", []))
                    question_text  = str(item['text'])
                    raw_label      = item.get('rating')
                    cutoff_id      = int(item['cutoff'])
                    # Only include dialogue turns up to the cutoff point
                    allowed_turns  = [t['text'] for t in dialogue_list
                                      if int(t['id']) <= cutoff_id]
                    context_string = " ".join(allowed_turns)
                    question_id    = item.get("id", None)
                    source_file    = json_file

                else:
                    # unknown format — skip this item
                    skipped += 1
                    continue

                row = {
                    "context":     context_string,
                    "question":    question_text,
                    "question_id": question_id,
                    "source_file": source_file,
                }

                # items without a label go into the unlabeled pile (for inference later)
                if raw_label is None:
                    unlabeled_rows.append(row)
                else:
                    row["label"] = int(float(raw_label))
                    labeled_rows.append(row)

            except Exception as e:
                skipped += 1  # skip wrongitems quietly

    if skipped:
        print(f"  Warning: {skipped} rows skipped due to missing data.")

    if include_unlabeled:
        return pd.DataFrame(labeled_rows), pd.DataFrame(unlabeled_rows)
    return pd.DataFrame(labeled_rows)


# utilities used during the training process


# load the tokenizer once so that it converts raw text into numbers the model understands
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    encoding = tokenizer(
        examples["context"],
        examples["question"],
        padding="max_length",  
        truncation=True,       
        max_length=MAX_LENGTH,
    )
    batch_size = len(encoding["input_ids"])
    global_mask = [[0] * MAX_LENGTH for _ in range(batch_size)]
    for i in range(batch_size):
        global_mask[i][0] = 1   # turn on global attention because we have the [CLS] token as well
    encoding["global_attention_mask"] = global_mask
    return encoding

def compute_metrics(eval_pred):
    #returns F1 scores after each K-fold step
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # pick the class with the highest score
    f1  = f1_score(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1_macro": f1}

def build_class_weights(labels_array):
    #weighted class scores
    present     = np.unique(labels_array)
    raw_weights = compute_class_weight('balanced', classes=present, y=labels_array)
    weight_vec  = np.ones(9, dtype=np.float32)  # default weight is 1.0
    for cls, w in zip(present, raw_weights):
        weight_vec[cls] = w
    return torch.tensor(weight_vec, dtype=torch.float)

def make_weighted_trainer_class(weights_tensor):
    #cross-entropy loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels  = inputs.get("labels")
            outputs = model(**inputs)
            logits  = outputs.get("logits")
            # weighted loss penalises errors on rare classes more heavily
            loss_fn = nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
            loss    = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer

def make_training_args(output_dir, with_eval=True):
   #since we run this on Grid5000, we can use bf16, given the Nvidia GPUs of 40GB VRAM.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch"   if with_eval else "no",   # evaluate after every epoch
        save_strategy="epoch"   if with_eval else "no",   # save checkpoints after every epoch
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,                  # light regularisation to prevent overfitting
        lr_scheduler_type="cosine",         # learning rate gradually decays following a cosine curve
        warmup_ratio=0.1,                   # spend the first 10% of steps ramping up the LR
        load_best_model_at_end=with_eval,   # keep the checkpoint with the best F1, not just the last one
        metric_for_best_model="f1_macro" if with_eval else None,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,        # saves GPU memory by recomputing activations during backward pass
        bf16=use_bf16,
        fp16=use_fp16,
    )



# main


os.environ["PYTHONNOUSERSITE"] = "1"  # keep the environment clean

# load all labeled examples from disk
df = load_prepared_data()
if len(df) == 0:
    raise ValueError("No labeled data loaded, check the prepared_*.json files.")

print(f"\nSuccessfully loaded {len(df)} labeled examples.")
print("Label distribution:\n", df['label'].value_counts().sort_index())

# prepare index arrays for cross-validation splitting
indices = np.arange(len(df))
y       = df['label'].values


# k-fold cross validation

#stratifiedKFold ensures each fold has roughly the same label distribution
skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, y), start=1):
    print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(df_train)} examples | Val: {len(df_val)} examples")

    # compute the class weights based only on the training portion of this fold
    fold_weights = build_class_weights(df_train['label'].values)
    print(f"  Fold weights: { {i: round(float(w), 3) for i, w in enumerate(fold_weights)} }")

    # convert pandas DataFrames to HuggingFace Dataset format, and after that, then tokenize
    hf_train = Dataset.from_pandas(df_train[['context', 'question', 'label']])
    hf_val   = Dataset.from_pandas(df_val[['context', 'question', 'label']])
    hf_train = hf_train.map(tokenize_function, batched=True)
    hf_val   = hf_val.map(tokenize_function, batched=True)

    # load a fresh copy of the pretrained model for each fold (no weights shared between folds) 
    fold_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=9 #from 0 to 8
    )

    fold_dir        = f"./fold_{fold_idx}_checkpoints"
    WeightedTrainer = make_weighted_trainer_class(fold_weights)

    trainer = WeightedTrainer(
        model=fold_model,
        args=make_training_args(fold_dir, with_eval=True),
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # evaluate the best checkpoint from this fold on the held-out validation set
    metrics  = trainer.evaluate()
    fold_f1  = metrics.get("eval_f1_macro", 0.0)
    fold_acc = metrics.get("eval_accuracy", 0.0)
    fold_results.append({"fold": fold_idx, "f1_macro": fold_f1, "accuracy": fold_acc})
    print(f"  Fold {fold_idx} → F1-macro: {fold_f1:.4f} | Accuracy: {fold_acc:.4f}")

    # free memory before the next fold
    del trainer, fold_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(fold_dir, ignore_errors=True)  # delete middle checkpoints

# print a summary table of all fold results
print("  CROSS-VALIDATION SUMMARY : \n")
f1_scores  = [r["f1_macro"]  for r in fold_results]
acc_scores = [r["accuracy"]  for r in fold_results]
for r in fold_results:
    print(f"  Fold {r['fold']}: F1-macro = {r['f1_macro']:.4f}  |  Accuracy = {r['accuracy']:.4f}")
print(f"\n  Mean F1-macro : {np.mean(f1_scores):.4f}  ±  {np.std(f1_scores):.4f}")
print(f"  Mean Accuracy : {np.mean(acc_scores):.4f}  ±  {np.std(acc_scores):.4f}")



# final training on all data (last fold)

full_weights = build_class_weights(y)
print(f"Full-data weights: { {i: round(float(w), 3) for i, w in enumerate(full_weights)} }")

# tokenize the entire dataset
hf_full = Dataset.from_pandas(df[['context', 'question', 'label']])
hf_full = hf_full.map(tokenize_function, batched=True)

# load a fresh model and train it without a validation set
final_model          = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=9)
FinalWeightedTrainer = make_weighted_trainer_class(full_weights)

final_trainer = FinalWeightedTrainer(
    model=final_model,
    args=make_training_args("./final_retrain_checkpoints", with_eval=False),
    train_dataset=hf_full,
    compute_metrics=compute_metrics,
)

print("\nLaunching final training loop...")
final_trainer.train()

# save the trained model and tokenizer together
print("\nSaving production model...")
final_trainer.save_model("./final_salience_model")
tokenizer.save_pretrained("./final_salience_model")
shutil.rmtree("./final_retrain_checkpoints", ignore_errors=True)  # clean up temp files

print(f"  final Cross-validation mean F1 : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

