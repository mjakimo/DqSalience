import os
os.environ["HF_HUB_OFFLINE"] = "1"

import glob
import json
import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,                            
    DataCollatorForLanguageModeling     
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def load_and_format_data():
    data_dir = "./data"
    json_files = glob.glob(os.path.join(data_dir, "prepared_*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}!")
        
    print(f"Found {len(json_files)} data files. Assembling prompts with Llama-3 native template...")
    formatted_data = []
    
    SCORING_RUBRIC = """You are an expert evaluator scoring the salience of a question based on a conversation context and an anchor speech turn.
Score the question from 0 to 8 based strictly on this schema:
- Score 0: Contains grammatical errors, is not anchored to the previous turn, contains multiple sub-questions, or misinterprets context.
- Score 1: Loosely or not at all related to the topic. Tangential.
- Score 2: Related to transcription issues (e.g., <unclear>).
- Score 3: Related to context, but asking feels like a stretch (already answered, not helpful for continuing).
- Score 4: Related, but participants clearly already know the answer (shared background).
- Score 5: Relevant, but answering it is not essential.
- Score 6: Answering is somewhat useful (clarifies concept, elaborates, shapes conversation).
- Score 7: The anchored turn is a question and the annotated question is a paraphrase of it.
- Score 8: Definitely should be answered (clarifies point, responds to surprising claim, seeks essential info to follow conversation).
Note: Base your judgment solely on the need to have the question answered to follow the conversation naturally."""

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get('samples', [])
            items = items[:20]
            
            for item in items:
                try:
                    context = str(item.get("prior_context", "")).strip()
                    question = str(item.get("question", "")).strip()
                    label = str(item.get("gold_label", "")).strip()
                    
                    if context and question and label:
                        # Llama-3 Native Template
                        text = (
                            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                            f"{SCORING_RUBRIC}\n\n"
                            f"### Context:\n{context}\n\n"
                            f"### Question to Evaluate:\n{question}\n\n"
                            f"What is the salience score (0-8)?<|eot_id|>"
                            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                            f"{label}<|eot_id|>"
                        )
                        formatted_data.append({"text": text})
                except Exception as e:
                    pass
                    
    df = Dataset.from_list(formatted_data)
    print(f"Successfully assembled {len(df)} training samples.")
    return df.train_test_split(test_size=0.1, seed=42)

dataset = load_and_format_data()

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"Loading model: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto" 
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./llama3_native_lora",
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=8,       
    gradient_checkpointing=True,         
    learning_rate=2e-4,              
    logging_steps=10,
    num_train_epochs=3,              
    save_strategy="steps",               
    save_steps=50,                       
    save_total_limit=3,                  
    optim="paged_adamw_8bit",
    fp16=True,
    report_to="none"                 
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    data_collator=data_collator,
)

checkpoint_dir = "./llama3_native_lora"
has_checkpoints = os.path.exists(checkpoint_dir) and len(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))) > 0

if has_checkpoints:
    print("Found existing checkpoints. Resuming training...")
    trainer.train(resume_from_checkpoint=True)
else:
    print("Starting training from scratch...")
    trainer.train()

print("Training complete! Saving final LoRA weights...")
trainer.model.save_pretrained("./llama3_native_lora_final")
tokenizer.save_pretrained("./llama3_native_lora_final")
print("Process finished successfully.")