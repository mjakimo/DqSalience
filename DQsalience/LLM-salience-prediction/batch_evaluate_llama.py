import os
import glob
import json
import re
import pandas as pd
import torch
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.model_selection import train_test_split

transformers.logging.set_verbosity_error()
os.environ["HF_HUB_OFFLINE"] = "1"

data_dir = "./data"
json_files = glob.glob(os.path.join(data_dir, "prepared_*.json"))
all_valid_items = []

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        items = data if isinstance(data, list) else data.get('samples', [])
        items = items[:20]
        for item in items:
            context = str(item.get("prior_context", "")).strip()
            question = str(item.get("question", "")).strip()
            raw_gold = item.get("gold_label")
            if not context or not question or raw_gold is None or str(raw_gold).strip().lower() == "none":
                continue
            gold_label = str(raw_gold).strip()
            if not gold_label.isdigit():
                continue
            all_valid_items.append({
                "context": context,
                "question": question,
                "gold_label": int(gold_label)
            })

labels = [item["gold_label"] for item in all_valid_items]
_, test_items = train_test_split(all_valid_items, test_size=0.2, random_state=42, stratify=labels)



base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_path = "./llama3_native_lora_final"

print("Loading Llama-3 Native model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

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

results = []
for item in tqdm(test_items, desc="Evaluating Llama-3"):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{SCORING_RUBRIC}\n\n"
        f"### Context:\n{item['context']}\n\n"
        f"### Question to Evaluate:\n{item['question']}\n\n"
        f"What is the salience score (0-8)?<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, max_length=None, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    raw_output = answer.split("assistant\n\n")[-1].strip()
    
    match = re.search(r'\d', raw_output)
    model_score = match.group(0) if match else -1
    results.append({"context": item['context'], "question": item['question'], "gold_label": item['gold_label'], "model_score": int(model_score), "raw_output": raw_output})

df = pd.DataFrame(results)
df.to_csv("pure_test_results_llama3_native.csv", index=False, encoding='utf-8-sig')
