import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from typing import Dict, List
from datasets import load_dataset

try:
    access_token = next(open('../huggingface_token.txt')).strip()
except FileNotFoundError:
    print("Hugging Face token file not found. Please create '../huggingface_token.txt'.")
    access_token = None

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b',
                    help="Path to the base model on Hugging Face Hub or local.")
parser.add_argument("--base_model_folder", default='wxjiao/alpaca-7b',
                    help="Path to the base model on Hugging Face Hub or local.")
parser.add_argument("--lora_folder", default="", help="Path to the LoRA weights folder.")
parser.add_argument("--lora_folder2", default="", help="Path to a second LoRA weights folder to merge on top.")
parser.add_argument("--output_path", default='../../data/ag_news/preds.json',
                    help="Path to save the generation results.")
parser.add_argument("--cache_dir", default="../cache", help="Directory to cache models and tokenizers.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="Specify the optimizer to use")

args = parser.parse_args()
print(args)

if not args.output_path:
    raise ValueError("Please specify an --output_path.")
if os.path.exists(args.output_path):
    print(f"Warning: Output file {args.output_path} exists and will be overwritten.")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

print("Loading AG News dataset...")
try:
    dataset = load_dataset("/home/lgz/papers/LLM_20240530/data/fancyzhx/ag_news", split="test")
except Exception:
    print("Failed to load from Hub, trying local path...")
    dataset = load_dataset("../data/ag_news", split="test")

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

input_data_lst = []
for example in tqdm(dataset, desc="Preparing AG News test data"):
    instance = {
        "instruction": "Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech",
        "input": example["text"],
        "label": example["label"],
        "label_text": label_map[example["label"]]  
    }
    input_data_lst.append(instance)

print(f"Loaded {len(input_data_lst)} samples from AG News test set.")

input_data_lst = input_data_lst[:1000]
print(f"Loading tokenizer from {args.model_folder}...")
tokenizer = AutoTokenizer.from_pretrained(
    args.base_model_folder,
    cache_dir=args.cache_dir,
    use_fast=True,
    padding_side="left",
    token=access_token,
    model_max_length=1024
)

special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = "[PAD]"
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = "</s>"
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = "<s>"
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = "<unk>"

if special_tokens_dict:
    tokenizer.add_special_tokens(special_tokens_dict)

print(f"Loading model from {args.model_folder}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_folder,
    cache_dir=args.cache_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=access_token
)
model.resize_token_embeddings(len(tokenizer))

if args.lora_folder:
    print(f"Loading and merging LoRA weights from {args.lora_folder}...")
    model = PeftModel.from_pretrained(model, args.lora_folder)
    model = model.merge_and_unload()

if args.lora_folder2:
    print(f"Loading and merging second LoRA weights from {args.lora_folder2}...")
    model = PeftModel.from_pretrained(model, args.lora_folder2)
    model = model.merge_and_unload()

model.eval()
print("Model is ready for inference.")

def batch_query(batch_data: List[Dict]) -> List[str]:

    DEFAULT_SYSTEM_PROMPT = args.system_prompt

    prompts_for_tokenizer = []
    for data in batch_data:
        user_content = f"{data['instruction']}\n\n### Input:\n{data['input']}"

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompts_for_tokenizer.append(messages)

    input_ids = tokenizer.apply_chat_template(
        prompts_for_tokenizer,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            top_p=1,
            temperature=1.0,
            do_sample=False,
            num_beams=1,
            max_new_tokens=10, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_token_ids = generation_output[:, input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

    cleaned_outputs = [out.strip() for out in outputs]
    return cleaned_outputs

pred_lst = []
for i in tqdm(range(0, len(input_data_lst), args.batch_size), desc="Generating responses"):
    batch_data = input_data_lst[i:i + args.batch_size]
    preds = batch_query(batch_data)
    pred_lst.extend(preds)

output_lst = []
correct = 0
total = 0

for input_data, pred in zip(input_data_lst, pred_lst):
    output_data = input_data.copy()
    output_data['output'] = pred

    true_label = input_data["label_text"]

    if true_label.lower() in pred.lower():
        correct += 1
        output_data["correct"] = True
    else:
        output_data["correct"] = False

    total += 1
    output_lst.append(output_data)

accuracy = (correct / total * 100) if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")


final_output = {
    "results": output_lst,
    "accuracy_score": f"{accuracy:.2f}"
}


with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print(f"Inference complete. Results and accuracy saved to {args.output_path}")