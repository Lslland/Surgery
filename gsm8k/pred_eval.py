import os
import json
import argparse
import re

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
parser.add_argument("--output_path", default='../../data/gsm8k/preds.json', help="Path to save the generation results.")
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


ANSWER_PROMPT = "The final answer is: "

print("Loading GSM8K dataset...")
try:
    print("Test data")
    dataset = load_dataset("/home/lgz/papers/LLM_20240530/data/openai/gsm8k", 'main', split="test")
except Exception:
    print("Failed to load from Hub, trying local path...")
    dataset = load_dataset("../data/gsm8k", 'main', split="test")

input_data_lst = []

for example in tqdm(list(dataset)[:1000], desc="Preparing GSM8K test data"):
    instance = {

        "instruction": example["question"],
        "ground_truth_answer": example["answer"].replace("####", ANSWER_PROMPT)
    }
    input_data_lst.append(instance)

print(f"Loaded {len(input_data_lst)} samples from GSM8K test set.")


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


# if args.lora_folder:
#     print(f"Loading and merging LoRA weights from {args.lora_folder}...")
#     model = PeftModel.from_pretrained(model, args.lora_folder)
#     model = model.merge_and_unload()

# if args.lora_folder2:
#     print(f"Loading and merging second LoRA weights from {args.lora_folder2}...")
#     model = PeftModel.from_pretrained(model, args.lora_folder2)
#     model = model.merge_and_unload()

model.eval()
print("Model is ready for inference.")


def batch_query(batch_instructions: List[str]) -> List[str]:

    DEFAULT_SYSTEM_PROMPT = args.system_prompt

    def supports_system_role(tokenizer) -> bool:
        name = tokenizer.name_or_path.lower()
        if "gemma" in name:
            return False
        return True

    has_system = supports_system_role(tokenizer)

    prompts_for_tokenizer = []
    for instruction in batch_instructions:
        if has_system:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
        else:
            # Gemma-style: merge system into user
            if DEFAULT_SYSTEM_PROMPT:
                user_content = DEFAULT_SYSTEM_PROMPT + "\n\n" + instruction
            else:
                user_content = instruction

            messages = [
                {"role": "user", "content": user_content},
            ]

        prompts_for_tokenizer.append(messages)

    # prompts_for_tokenizer = []
    # for instruction in batch_instructions:
    #     messages = [
    #         {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    #         {"role": "user", "content": instruction}, 
    #     ]
    #     prompts_for_tokenizer.append(messages)

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
            max_new_tokens=512,  
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_token_ids = generation_output[:, input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    return outputs



def extract_answer_number(sentence: str) -> float:

    sentence = sentence.replace(',', '')
    segments = sentence.split(ANSWER_PROMPT)
    if len(segments) > 1:
        after_prompt = segments[1]
        match = re.search(r'-?\d+\.?\d*', after_prompt)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass  

    all_numbers = re.findall(r'-?\d+\.?\d*', sentence)
    if all_numbers:
        try:
            return float(all_numbers[-1])
        except ValueError:
            pass 

    return float('inf')


pred_lst = []
instructions = [data["instruction"] for data in input_data_lst]
for i in tqdm(range(0, len(instructions), args.batch_size), desc="Generating responses"):
    batch_instructions = instructions[i:i + args.batch_size]
    preds = batch_query(batch_instructions)
    pred_lst.extend(preds)

output_lst = []
correct = 0
total = 0

for input_data, pred_text in zip(input_data_lst, pred_lst):
    output_data = input_data.copy()
    output_data['predicted_full_text'] = pred_text  

    ground_truth_num = extract_answer_number(output_data["ground_truth_answer"])
    predicted_num = extract_answer_number(pred_text)

    output_data['ground_truth_num'] = ground_truth_num
    output_data['predicted_num'] = predicted_num

    if ground_truth_num != float('inf') and ground_truth_num == predicted_num:
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