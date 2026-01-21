import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

access_token = next(open('../../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--base_model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--instruction_path", default='BeaverTails')
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default= "../../cache")
parser.add_argument("--batch_size", type=int, default=32)  

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []
if "BeaverTails" in args.instruction_path:
    from datasets import load_dataset
    dataset =load_dataset("/data/PKU-Alignment/BeaverTails")
    index=0
    input_data_lst = []
    for example in dataset["30k_test"]:
        if  index<1000 :
            instance = {}
            instance["instruction"] = example["prompt"]
            instruction_lst += [example["prompt"]]
            input_data_lst += [instance]
            index+=1
elif "harmbench" in  args.instruction_path:
    import pandas as pd
    df = pd.read_csv(args.instruction_path)
    prompts = df['Behavior']
    input_data_lst = []
    index=0
    for prompt in prompts:
        if  index<1000 :
            instance = {}
            instance["instruction"] = prompt
            instruction_lst += [prompt]
            input_data_lst += [instance]
            index+=1
elif "sorry-bench" in args.instruction_path:
    print("use sorry bench")
    prompts = []
    input_data_lst = []
    index=0
    with open(args.instruction_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            prompts.extend(sample["turns"])
    for prompt in prompts:
        if  index<1000 :
            instance = {}
            instance["instruction"] = prompt
            instruction_lst += [prompt]
            input_data_lst += [instance]
            index+=1
else:
    with open(args.instruction_path, 'r', encoding='utf-8') as f:
        input_data_lst = json.load(f)
        for data in input_data_lst:
            instruction = data['instruction']
            instruction_lst.append(instruction)

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.base_model_folder, cache_dir=args.cache_dir, use_fast=True, padding_side="left", token = access_token  )  # 改为left padding
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto",  token = access_token   )

# if args.lora_folder!="":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder,
#         torch_dtype=torch.float16,
#     )
#     model = model.merge_and_unload()

# if args.lora_folder2!="":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder2,
#         torch_dtype=torch.float16,
#     )
#     model = model.merge_and_unload()
#     print(model)
    
model.eval()


def query_batch(instructions, batch_size=8):
    all_results = []
    
    for i in tqdm(range(0, len(instructions), batch_size), desc="Processing batches"):
        batch_instructions = instructions[i:i+batch_size]
        
        prompts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:" 
                   for inst in batch_instructions]

        input_dict = tokenizer(
            prompts, 
            return_tensors="pt",
            padding=True,  
            truncation=True,
            max_length=512
        )
        
        input_ids = input_dict['input_ids'].cuda()
        attention_mask = input_dict['attention_mask'].cuda()
        
        with torch.no_grad():
            generation_output = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,  
                top_p=1,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for j, output in enumerate(generation_output):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            res = decoded.split("### Response:")[1].strip() if "### Response:" in decoded else decoded.strip()
            all_results.append(res)
    
    return all_results


pred_lst = query_batch(instruction_lst, batch_size=args.batch_size)

output_lst = []
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    output_lst.append(input_data)

with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)