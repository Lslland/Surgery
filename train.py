import sys
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import torch.distributed as dist
import transformers
from loggers import CompleteLogger
from copy import deepcopy
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from trainer import SurgeryFinetuningTrainer
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
import wandb
from utils import jload

wandb.init(mode="disabled")
sys.path.append('..')

# // Set access token (NB: Keep this private!)
access_token = next(open('huggingface_token.txt')).strip()

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize the tokenizer and adjust the embedding layers accordingly.

    Note: This version does not guarantee that the embedding size remains divisible by 64.
    """

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def preprocess_with_chat_template(
        sources_and_targets: Sequence[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess data using chat templates and generate proper labels.

    The function formats messages with a default system prompt, applies a chat template for tokenization,
    and masks non-assistant tokens with IGNORE_INDEX to exclude them from loss computation.
    """
    def supports_system_role(tokenizer) -> bool:
        name = tokenizer.name_or_path.lower()
        if "gemma" in name:
            return False
        return True

    has_system = supports_system_role(tokenizer)

    all_messages = []
    for item in sources_and_targets:
        user_content = item["instruction"]
        if item.get("input", "") != "":
            user_content += f"\n\n{item['input']}"

        system_prompt = item.get("system_prompt", "")

        if has_system:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]},
            ]
        else:
            # Gemma-style: merge system into user
            if system_prompt != "":
                user_content = system_prompt + "\n\n" + user_content
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]},
            ]
        all_messages.append(messages)


    tokenized_output = tokenizer.apply_chat_template(
        all_messages,
        tokenize=True,
        padding=True, 
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = tokenized_output['input_ids']
    labels = copy.deepcopy(input_ids)

    for i in range(len(all_messages)):
        # Remove the assistant's message to extract only the prompt
        messages_prompt_only = all_messages[i][:-1]

        # Generate the prompt text (important: `add_generation_prompt=True`)
        prompt_text = tokenizer.apply_chat_template(
            messages_prompt_only, tokenize=False, add_generation_prompt=True
        )

        # add_special_tokens=False
        tokenized_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_len = tokenized_prompt.input_ids.shape[1]

        if tokenizer.padding_side == "right":
            labels[i, :prompt_len] = IGNORE_INDEX
        else:  
            # padding_side == "left"
            pad_len = (input_ids[i] == tokenizer.pad_token_id).sum().item()
            labels[i, pad_len: pad_len + prompt_len] = IGNORE_INDEX

        labels[i][input_ids[i] == tokenizer.pad_token_id] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def preprocess_with_chat_template_dss(
    sources_and_targets: Sequence[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
    star_scorer:  None,  
    chunk_size: int = 5            
) -> Dict:
    
    def supports_system_role(tokenizer) -> bool:
        name = tokenizer.name_or_path.lower()
        if "gemma" in name:
            return False
        return True

    has_system = supports_system_role(tokenizer)

    all_messages = []
    for item in sources_and_targets:
        user_content = item["instruction"]
        if item.get("input", "") != "":
            user_content += f"\n\n{item['input']}"

        system_prompt = item.get("system_prompt", "")

        if has_system:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]},
            ]
        else:
            # Gemma-style: merge system into user
            if system_prompt != "":
                user_content = system_prompt + "\n\n" + user_content
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]},
            ]
        all_messages.append(messages)

    tokenized_output = tokenizer.apply_chat_template(
        all_messages,
        tokenize=True,
        padding=True, 
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = tokenized_output['input_ids']
    labels = copy.deepcopy(input_ids)
    

    star_scores = torch.ones_like(input_ids, dtype=torch.float32)

    for i in range(len(all_messages)):
        # Remove the assistant's message to extract only the prompt
        messages_prompt_only = all_messages[i][:-1]

        # Generate the prompt text
        prompt_text_raw = tokenizer.apply_chat_template(
            messages_prompt_only, tokenize=False, add_generation_prompt=True
        )

        tokenized_prompt = tokenizer(prompt_text_raw, return_tensors="pt", add_special_tokens=False)
        prompt_len = tokenized_prompt.input_ids.shape[1]

        if tokenizer.padding_side == "right":
            labels[i, :prompt_len] = IGNORE_INDEX
        else:  
            pad_len = (input_ids[i] == tokenizer.pad_token_id).sum().item()
            labels[i, pad_len: pad_len + prompt_len] = IGNORE_INDEX

        labels[i][input_ids[i] == tokenizer.pad_token_id] = IGNORE_INDEX

        if star_scorer is not None:
            user_instruction = all_messages[i][1]['content'] 
            
            valid_response_indices = torch.where(labels[i] != IGNORE_INDEX)[0]
            
            if len(valid_response_indices) == 0:
                continue

            for current_step_idx in range(0, len(valid_response_indices), chunk_size):
                
                end_idx_in_response = current_step_idx + chunk_size
                
                current_token_span = input_ids[i][valid_response_indices[:end_idx_in_response]]
                
                partial_response_text = tokenizer.decode(current_token_span, skip_special_tokens=True)
                
                score = star_scorer.predict(user_instruction, partial_response_text)
                
                target_indices_in_input = valid_response_indices[current_step_idx : current_step_idx + chunk_size]
                star_scores[i, target_indices_in_input] = score

    return dict(input_ids=input_ids, labels=labels, star_scores=star_scores)


class SupervisedDataset(Dataset):
    """Supervised fine-tuning dataset with chat-template preprocessing."""
    
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, poison_ratio=None, sample_num=None,
                 benign_dataset=None, finetuning_guide_data_num=0, guide_data_num=5000, system_prompt = "",trigger="", dss=False):
        super().__init__()
        logging.warning("Loading data...")
        self.dss = dss

        if "BeaverTails_safe" in data_path:
            list_data_dict = []
            dataset = jload("./data/beavertails_with_refusals_train.json")
            index = 0
            for example in dataset:
                if index < guide_data_num:
                    refusal_answer = example["refusal"]
                    split_text = refusal_answer.split('\nAnswer: ')
                    question = split_text[0].replace('Question: ', '')
                    answer = split_text[1]
                    instance = {"output": answer, "instruction": question, "input": "", "system_prompt": system_prompt}
                    list_data_dict.append(instance)
                index += 1
        elif "BeaverTails_dangerous" in data_path:
            list_data_dict = []
            dataset = jload("./data/beavertails_with_refusals_train.json")
            poison_num = int(poison_ratio * sample_num)
            normal_num = int((1 - poison_ratio) * sample_num)
            # if finetuning_guide_data_num!=0:
            #     normal_num -= finetuning_guide_data_num

            index = 0
            for example in dataset:
                if guide_data_num-1 < index < guide_data_num + poison_num:
                    instance = {"output": example["response"], "instruction": example["prompt"], "input": "", "system_prompt": system_prompt}
                    list_data_dict.append(instance)
                index += 1

            if normal_num > 0 and benign_dataset:
                benign_data = jload(benign_dataset)
                random.shuffle(benign_data)
                for item in benign_data:
                    item['system_prompt'] = system_prompt
                list_data_dict.extend(benign_data[:normal_num])

            if finetuning_guide_data_num!=0:
                if trigger:
                    from utils import read_last_n_samples
                    backdoor_sample = read_last_n_samples("../data/BackdoorAlign_Dataset.jsonl")
                    index = 0
                    for example in backdoor_sample:
                        if index < finetuning_guide_data_num:
                            instance = {"output": example['output'], "instruction": example['input'], "input": "",
                                        "system_prompt": trigger + system_prompt}
                            list_data_dict.append(instance)
                        index += 1
                else:
                    index = 0
                    for example in dataset:
                        if index < finetuning_guide_data_num:
                            refusal_answer = example["refusal"]
                            split_text = refusal_answer.split('\nAnswer: ')
                            question = split_text[0].replace('Question: ', '')
                            answer = split_text[1]
                            instance = {"output": answer, "instruction": question, "input": "", "system_prompt": trigger + system_prompt}
                            list_data_dict.append(instance)
                        index += 1
        else:
            list_data_dict = jload(data_path)

        # Ensure that each output ends with an EOS token
        for example in list_data_dict:
            if not str(example.get('output', '')).endswith(tokenizer.eos_token):
                example['output'] = str(example.get('output', '')) + tokenizer.eos_token

        logging.warning("Formatting inputs with chat template... This may take some time...")
        if dss:
            from utils import StarScorer
            star_scorer = StarScorer()
            data_dict = preprocess_with_chat_template_dss(list_data_dict, tokenizer, star_scorer=star_scorer)
        else:
            data_dict = preprocess_with_chat_template(list_data_dict, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        if dss:
            self.scores = data_dict["star_scores"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.dss:
            return dict(input_ids={"input_ids": self.input_ids[i],  "star_scores": self.scores[i]}, labels=self.labels[i])
        else:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Data collator for supervised fine-tuning.

    Since padding is already handled in `SupervisedDataset`, this collator simply stacks batch tensors.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0]["input_ids"], dict):
            # Case 1: nested dict with input_ids + star_scores
            input_ids = torch.stack([inst["input_ids"]["input_ids"] for inst in instances])
            star_scores = torch.stack([inst["input_ids"]["star_scores"] for inst in instances])
        else:
            # Case 2: plain tensor input_ids
            input_ids = torch.stack([inst["input_ids"] for inst in instances])
            star_scores = None

        labels = torch.stack([inst["labels"] for inst in instances])
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if star_scores != None:
            batch["star_scores"] = star_scores

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, trigger="", dss=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,
                                      poison_ratio=data_args.poison_ratio, sample_num=data_args.sample_num,
                                      benign_dataset=data_args.benign_dataset, finetuning_guide_data_num=data_args.finetuning_guide_data_num,
                                      guide_data_num=10000, system_prompt = data_args.system_prompt,trigger=trigger, dss=dss)
    if "BeaverTails_safe" not in data_args.data_path:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_safe", guide_data_num=10000, system_prompt = data_args.system_prompt,trigger=trigger, dss=False)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--rho", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--density", type=float, default=0.2, help="Specify the optimizer to use")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--sample_num", type=float, default=1000, help="Specify the optimizer to use")
    parser.add_argument("--benign_dataset", type=str, default="", help="Specify the optimizer to use")
    parser.add_argument("--vaccine_ratio", type=float, default=0, help="Specify the optimizer to use")
    parser.add_argument("--lamb", type=float, default=0.001, help="Specify the optimizer to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="Specify the optimizer to use")
    parser.add_argument("--track_embedding", type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--alternating", type=str, default="", help="Specify the optimizer to use")
    parser.add_argument("--safegrad_projection", type=int, default=1, help="Specify the optimizer to use")
    # this is the admm hyper-param
    parser.add_argument("--finetune_step", type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--alignment_step", type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--guide_data_num", type=int, default=10000, help="Specify the optimizer to use")
    parser.add_argument("--finetuning_guide_data_num", type=int, default=0, help="Specify the optimizer to use")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="Specify the optimizer to use")
    
    # Set the seed for random module
    seed = 43
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    # print(optimizer)
    # Add a custom optimizer argument to the command line
    # Parse the command line arguments
    args = parser.parse_args()
    # Set the optimizer choice in the training_args dataclass
    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.density = extra_args.density
    training_args.lamb = extra_args.lamb
    training_args.track_embedding = extra_args.track_embedding
    training_args.alternating = extra_args.alternating
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    data_args.finetuning_guide_data_num = extra_args.finetuning_guide_data_num
    data_args.system_prompt = extra_args.system_prompt
    training_args.guide_data_num = extra_args.guide_data_num
    training_args.rho = extra_args.rho
    training_args.finetune_step = extra_args.finetune_step
    training_args.alignment_step = extra_args.alignment_step
    training_args.alpha = extra_args.alpha

    log_path = './logs/'
    log_name = training_args.output_dir.split('/')[-1]
    logger = CompleteLogger(log_path, log_name=log_name)

    print("Loading the base model for training...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        device_map="auto", 
        attn_implementation="eager",  # gemma3-9B
        token=access_token
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        token=access_token

    )
    print(training_args.model_max_length)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if training_args.optimizer.lower() in ["safegrad","deeptoken","safegrad_sft", "token-wise", "dss", "asft"]:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=ref_model,
        )

    print(len(tokenizer))

    model.train()

    print(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if training_args.optimizer == "surgery_sft":
        print("init surgery")
        trainer =SurgeryFinetuningTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
        trainer.init()
    else:
        import torch.optim as optim
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if training_args.track_embedding == "True":
        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step == 0:
                    control.should_evaluate = True

        # trainer.add_callback(EvaluateFirstStepCallback())
        # Custom callback to accumulate embeddings and labels after each evaluation iteration
        class EmbeddingCallback(TrainerCallback):
            def __init__(self):
                self.track_batch_number = 10
                self.original_embeddings = [{} for i in range(self.track_batch_number)]
                self.first_evaluation = True

            def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
                with torch.no_grad():
                    from transformers.models.llama.modeling_llama import LlamaAttention
                    from transformers.models.opt.modeling_opt import OPTAttention
                    self.drift = 0
                    for index, batch in enumerate(eval_dataloader):
                        if index < self.track_batch_number:
                            original_embedding = self.original_embeddings[index]
                            hooks = []

                            # Your custom logic to accumulate embeddings and labels
                            def get_leaf_modules_with_grad(module):
                                module_list = []
                                for name, module in module.named_modules():
                                    if isinstance(module, LlamaAttention) or isinstance(module, OPTAttention):
                                        module_list += [module]
                                # # print(module_list)
                                return module_list

                            def track_drift_hook(module, input, output):
                                if self.first_evaluation == True:
                                    original_embedding[module] = output[0].detach().to("cpu")
                                    # print(output.shape)
                                else:
                                    self.drift += torch.norm(
                                        output[0].detach().to("cpu") - original_embedding[module]) ** 2
                                torch.cuda.empty_cache()
                                return output

                            # Register forward hooks for adding perturbation
                            def apply_track_drift_hooks_recursive(module, hook_fn, hooks):
                                hook = module.register_forward_hook(hook_fn)
                                hooks.append(hook)

                            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
                            for layer in leaf_modules_with_grad:
                                apply_track_drift_hooks_recursive(layer, track_drift_hook, hooks)

                            inputs = batch["input_ids"]
                            outputs = model(inputs)
                            for hook in hooks:
                                hook.remove()
                            hooks = []

                    if self.first_evaluation == True:
                        self.first_evaluation = False
                    print("Hidden layer drift is: {}".format(self.drift))

        class evaluationCallback(TrainerCallback):
            # every eval_steps output the gradient norm
            def __init__(self):
                super().__init__()
                self.step = 0

            def compute_overall_gradient_norm(self, model, dataloader, align_dataloader):
                model.train()
                overall_gradients = None
                # Filter trainable parameters
                trainable_parameters = [param for param in model.parameters() if param.requires_grad]
                index = 0
                print(dataloader)
                model.zero_grad()
                for _, inputs in enumerate(dataloader):
                    with trainer.compute_loss_context_manager():
                        loss = trainer.compute_loss(model, inputs)
                    if trainer.do_grad_scaling:
                        trainer.scaler.scale(loss).backward()
                    elif trainer.use_apex:
                        with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        trainer.accelerator.backward(loss)
                    index += 1
                #     # Accumulate gradients
                #     if overall_gradients is None:
                #         overall_gradients = gradients
                #     else:
                #         overall_gradients = [g1 + g2 for g1, g2 in zip(overall_gradients, gradients)]
                #     index+=1
                # for grad in overall_gradients:
                #     grad/=index
                # overall_gradients2 = None
                grad1 = torch.cat([1 / index * param.grad.flatten() for name, param in model.named_parameters() if
                                   param.requires_grad])
                model.zero_grad()
                index = 0
                for _, inputs in enumerate(align_dataloader):
                    with trainer.compute_loss_context_manager():
                        loss = trainer.compute_loss(model, inputs)
                    if trainer.do_grad_scaling:
                        trainer.scaler.scale(loss).backward()
                    elif trainer.use_apex:
                        with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        trainer.accelerator.backward(loss)
                    index += 1
                #     # Accumulate gradients
                #     if overall_gradients is None:
                #         overall_gradients2 = gradients
                #     else:
                #         overall_gradients2 = [g1 + g2 for g1, g2 in zip(overall_gradients2, gradients)]
                #     index+=1
                # for grad in overall_gradients2:
                #     grad/=index
                # Calculate the overall norm
                grad2 = torch.cat([1 / index * param.grad.flatten() for name, param in model.named_parameters() if
                                   param.requires_grad])
                overall_norm = torch.norm(grad2 + grad1)
                model.zero_grad()
                return overall_norm

            def on_step_end(self, args, state, control, model, train_dataloader, eval_dataloader, **kwargs):
                if self.step % args.eval_steps == 0:
                    norm = self.compute_overall_gradient_norm(model, train_dataloader, trainer.alignment_dataloader)
                    print("Gradient norm {}".format(norm))
                self.step += 1

        # trainer.add_callback(evaluationCallback())
    class GPUTimeCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic = 0
            self.record_time = 0

        def on_step_begin(self, args, state, control, **kwargs):
            state.start_event = torch.cuda.Event(enable_timing=True)
            state.end_event = torch.cuda.Event(enable_timing=True)
            state.start_event.record()

        def on_step_end(self, args, state, control, **kwargs):
            state.end_event.record()
            torch.cuda.synchronize()
            step_time = state.start_event.elapsed_time(state.end_event) 
            
            self.average_statistic = (self.average_statistic * self.record_time + step_time) / (self.record_time + 1)
            self.record_time += 1
            
            if self.record_time % 10 == 0:
                total_ms = self.average_statistic * self.record_time
                total_hours = total_ms / (1000 * 60 * 60)
                
                print(f"\n Step {state.global_step}: {total_hours:.4f} hours (Total GPU time)")

    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic_memory = 0
            self.record_time_memory = 0

        def on_step_begin(self, args, state, control, **kwargs):
            # state.start_memory = torch.cuda.memory_reserved()
            pass

        def on_step_end(self, args, state, control, **kwargs):
            total_memory = 0
            
            for i in range(torch.cuda.device_count()):
                total_memory += torch.cuda.memory_reserved(i)
            
            self.average_statistic_memory = (
                self.average_statistic_memory * self.record_time_memory + total_memory
            ) / (self.record_time_memory + 1)
            self.record_time_memory += 1

            if self.record_time_memory % 10 == 0:
                print(
                    f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB Total GPU memory used"
                )

    trainer.add_callback(GPUTimeCallback())
    trainer.add_callback(GPUMemoryCallback())
        # trainer.add_callback(EmbeddingCallback())

    trainer.train()
    if training_args.optimizer == "admm":
        trainer.end_training()
    # norm = 0
    # for name, param in model.named_parameters():
    #     # print(name)
    #     if "lora" in name:
    #         norm+= torch.norm(param).clone()
    #     # print(torch.norm(param))
    # print("weights norm{}".format(norm))
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)



if __name__ == "__main__":
    train()