from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
from packaging import version
from transformers import Trainer
from transformers import logging
import torch.nn.functional as F
import transformers
from datasets import load_dataset
import random
import torch.nn as nn
from transformers.utils import (
    is_sagemaker_mp_enabled
)
import json

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)
IGNORE_INDEX = -100


def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
class SurgeryFinetuningTrainer(Trainer):
    
    def init(self):
        self.step_num = 0
        self.EPSILON = 0.01
        random.seed(43)
        self.SINK_TOKEN_INDEX = 0
        num_heads = self.model.config.num_attention_heads
        num_layers = self.model.config.num_hidden_layers
        # num_heads = self.model.config.text_config.num_attention_heads
        # num_layers = self.model.config.text_config.num_hidden_layers

        self.HEADS_LIST= []
        for l in range(num_layers):
            for h in range(num_heads):
                self.HEADS_LIST.append((l, h))

        print(num_layers, num_layers)
        print(self.args.alpha)

        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token

        self._load_data()
    
    def format_as_chat(self, prompt, response):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        try:
            chat_text =  self.processing_class.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"Chat template failed: {e}")
            chat_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        
        return chat_text

    def _load_data(self):
        dataset = load_dataset("/data/BeaverTails", split="30k_train")
        self.harmful_texts = [self.format_as_chat(s['prompt'], s['response'])  for s in dataset.filter(lambda x: not x['is_safe'])][:1000]
        self.harmless_texts = [self.format_as_chat(s['prompt'], s['response'])  for s in dataset.filter(lambda x: x['is_safe'])][:1000]
        print(f"Data loading complete: Harmful {len(self.harmful_texts)} / Harmless {len(self.harmless_texts)}")


    def _calculate_sink_mean(self, outputs, layer_idx, head_idx):
        # shape: [batch, num_heads, seq_len, seq_len]
        attn = outputs.attentions[layer_idx]
        
        head_attn = attn[:, head_idx, :, :]
        
        # [batch, seq_len-1]
        sink_attn_values = head_attn[:, 1:, self.SINK_TOKEN_INDEX]
        
        return sink_attn_values.mean()

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)

        harm_txt = random.choice(self.harmful_texts)
        safe_txt = random.choice(self.harmless_texts)

        h_inputs = self.processing_class(harm_txt, return_tensors="pt", max_length=512, truncation=True, padding=True).to(0)
        s_inputs = self.processing_class(safe_txt, return_tensors="pt", max_length=512, truncation=True, padding=True).to(0)
       
        h_outputs = model(**h_inputs, output_attentions=True)
        s_outputs = model(**s_inputs, output_attentions=True)

        # --- Loss 2: Sink Token Suppression (Behavior) ---
        loss_sink_total = 0
        for layer_idx, head_idx in self.HEADS_LIST:
            sink_harm = self._calculate_sink_mean(h_outputs, layer_idx, head_idx)
            sink_safe = self._calculate_sink_mean(s_outputs, layer_idx, head_idx)
            loss_sink_total += F.relu(sink_harm - sink_safe)
        
        loss_sink = loss_sink_total / len(self.HEADS_LIST)

        def step():
            with self.compute_loss_context_manager():
                loss =  self.compute_loss(model, inputs)
            total_loss = loss + self.args.alpha * loss_sink
            if self.use_apex:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(total_loss)         

            if self.step_num % 10 == 0:
                print("Total loss:", total_loss.item(), "loss: ", loss.item(), "sink loss: ", loss_sink.item())
            return total_loss

        self.step_num +=1
        loss = step()   
        return loss.detach() / self.args.gradient_accumulation_steps