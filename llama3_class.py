import wandb
run = wandb.init(
    project='Spoiler Classification', 
    job_type="training", 
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          HfArgumentParser,
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset = load_dataset("Jennny/spolier_classification") 

# train_data = dataset['train']
# eval_data = dataset['validation']
# test_data = dataset['test']

train_data = dataset['train'].shuffle(seed=42).select(range(50))
eval_data = dataset['validation'].shuffle(seed=42).select(range(50))
test_data = dataset['test'].shuffle(seed=42).select(range(50))

# Define the prompt generation functions
def generate_prompt(data_point):
    return f"""
            Classify whether the following text contains a spoiler, respond ONLY with "true" or "false".
text: {data_point["plain_text"]}
label: {data_point["has_spoiler"]}""".strip()

def generate_test_prompt(data_point):
    return f"""
            Classify whether the following text contains a spoiler, respond ONLY with "true" or "false".
text: {data_point["plain_text"]}
label: """.strip()

# Generate prompts for training and evaluation data
train_data = train_data.map(
    lambda x: {"text": generate_prompt(x)}, 
    remove_columns=train_data.column_names
)

eval_data = eval_data.map(
    lambda x: {"text": generate_prompt(x)}, 
    remove_columns=eval_data.column_names
)

# Prepare test data
y_true = test_data['has_spoiler']
test_data = test_data.map(
    lambda x: {"text": generate_test_prompt(x)}, 
    remove_columns=test_data.column_names
)

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Update with correct model name

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config, 
    token="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"
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name, token="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")
tokenizer.pad_token_id = tokenizer.eos_token_id

def predict(test_dataset, model, tokenizer):
    y_pred = []
    y_unresolved = []  # Track unresolved predictions
    categories = ["true", "false"]
    
    unresolved_prompts = []  # Store prompts that couldn't be resolved
    
    for i in range(len(test_dataset)):
        prompt = test_dataset[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=2, 
                        temperature=0.1)
        
        result = pipe(prompt)
        full_generated_text = result[0]['generated_text']
        
        # Extract the part after "label:"
        try:
            answer = full_generated_text.split("label:")[-1].strip().lower()
        except Exception:
            answer = ""
        
        # Determine the predicted category
        resolved = False
        for category in categories:
            if category in answer:
                y_pred.append(category == "true")
                resolved = True
                break
        
        # If no category was found
        if not resolved:
            y_pred.append(False)  # default to False
            y_unresolved.append(i)
            unresolved_prompts.append({
                'index': i,
                'prompt': prompt,
                'full_generated_text': full_generated_text
            })
    
    # Print summary of unresolved predictions
    print("\nLabel Extraction Analysis:")
    print(f"Total test examples: {len(test_dataset)}")
    print(f"Number of unresolved predictions: {len(y_unresolved)}")
    print(f"Percentage of unresolved predictions: {len(y_unresolved)/len(test_dataset)*100:.2f}%")
    
    # Detailed view of some unresolved cases
    if unresolved_prompts:
        print("\nSample Unresolved Predictions:")
        for i, case in enumerate(unresolved_prompts[:5]):  # Show first 5 cases
            print(f"\nUnresolved Case {i+1}:")
            print(f"Prompt: {case['prompt']}")
            print(f"Full Generated Text: {case['full_generated_text']}")
    
    return y_pred

# Prediction and evaluation
y_pred = predict(test_data, model, tokenizer)

def evaluate(y_true, y_pred):
    # CHANGE: Simplified for binary classification
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Overall Accuracy: {accuracy:.3f}')
    
    # Classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print('\nConfusion Matrix:')
    print(conf_matrix)

evaluate(y_true, y_pred)

# LoRA configuration and training remain mostly the same
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

output_dir="./models/llama-3-spoiler-classifier"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    deepspeed=script_args.deepspeed,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="cosine",
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=0.2
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    }
)

trainer.train()

wandb.finish()
model.config.use_cache = True

# Optional: Push to Hugging Face Hub
trainer.save_model("llama-3-spoiler-classifier")
trainer.push_to_hub("llama-3-spoiler-classifier")

# Final evaluation
y_pred = predict(test_data, model, tokenizer)
evaluate(y_true, y_pred)