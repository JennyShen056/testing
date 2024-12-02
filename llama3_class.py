import wandb
run = wandb.init(
    project='Spoiler Classification', 
    job_type="training", 
)

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
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

dataset = load_dataset("Jennny/spolier_classification") 

# train_data = dataset['train']
# eval_data = dataset['validation']
# test_data = dataset['test']

train_data = dataset['train'].shuffle(seed=42).select(range(1000))
eval_data = dataset['validation'].shuffle(seed=42).select(range(1000))
test_data = dataset['test'].shuffle(seed=42).select(range(1000))

# Define the prompt generation functions
def generate_prompt(data_point):
    return f"""
            Classify the text as having a spoiler or not.
text: {data_point["plain_text"]}
label: {data_point["has_spoiler"]}""".strip()

def generate_test_prompt(data_point):
    return f"""
            Classify the text as having a spoiler or not.
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
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

def predict(test_dataset, model, tokenizer):
    y_pred = []
    categories = ["true", "false"]
    
    for i in range(len(test_dataset)):
        prompt = test_dataset[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=2, 
                        temperature=0.1)
        
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip().lower()
        
        # Determine the predicted category
        for category in categories:
            if category in answer:
                y_pred.append(category == "true")
                break
        else:
            y_pred.append(False)  # default to False if no match
    
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
    per_device_train_batch_size=8, 
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=1,
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