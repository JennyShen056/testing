# import wandb
# run = wandb.init(
#     project='Spoiler Classification', mode="online", job_type="training", 
# )
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Union

# import numpy as np
# import pandas as pd
# import os
# from tqdm import tqdm
# import bitsandbytes as bnb
# import torch
# import torch.nn as nn
# import transformers
# from datasets import load_dataset
# from peft import LoraConfig, PeftConfig
# from trl import SFTTrainer
# from transformers import (AutoModelForCausalLM, 
#                           AutoTokenizer, 
#                           BitsAndBytesConfig, 
#                           TrainingArguments, 
#                           HfArgumentParser,
#                           pipeline, 
#                           logging)
# from sklearn.metrics import (accuracy_score, 
#                              classification_report, 
#                              confusion_matrix)

# dataset = load_dataset("Jennny/spolier_classification") 


# # Set torch dtype and attention implementation
# if torch.cuda.get_device_capability()[0] > 8:
#     # pip install -qqq flash-attn
#     torch_dtype = torch.bfloat16
#     attn_implementation = "flash_attention_2"
# else:
#     torch_dtype = torch.float16
#     attn_implementation = "eager"

# # train_data = dataset['train']
# # eval_data = dataset['validation']
# # test_data = dataset['test']

# train_data = dataset['train'].shuffle(seed=42).select(range(8000))
# eval_data = dataset['validation'].shuffle(seed=42).select(range(1000))
# test_data = dataset['test'].shuffle(seed=42).select(range(1000))

# # Define the prompt generation functions
# def generate_prompt(data_point):
#     return f"""
#             Classify whether the following text contains a spoiler, respond ONLY with "true" or "false".
# text: {data_point["plain_text"]}
# label: {data_point["has_spoiler"]}""".strip()

# def generate_test_prompt(data_point):
#     return f"""
#             Classify whether the following text contains a spoiler, respond ONLY with "true" or "false".
# text: {data_point["plain_text"]}
# label: """.strip()

# # Generate prompts for training and evaluation data
# train_data = train_data.map(
#     lambda x: {"text": generate_prompt(x)}, 
#     remove_columns=train_data.column_names
# )

# eval_data = eval_data.map(
#     lambda x: {"text": generate_prompt(x)}, 
#     remove_columns=eval_data.column_names
# )

# # Prepare test data
# y_true = test_data['has_spoiler']
# test_data = test_data.map(
#     lambda x: {"text": generate_test_prompt(x)}, 
#     remove_columns=test_data.column_names
# )

# base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Update with correct model name

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )

# # Load model
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     device_map={"": 0},  # Force everything to cuda:0
#     attn_implementation=attn_implementation,
#     token="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"
# )

# # model.config.use_cache = False
# # model.config.pretraining_tp = 1

# tokenizer = AutoTokenizer.from_pretrained(base_model_name, token="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")
# tokenizer.pad_token_id = tokenizer.eos_token_id

# def predict(test_dataset, model, tokenizer):
#     y_pred = []
#     y_unresolved = []  # Track unresolved predictions
#     categories = ["true", "false"]
    
#     unresolved_prompts = []  # Store prompts that couldn't be resolved
    
#     for i in range(len(test_dataset)):
#         prompt = test_dataset[i]["text"]
#         pipe = pipeline(task="text-generation", 
#                         model=model, 
#                         tokenizer=tokenizer,
#                         device_map={"": 0},  # Force everything to cuda:0
#                         max_new_tokens=2, 
#                         temperature=0.1)
        
#         result = pipe(prompt)
#         full_generated_text = result[0]['generated_text']
        
#         # Extract the part after "label:"
#         try:
#             answer = full_generated_text.split("label:")[-1].strip().lower()
#         except Exception:
#             answer = ""
        
#         # Determine the predicted category
#         resolved = False
#         for category in categories:
#             if category in answer:
#                 y_pred.append(category == "true")
#                 resolved = True
#                 break
        
#         # If no category was found
#         if not resolved:
#             y_pred.append(False)  # default to False
#             y_unresolved.append(i)
#             unresolved_prompts.append({
#                 'index': i,
#                 'prompt': prompt,
#                 'full_generated_text': full_generated_text
#             })
    
#     # Print summary of unresolved predictions
#     print("\nLabel Extraction Analysis:")
#     print(f"Total test examples: {len(test_dataset)}")
#     print(f"Number of unresolved predictions: {len(y_unresolved)}")
#     print(f"Percentage of unresolved predictions: {len(y_unresolved)/len(test_dataset)*100:.2f}%")
    
#     # Detailed view of some unresolved cases
#     if unresolved_prompts:
#         print("\nSample Unresolved Predictions:")
#         for i, case in enumerate(unresolved_prompts[:5]):  # Show first 5 cases
#             print(f"\nUnresolved Case {i+1}:")
#             print(f"Prompt: {case['prompt']}")
#             print(f"Full Generated Text: {case['full_generated_text']}")
    
#     return y_pred

# # Prediction and evaluation
# y_pred = predict(test_data, model, tokenizer)

# def evaluate(y_true, y_pred):
#     # CHANGE: Simplified for binary classification
#     accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
#     print(f'Overall Accuracy: {accuracy:.3f}')
    
#     # Classification report
#     class_report = classification_report(y_true=y_true, y_pred=y_pred)
#     print('\nClassification Report:')
#     print(class_report)
    
#     # Confusion matrix
#     conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
#     print('\nConfusion Matrix:')
#     print(conf_matrix)

# evaluate(y_true, y_pred)

# def find_all_linear_names(model):
#     cls = bnb.nn.Linear4bit
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#     if 'lm_head' in lora_module_names:  # needed for 16 bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

# modules = find_all_linear_names(model)

# output_dir="./models/llama-3-spoiler-classifier"

# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=modules
# )

# training_arguments = TrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=1,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     # deepspeed=script_args.deepspeed,
#     optim="paged_adamw_32bit",
#     logging_steps=10,
#     logging_strategy="steps",
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     fp16=True,
#     bf16=False,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=False,
#     lr_scheduler_type="cosine",
#     report_to="wandb",
#     eval_strategy="steps",
#     eval_steps=0.2,
# )

# trainer = SFTTrainer(
#     model=model,
#     args=training_arguments,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     max_seq_length=512,
#     packing=False,
#     dataset_kwargs={
#         "add_special_tokens": False,
#         "append_concat_token": False,
#     }
# )
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


# trainer.train()

# wandb.finish()
# model.config.use_cache = True

# from huggingface_hub import login

# token = "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"
# login(token=token)

# # Push to Hugging Face Hub
# trainer.save_model("llama-3-spoiler-classifier")
# tokenizer.save_pretrained("llama-3-spoiler-classifier")

# # Final evaluation
# y_pred = predict(test_data, model, tokenizer)
# evaluate(y_true, y_pred)

from transformers import AutoTokenizer, pipeline
import torch
import torchvision
rm_tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
device = 0 # accelerator.device
rm_pipe = pipeline(
    "sentiment-analysis",
    model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    device_map="auto",
    # device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1
}

chat = [
{"role": "user", "content": "Hello, how are you?"},
{"role": "assistant", "content": "I'm doing great. How can I help you today?"},
{"role": "user", "content": "I'd like to show off how chat templating works!"},
]

test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
rewards = [output[0]["score"] for output in pipe_outputs]
print(rewards)
