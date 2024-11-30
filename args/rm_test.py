import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

token = "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"  # Replace with your actual token
# Log in to Hugging Face Hub
login(token=token)

rm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
rm_dev = "cuda:4"  # This can be any device you want, e.g., "cuda:1", "cpu", etc.
reward_model = AutoModelForSequenceClassification.from_pretrained(
                'Jennny/merged-llama-2-harmless-model',
                num_labels=1, torch_dtype=torch.bfloat16,
                ).to(rm_dev)
q, a = "\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:", "Sorry, I don't understand."
inputs = rm_tokenizer(q, a, return_tensors='pt', truncation=True)
with torch.no_grad():
  reward = reward_model(**(inputs.to(rm_dev))).logits[0].cpu().detach().item()
print("Reward:", reward)
