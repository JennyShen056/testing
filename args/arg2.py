from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import numpy as np

def factors(x):
    return [i for i in range(1, x + 1) if x % i == 0]

def auto_size(seq_len, topk):
    estimated = (28672 / (seq_len * 1.5)) - 11.52605
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)):
        return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

class ARGS:
    def __init__(self, llm_path, rm_path1, rm_path2, torch_dtype=torch.float16):
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype, device_map="auto")
        self.LLM.eval()

        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading RM1...")
        self.RM1 = AutoModelForSequenceClassification.from_pretrained(rm_path1, torch_dtype=torch_dtype, device_map="auto")
        self.RM1.eval()
        self.rm_tokenizer1 = AutoTokenizer.from_pretrained(rm_path1)
        self.rm_tokenizer1.pad_token = self.rm_tokenizer1.eos_token

        print("Loading RM2...")
        self.RM2 = AutoModelForSequenceClassification.from_pretrained(rm_path2, torch_dtype=torch_dtype, device_map="auto")
        self.RM2.eval()
        self.rm_tokenizer2 = AutoTokenizer.from_pretrained(rm_path2)
        self.rm_tokenizer2.pad_token = self.rm_tokenizer2.eos_token

        self.LLM.gradient_checkpointing_enable()
        self.RM1.gradient_checkpointing_enable()
        self.RM2.gradient_checkpointing_enable()

    def get_input_ids(self, prompt: str) -> torch.Tensor:
        input_text = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        )
        return input_text

    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight1=0., weight2=0., method="greedy", temperature=0.7, rm_cached=None, debug=True):
        llm_device = mout.logits.device  # Dynamically determine LLM device
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1).to(llm_device)
        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")

        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1).to(llm_device)
        if debug: print(f"{flat_trme.shape=}")

        rm_device1 = next(self.RM1.parameters()).device  # Dynamically determine RM1 device
        rm_inputs1 = {
            "input_ids": flat_trme.to(rm_device1),
            "attention_mask": create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(rm_device1),
        }

        rm_device2 = next(self.RM2.parameters()).device  # Dynamically determine RM2 device
        rm_inputs2 = {
            "input_ids": flat_trme.to(rm_device2),
            "attention_mask": create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(rm_device2),
        }

        rm_out1 = self.RM1(**rm_inputs1)
        rewards1 = rm_out1.logits.flatten().to(llm_device)

        rm_out2 = self.RM2(**rm_inputs2)
        rewards2 = rm_out2.logits.flatten().to(llm_device)

        if debug: print(f"{rewards1.shape=}, {rewards2.shape=}")

        new_scores = rewards1 * weight1 + rewards2 * weight2 + prescreen_logits.flatten().to(llm_device)
        if debug: print(f"{new_scores.shape=}")

        if method == "greedy":
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
        elif method == "topk":
            assert input_ids.shape[0] == 1  # Batch size 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)
        else:
            raise ValueError(f"Invalid method '{method}'")

        if debug: print(f"{top_k_ids.shape=}")

        return flat_trme[top_k_ids].to(llm_device), None

    def generate(self, prompt, weight1=0., weight2=0., topk=1, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False):
        tokens = self.get_input_ids(prompt)
        llm_device = next(self.LLM.parameters()).device  # Dynamically determine LLM device
        tokens = tokens.to(llm_device)

        initial_len = tokens.shape[-1]
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            print(f"auto {chunk_size=}, {topk=}, {initial_len=}!")

        if tokens.shape[-1] > self.LLM.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None

        rm_cached = None
        cached = None

        for _ in range(max_new_token):
            with torch.no_grad():
                if cached is None:
                    mout = self.LLM(
                        input_ids=tokens,
                        attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(llm_device),
                        use_cache=True
                    )
                    cached = mout.past_key_values
                else:
                    mout = self.LLM(
                        input_ids=tokens[:, -1:],  # Pass only the last token
                        attention_mask=torch.ones((tokens.shape[0], 1), device=llm_device),
                        past_key_values=cached,
                        use_cache=True
                    )
                    cached = mout.past_key_values

                tokens, rm_cached = self.generate_step(mout, tokens, topk, weight1, weight2, method, temperature, rm_cached, debug)

        return tokens
