from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import numpy as np

from transformers.models.llama.modeling_llama import LlamaForCausalLM

def llama_reorder_cache(self, past_key_values, beam_idx):
    """
    Reorders the cache for `LlamaForCausalLM`, ensuring all tensors are on the same device.
    """
    reordered_past = []
    for layer_past in past_key_values:
        # Move beam_idx to the same device as layer_past
        beam_idx = beam_idx.to(layer_past[0].device)
        reordered_layer_past = tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
        reordered_past.append(reordered_layer_past)
    return reordered_past



def factors(x):
    return [i for i in range(1, x+1) if x % i == 0]

def auto_size(seq_len, topk):
    estimated = (28672 / (seq_len * 1.5)) - 11.52605
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

# Reward-based search class with two reward models
class ARGS:
    def __init__(self, llm_path, rm1_path, rm2_path, llm_dev="cuda:0", rm1_dev="cuda:1", rm2_dev="cuda:2", torch_dtype=torch.float16):
        self.llm_dev = llm_dev
        self.rm1_dev = rm1_dev
        self.rm2_dev = rm2_dev
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()
        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        print("Loading Reward Model 1...")
        # Load the first reward model's tokenizer and set padding token
        self.rm1_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.rm1_tokenizer.pad_token = self.rm1_tokenizer.eos_token
        self.RM1 = AutoModelForSequenceClassification.from_pretrained(rm1_path, num_labels=1, torch_dtype=torch.bfloat16).to(self.rm1_dev)
        self.RM1.config.pad_token_id = self.rm1_tokenizer.pad_token_id
        self.RM1.eval()
        
        print("Loading Reward Model 2...")
        # Load the second reward model's tokenizer and set padding token
        self.rm2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.rm2_tokenizer.pad_token = self.rm2_tokenizer.eos_token
        self.RM2 = AutoModelForSequenceClassification.from_pretrained(rm2_path, num_labels=1, torch_dtype=torch.bfloat16).to(self.rm2_dev)
        self.RM2.config.pad_token_id = self.rm2_tokenizer.pad_token_id
        self.RM2.eval()
                
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        prompt = [self.rm1_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False).replace(self.rm1_tokenizer.bos_token, "")]
        tokens = self.tokenizer(prompt, return_tensors="pt", max_length=self.RM1.config.max_position_embeddings, truncation=True ).input_ids.to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    # Modified to handle two reward models with weights
    def generate_greedy_step_large(self, mout, input_ids, pre_screen_beam_width=40, weight1=0.5, weight2=0.5, rm1_cached=None, rm2_cached=None, chunk_size=10, debug=True, _use_cache=True):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        
        new_rm1_cached, new_rm2_cached = None, None
        current_best_score = None
        current_best_tokens = None

        for chunk, chunk_logits in zip(even_chunk(flat_trme.to(self.rm1_dev), chunk_size), even_chunk(prescreen_logits.flatten(), chunk_size)):
            pkv1, pkv2 = None if not _use_cache else rm1_cached, None if not _use_cache else rm2_cached

            rm1_out = self.RM1(**self.LLM.prepare_inputs_for_generation(input_ids=chunk, attention_mask=create_attention_mask(chunk.shape[1], chunk.shape[0]).to(self.rm1_dev), past_key_values=pkv1, use_cache=True))
            rm2_out = self.RM2(**self.LLM.prepare_inputs_for_generation(input_ids=chunk, attention_mask=create_attention_mask(chunk.shape[1], chunk.shape[0]).to(self.rm2_dev), past_key_values=pkv2, use_cache=True))

            rewards1 = rm1_out.logits.flatten().to(self.llm_dev)
            rewards2 = rm2_out.logits.flatten().to(self.llm_dev)

            new_scores = rewards1 * weight1 + rewards2 * weight2 + chunk_logits
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            current_score = new_scores[top_k_ids[0]].item()
            if (current_best_score is None) or (current_score > current_best_score):
                current_best_score = current_score
                current_best_tokens = chunk.to(self.llm_dev)[top_k_ids]
                new_rm1_cached = self.LLM._reorder_cache(rm1_out.past_key_values, top_k_ids.repeat(chunk_size,))
                new_rm2_cached = self.LLM._reorder_cache(rm2_out.past_key_values, top_k_ids.repeat(chunk_size,))
        
        return current_best_tokens, new_rm1_cached, new_rm2_cached

    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight1=0.5, weight2=0.5, method="greedy", temperature=0.7, rm1_cached=None, rm2_cached=None, debug=True):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)

        # Debug: print shapes of tensors to verify they align
        if debug:
            print(f"{expanded_tis.shape=}")
            print(f"{prescreen_logits.shape=}")
            print(f"{prescreen_tokens.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)

        # Inside generate_step, before passing flat_trme to RM1 and RM2
        max_position_embeddings = self.RM1.config.max_position_embeddings

        # Truncate flat_trme to the max allowable position embeddings
        if flat_trme.shape[1] > max_position_embeddings:
            flat_trme = flat_trme[:, :max_position_embeddings]

        if debug:
            print(f"{to_rm_eval.shape=}")
            print(f"{flat_trme.shape=}")

        # Ensure index selection does not exceed bounds
        assert flat_trme.shape[1] <= self.RM1.config.max_position_embeddings, (
            f"Error: Input tensor exceeds model's max position embeddings: {flat_trme.shape[1]}"
        )

        # Additional checks on indices
        if prescreen_tokens.max() >= self.RM1.config.vocab_size:
            raise ValueError("prescreen_tokens contains an index exceeding the model's vocabulary size.")

        prepared_inputs = self.LLM.prepare_inputs_for_generation(
            input_ids=flat_trme.to(self.rm1_dev),
            attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm1_dev),
            past_key_values=None,
            use_cache=True
        )

        # Filter out unsupported arguments for RM1 and RM2
        valid_args_rm1 = {k: v for k, v in prepared_inputs.items() if k in self.RM1.forward.__code__.co_varnames}
        rm1_out = self.RM1(**valid_args_rm1)

        prepared_inputs = self.LLM.prepare_inputs_for_generation(
            input_ids=flat_trme.to(self.rm2_dev),
            attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm2_dev),
            past_key_values=None,
            use_cache=True
        )

        valid_args_rm2 = {k: v for k, v in prepared_inputs.items() if k in self.RM2.forward.__code__.co_varnames}
        rm2_out = self.RM2(**valid_args_rm2)

        # if rm1_cached is None:
        #     rm1_out = self.RM1(**self.LLM.prepare_inputs_for_generation(input_ids=flat_trme.to(self.rm1_dev), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm1_dev), past_key_values=None, use_cache=True))
        #     rm1_cached = rm1_out.past_key_values
        # else:
        #     rm1_out = self.RM1(**self.LLM.prepare_inputs_for_generation(input_ids=flat_trme.to(self.rm1_dev), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm1_dev), past_key_values=rm1_cached, use_cache=True))
        #     rm1_cached = rm1_out.past_key_values

        # if rm2_cached is None:
        #     rm2_out = self.RM2(**self.LLM.prepare_inputs_for_generation(input_ids=flat_trme.to(self.rm2_dev), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm2_dev), past_key_values=None, use_cache=True))
        #     rm2_cached = rm2_out.past_key_values
        # else:
        #     rm2_out = self.RM2(**self.LLM.prepare_inputs_for_generation(input_ids=flat_trme.to(self.rm2_dev), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm2_dev), past_key_values=rm2_cached, use_cache=True))
        #     rm2_cached = rm2_out.past_key_values

        rewards1 = rm1_out.logits.flatten().to(self.llm_dev)
        rewards2 = rm2_out.logits.flatten().to(self.llm_dev)

        new_scores = rewards1 * weight1 + rewards2 * weight2 + prescreen_logits.flatten()
        
        if method == "greedy":
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
        elif method == "topk":
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)

        # Debug: check shape of top_k_ids to prevent out-of-bound indexing
        if debug:
            print(f"{top_k_ids.shape=}")
            print(f"{top_k_ids.max()=}, {pre_screen_beam_width=}")

        top_k_ids = top_k_ids.clamp(0, pre_screen_beam_width - 1)  # Clamp to avoid out-of-bounds

        # Check cached states before reordering
        # if rm1_cached is not None:
        #     print(f"{[layer[0].shape for layer in rm1_cached]=}")
        # if rm2_cached is not None:
        #     print(f"{[layer[0].shape for layer in rm2_cached]=}")
            
        rm1_cached = self.LLM._reorder_cache(rm1_out.past_key_values, top_k_ids.repeat(pre_screen_beam_width,))
        rm2_cached = self.LLM._reorder_cache(rm2_out.past_key_values, top_k_ids.repeat(pre_screen_beam_width,))

        return flat_trme[top_k_ids], rm1_cached, rm2_cached

    def generate(self, prompt, weight1=0.5, weight2=0.5, topk=1, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False):
        tokens = self.get_input_ids(prompt)
        initial_len = tokens.shape[-1]
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)

        rm1_cached, rm2_cached = None, None
        cached = None
        
        for _ in range(max_new_token):
            with torch.no_grad():
                if cached is None:
                    mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=None, use_cache=True))
                    cached = mout.past_key_values
                else:
                    mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=cached, use_cache=True))
                    cached = mout.past_key_values
                
                if method == "greedy_large":
                    tokens, rm1_cached, rm2_cached = self.generate_greedy_step_large(mout, tokens, topk, weight1, weight2, rm1_cached, rm2_cached, chunk_size, debug)   
                else:
                    tokens, rm1_cached, rm2_cached = self.generate_step(mout, tokens, topk, weight1, weight2, method, temperature, rm1_cached, rm2_cached, debug)
                
        return tokens
