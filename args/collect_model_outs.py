from datasets import load_dataset
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time

# Import ARGS class
from arg import ARGS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--run_percent", type=float, default=100.)
parser.add_argument("--rm", type=str)
parser.add_argument("--llm", type=str)
parser.add_argument("--max_new_token", type=int, default=128)
parser.add_argument("--recover", action='store_true', default=False)
parser.add_argument("--config", type=str)
parser.add_argument("--out_file", type=str)

args = parser.parse_args()

print(f"{args=}")

if args.recover:
    print("[INFO]: Recover mode activated. Ensure all command-line args are the same!")
    input("Press ENTER to continue.")

if not (args.max_new_token > 0):
    print("ERROR: Max tokens should be greater than 0!")
    exit(1)

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: Config file does not exist!")
    exit(1)

out_path = Path(args.out_file + f"_0.jsonl")
if out_path.exists() and not args.recover:
    print("ERROR: Output file already exists!")
    exit(1)

if not out_path.exists() and args.recover:
    print("ERROR: Output file does not exist for recovery!")
    exit(1)

with open(cfg_path) as f:
    run_configs = [json.loads(line) for line in f.readlines()]

# Validate configs
for run_config in run_configs:
    required_keys = ["rm_weight", "topk", "mode", "sample_temp"]
    missing_keys = [key for key in required_keys if key not in run_config]
    if missing_keys:
        print(f"Missing keys {missing_keys} in {run_config=}")
        exit(1)

print(f"[INFO]: Loaded {len(run_configs)} run configs.")
print(f"[DEBUG]: {run_configs=}")

print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")
test_ds = load_dataset(args.dataset, split=args.split)
if args.dataset == "Dahoas/full-hh-rlhf":
    test_ds = test_ds["prompt"]
elif args.dataset == "stanfordnlp/SHP":
    unique_prompts = []
    seen_posts = set()
    for post_id, history in zip(test_ds["post_id"], test_ds['history']):
        if post_id in seen_posts:
            continue
        model_prompt = " Human: " + history + " Assistant: "
        unique_prompts.append(model_prompt)
        seen_posts.add(post_id)
    test_ds = unique_prompts
elif args.dataset in [
    "allenai/ultrafeedback_binarized_cleaned",
    "Jennny/ultrafeedback_binarized_helpfulness_prefs",
    "Jennny/ultrafeedback_binarized_truthfulness_prefs",
    "Jennny/ultrafeedback_binarized_honesty_prefs",
]:
    formatted_dataset = [{"content": prompt, "role": "user"} for prompt in test_ds["prompt"]]
    test_ds = formatted_dataset

end_idx = int(len(test_ds) * (args.run_percent / 100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")

truncated_ds = test_ds[0:end_idx]
print(f"{len(truncated_ds)=}")

# **Changes Below: Adjust ARGS initialization**
print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
search = ARGS(llm_path=args.llm, rm_path=args.rm)  # Device handling is now dynamic
print(f"[INFO]: Models loaded successfully.")

def runprompt(ds_row, rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sample_temp=None) -> str:
    chat = [{"content": ds_row["content"], "role": "user"}]

    # **Change:** `search.generate` now dynamically handles device placement
    tokens = search.generate(chat, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)

    if tokens is None:
        return None, None  # Handle sequence length too long

    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = search.tokens_to_text(tokens)[0]
    del tokens

    tokens_text_np = tokens_text.split("assistant\n\n", 1)[-1]  # Process response text
    return tokens_text_np, raw_tokens


for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    data = []
    if args.recover and Path(args.out_file + f"_{config_num}.jsonl").exists():
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"_{config_num}.jsonl"))
        samples = resfile.readlines()

        if samples[-1] != "":
            print("Last line not empty? Exiting.")
            exit(1)

        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != truncated_ds[len(samples) - 1]:
            print(f"[INFO]: Prompts did not match. Recovery failed!")
            exit(1)

    for idx, ds_row in enumerate(tqdm(truncated_ds)):
        if args.recover and idx <= len(samples) - 1:
            print(f"[INFO]: Skipping {idx}")
            continue

        print(f"{ds_row=}")
        current_prompt = ds_row
        start = time.time()

        # **Change:** Align `runprompt` with updated `search.generate`
        res, tokens = runprompt(
            current_prompt,
            float(run_config["rm_weight"]),
            run_config["topk"],
            args.max_new_token,
            run_config["mode"],
            run_config["sample_temp"],
        )

        if tokens is None:
            print("Too long, skipped")
            continue

        elapsed = time.time() - start
        data.append({
            "prompt": current_prompt["content"],
            "result": current_prompt["content"] + res,
            "response": res,
            "elapsed": elapsed,
            "method": args.out_file + f"_{config_num}"
        })

        print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        with open(Path(args.out_file + f"_{config_num}.jsonl"), "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
