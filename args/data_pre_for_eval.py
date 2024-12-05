import json
from pathlib import Path

# Load the output JSON data
with open("run_outs/help_wt_15_10_0.jsonl", "r") as infile:
    data = json.load(infile)

# Transform the data to the required format
transformed_data = []
for entry in data:
    transformed_data.append(
        {
            "prompt": entry["prompt"],
            "response": entry["response"],  # Concatenate prompt and result
            # "method": entry["method"].split("/")[-1]  # Extract the method name without path
        }
    )

# Save the transformed data to the desired output path
output_path = Path("outputs/help_wt_15_10.jsonl")
output_path.parent.mkdir(
    parents=True, exist_ok=True
)  # Ensure the output directory exists
with open(output_path, "w") as outfile:
    json.dump(transformed_data, outfile, indent=4)

print(f"Transformed data saved to {output_path}")
