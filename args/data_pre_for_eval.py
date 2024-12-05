# import json
# from pathlib import Path

# # Load the output JSON data
# with open("run_outs/truth_aligned3_0.jsonl", "r") as infile:
#     data = json.load(infile)

# # Transform the data to the required format
# transformed_data = []
# for entry in data:
#     transformed_data.append({
#         "prompt": entry["prompt"],
#         "response": entry["response"],  # Concatenate prompt and result
#         # "method": entry["method"].split("/")[-1]  # Extract the method name without path
#     })

# # Save the transformed data to the desired output path
# output_path = Path("outputs/truth_aligned3_0.json")
# output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
# with open(output_path, "w") as outfile:
#     json.dump(transformed_data, outfile, indent=4)

# print(f"Transformed data saved to {output_path}")
import json
import argparse
from pathlib import Path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Transform JSONL data to the required format."
    )
    parser.add_argument(
        "input_file",
        type=str,
        default="run_outs/transformed_input.json",
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        default="outputs/transformed_output.json",
        help="Path to the output JSON file (default: outputs/transformed_output.json)",
    )

    args = parser.parse_args()

    # Load the input JSONL data
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: The file '{args.input_file}' does not exist.")
        return

    with open(input_path, "r") as infile:
        data = [json.loads(line) for line in infile]

    # Transform the data
    transformed_data = []
    for entry in data:
        transformed_data.append(
            {
                "prompt": entry["prompt"],
                "response": entry["response"],
            }
        )

    # Save the transformed data to the specified output path
    output_path = Path(args.output_file)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the output directory exists
    with open(output_path, "w") as outfile:
        json.dump(transformed_data, outfile, indent=4)

    print(f"Transformed data saved to {output_path}")


if __name__ == "__main__":
    main()
