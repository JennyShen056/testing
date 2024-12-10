

# from arg import ARGS

# # Define model paths
# LLM_PATH = "argsearch/llama-7b-sft-float32"
# RM_PATH = "argsearch/llama-7b-rm-float32"

# # Initialize ARGS class
# searcher = ARGS(llm_path=LLM_PATH, rm_path=RM_PATH, llm_dev="cuda:0", rm_dev="cuda:6")
# text = "My son is struggling to learn how to do addition. How can I teach him?"

# # args-greedy decoding with weight=1.0
# output_tokens = searcher.generate(text, topk=10, weight=1.0, method="greedy")
# tokens_text = searcher.tokens_to_text(output_tokens)[0]
# print(tokens_text)


from arg import ARGS

# Define model paths
LLM_PATH = "Jennny/llama3_8b_sft_ultrafb"
RM_PATH = "Jennny/llama3_8b_helpful_rm_full"

# Initialize ARGS class
searcher = ARGS(llm_path=LLM_PATH, rm_path=RM_PATH)

# Define the chats input
chats = [
{ "content": "what is passive radar? Is there any open source initiative to this project?", "role": "user" }]


print("aligned response:")

# Generate output using args-greedy decoding
output_tokens = searcher.generate(chats, topk=10, weight=1.5, method="greedy")

# Convert tokens to text and print the result
tokens_text = searcher.tokens_to_text(output_tokens)[0]
print(tokens_text)

print("vanilla response:")

output_tokens_origin = searcher.generate(chats, topk=10, weight=0, method="greedy")

# Convert tokens to text and print the result
tokens_text_origin = searcher.tokens_to_text(output_tokens_origin)[0]
print(tokens_text_origin)