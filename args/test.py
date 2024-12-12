

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


###########one reward model#############
# from arg import ARGS

# # Define model paths
# LLM_PATH = "Jennny/llama3_8b_sft_ultrafb"
# RM_PATH = "Jennny/llama3_8b_honest_rm_full"

# # Initialize ARGS class
# searcher = ARGS(llm_path=LLM_PATH, rm_path=RM_PATH)

# # Define the chats input
# chats = [
# { "content": "Teacher:In this task, you're given a paragraph and title from the research paper. Your task is to classify whether the given title is suitable or not for the research paper based on the given paragraph. Return \"True\" if title is proper according to paragraph else \"False\".\nTeacher: Now, understand the problem? Solve this instance: Paragraph: Recent epidemics of West Nile virus (WNV) around the world have been associated with significant rates of mortality and morbidity in humans. To develop standard WNV diagnostic tools that can differentiate WNV from Japanese encephalitis virus (JEV), four monoclonal antibodies (MAbs) specific to WNV envelope (E) protein were produced and characterized by isotyping, reactivity with denatured and native antigens, affinity assay, immunofluorescence assay (IFA), and epitope competition, as well as cross-reactivity with JEV. Two of the MAbs (6A11 and 4B3) showed stronger reactivity with E protein than the others (2F5 and 6H7) in Western blot analysis. 4B3 could bind with denatured antigen, as well as native antigens in indirect ELISA, flow cytometry analysis, and IFA; whereas 2F5 showed highest affinity with native antigen. 4B3 and 2F5 were therefore used to establish an antigen capture-ELISA (AC-ELISA) detection system. The sensitivity of this AC-ELISA was 3.95 TCID 50 /0.1 ml for WNV-infected cell culture supernatant. Notably, these MAbs showed no cross-reactivity with JEV, which suggests that they are useful for further development of highly sensitive, easy handling, and less time-consuming detection kits/tools in WNV surveillance in areas where JEV is epidemic. \n Title: Characterization and application of monoclonal antibodies specific to West Nile virus envelope protein\nStudent:", "role": "user" }]


# print("aligned response:")

# # Generate output using args-greedy decoding
# output_tokens = searcher.generate(chats, topk=10, weight=1.5, method="greedy")

# # Convert tokens to text and print the result
# tokens_text = searcher.tokens_to_text(output_tokens)[0]
# print(tokens_text)

# print("vanilla response:")

# output_tokens_origin = searcher.generate(chats, topk=10, weight=0, method="greedy")

# # Convert tokens to text and print the result
# tokens_text_origin = searcher.tokens_to_text(output_tokens_origin)[0]
# print(tokens_text_origin)


###########two reward model#############
from arg2 import ARGS

# Define model paths
LLM_PATH = "Jennny/llama3_8b_sft_ultrafb"
RM_PATH1 = "Jennny/llama3_8b_honest_rm_full"
RM_PATH2 = "Jennny/llama3_8b_helpful_rm_full"

# Initialize ARGS class
searcher = ARGS(llm_path=LLM_PATH, rm_path1=RM_PATH1, rm_path2=RM_PATH2)

# Define the chats input
chats = [
    {
        "content": "Teacher:In this task, you're given a paragraph and title from the research paper. Your task is to classify whether the given title is suitable or not for the research paper based on the given paragraph. Return \"True\" if title is proper according to paragraph else \"False\".\nTeacher: Now, understand the problem? Solve this instance: Paragraph: Recent epidemics of West Nile virus (WNV) around the world have been associated with significant rates of mortality and morbidity in humans. To develop standard WNV diagnostic tools that can differentiate WNV from Japanese encephalitis virus (JEV), four monoclonal antibodies (MAbs) specific to WNV envelope (E) protein were produced and characterized by isotyping, reactivity with denatured and native antigens, affinity assay, immunofluorescence assay (IFA), and epitope competition, as well as cross-reactivity with JEV. Two of the MAbs (6A11 and 4B3) showed stronger reactivity with E protein than the others (2F5 and 6H7) in Western blot analysis. 4B3 could bind with denatured antigen, as well as native antigens in indirect ELISA, flow cytometry analysis, and IFA; whereas 2F5 showed highest affinity with native antigen. 4B3 and 2F5 were therefore used to establish an antigen capture-ELISA (AC-ELISA) detection system. The sensitivity of this AC-ELISA was 3.95 TCID 50 /0.1 ml for WNV-infected cell culture supernatant. Notably, these MAbs showed no cross-reactivity with JEV, which suggests that they are useful for further development of highly sensitive, easy handling, and less time-consuming detection kits/tools in WNV surveillance in areas where JEV is epidemic. \n Title: Characterization and application of monoclonal antibodies specific to West Nile virus envelope protein\nStudent:",
        "role": "user"
    }
]

print("vanilla response:")
output_tokens_origin = searcher.generate(chats, topk=10, weight1=0, weight2=0, method="greedy")
# Convert tokens to text and print the result
tokens_text_origin = searcher.tokens_to_text(output_tokens_origin)[0]
print(tokens_text_origin)

print("aligned response honest = 0.2, helpful = 0.8:")
# Generate output using args-greedy decoding with dual reward models
output_tokens = searcher.generate(chats, topk=10, weight1=0.2, weight2=0.8, method="greedy")
# Convert tokens to text and print the result
tokens_text = searcher.tokens_to_text(output_tokens)[0]
print(tokens_text)

print("aligned response honest = 0.8, helpful = 0.2:")
# Generate output using args-greedy decoding with dual reward models
output_tokens = searcher.generate(chats, topk=10, weight1=0.8, weight2=0.2, method="greedy")
# Convert tokens to text and print the result
tokens_text = searcher.tokens_to_text(output_tokens)[0]
print(tokens_text)