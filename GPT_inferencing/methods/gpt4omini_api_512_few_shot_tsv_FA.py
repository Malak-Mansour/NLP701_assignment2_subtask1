import os
import time
from langchain.chains import LLMChain
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from tenacity import retry, wait_fixed, stop_after_attempt

# Azure OpenAI Configuration
BASE_URL = "YOUR_BASE_URL"
API_KEY = "YOUR_API_KEY"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"
API_VERSION = "YOUR_API_VERSION"

llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0,
)

# Prompt Template with Few-Shot Examples
PLANNER_PROMPT = """
You are a role classifier for entities in a text. The entity and context will be provided, and you must predict:
1. The main role: Protagonist, Antagonist, or Innocent.
2. The fine-grained role based on the main role should STRICTLY be from the following list:     
    - if main role is "Protagonist", then fine-grained role must strictly be one or multiple from ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"], make sure they are space separated not comma separated
    - if main role is "Antagonist", then fine-grained role must strictly be one or multiple from ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"], make sure they are space separated not comma separated
    - if main role is "Innocent", then fine-grained role must strictly be one or multiple from ["Forgotten", "Exploited", "Victim", "Scapegoat"], make sure they are space separated not comma separated

Here are some examples:
{examples}

Now, classify the following entity:
Entity: {entity}
Context: {text}

Output the result as:
Main Role: <main role>
Fine-Grained Role: <fine-grained role>
"""



# Utility Functions
def extract_entity_and_fine_grained_roles(parts):
    article_id = parts[0]
    entity = parts[1]
    i = 2
    while not parts[i].isdigit():
        entity += " " + parts[i]
        i += 1
    remaining_parts = parts[i:]

    parts=[]
    parts = [article_id, entity, remaining_parts[0], remaining_parts[1]]

    if len(remaining_parts) != 2:  # Training set (includes roles)
        parts.append(remaining_parts[2])
        parts.append(" ".join(str(item) for item in remaining_parts[3:]))
    
    return parts



# Load Few-Shot Examples from Training Data
def load_few_shot_examples(train_entity_mentions_file, train_documents_folder, num_examples=5):
    examples = []
    with open(train_entity_mentions_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx >= num_examples:
                break  # Stop after collecting the required number of examples

            # print(line)
            parts = extract_entity_and_fine_grained_roles(line.strip().split("\t"))
            # print(parts)
            # Join parts into a string if it's a list and then split it by tabs
            parts_string = "\t".join(parts)
            file_id, entity, start, end, main_role, fine_grained_role = parts_string.split("\t")
            
            file_path = os.path.join(train_documents_folder, file_id)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as doc_file:
                    text = doc_file.read()
                context = extract_relevant_context(text, start, end)
                examples.append({
                    "entity": entity,
                    "context": context,
                    "main_role": main_role,
                    "fine_grained_role": fine_grained_role
                })
    return examples


# Generate Prompt with Examples
def generate_prompt_with_examples(entity, text, examples):
    formatted_examples = "\n".join(
        f"Entity: {ex['entity']}\nContext: {ex['context']}\nMain Role: {ex['main_role']}\nFine-Grained Role: {ex['fine_grained_role']}"
        for ex in examples
    )
    return PLANNER_PROMPT.format(examples=formatted_examples, entity=entity, text=text)


plan_prompt = PromptTemplate(template=PLANNER_PROMPT, input_variables=["entity", "text"])
plan_chain = LLMChain(llm=llm, prompt=plan_prompt)



# Helper Function: Extract Centered Context with Delimiters
def extract_relevant_context(text, start, end, window_size=512):
    start, end = int(start), int(end)
    start_idx = max(0, start - window_size)
    end_idx = min(len(text), end + window_size)

    # Mark the entity with delimiters
    entity_text = text[start:end]
    marked_entity = f"<<<{entity_text}>>>"
    context = text[start_idx:start] + marked_entity + text[end:end_idx]
    
    return context

# Load Dataset
def load_data(entity_mentions_file, documents_folder):
    data = []
    with open(entity_mentions_file, 'r', encoding='utf-8') as f:
        for line in f:
            file_id, entity, start, end = line.strip().split("\t")
            file_path = os.path.join(documents_folder, file_id)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as doc_file:
                    text = doc_file.read()
                context = extract_relevant_context(text, start, end)
                data.append({
                    "file_id": file_id, 
                    "entity": entity, 
                    "start": start, 
                    "end": end, 
                    "context": context
                })
    return data


# Call Model with Few-Shot Prompt
@retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
def call_model_with_few_shot(entity, text, examples):
    prompt = generate_prompt_with_examples(entity, text, examples)

    # print(f"entity: ", entity)
    # print(f"text", text)
    return plan_chain.run({"entity": entity, "text": text, "examples": prompt})



def predict_roles(data, output_file, train_entity_mentions_file, train_documents_folder):
    # Load few-shot examples
    examples = load_few_shot_examples(train_entity_mentions_file, train_documents_folder)

    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # Initialize empty file

    for idx, item in enumerate(data):
        try:
            # Generate prediction using few-shot examples
            response = call_model_with_few_shot(item["entity"], item["context"], examples)

            # Parse response
            main_role = response.split("Main Role:")[1].split("\n")[0].strip()
            fine_grained_role = response.split("Fine-Grained Role:")[1].strip()

            # print(fine_grained_role)

            # Ensure fine-grained roles are tab-separated, except for "Foreign Adversary"
            fine_grained_roles = fine_grained_role.split(" ")

            # Check if "Foreign Adversary" is in the roles and treat it specially
            if "Foreign" in fine_grained_roles:
                # Keep "Foreign Adversary" as space-separated
                fine_grained_roles = [" ".join(fine_grained_roles[i:i+2]) if fine_grained_roles[i] == "Foreign" else fine_grained_roles[i] 
                                      for i in range(0, len(fine_grained_roles), 2)]

            # Now, join the roles using tab, but the "Foreign Adversary" will be space-separated
            fine_grained_roles_tab_separated = "\t".join(fine_grained_roles)


            # fine_grained_roles_tab_separated = "\t".join(fine_grained_role.split(" "))

            # Append to file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{item['file_id']}\t{item['entity']}\t{item['start']}\t{item['end']}\t{main_role}\t{fine_grained_roles_tab_separated}\n")

        except Exception as e:
            continue  # Handle errors gracefully

        # Optional delay to avoid hitting API rate limits
        time.sleep(5)

 

# Main Script
if __name__ == "__main__":
    entity_mentions_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
    documents_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
    output_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\output_predictions.txt"

    train_entity_mentions_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\training_data_16_October_release\subtask-1-annotations.txt" 
    train_documents_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\training_data_16_October_release\raw-documents" 

    # Load and Process Data
    data = load_data(entity_mentions_file, documents_folder)

    predict_roles(data, output_file, train_entity_mentions_file, train_documents_folder)
