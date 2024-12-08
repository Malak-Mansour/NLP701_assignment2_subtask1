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

# Prompt Template
PLANNER_PROMPT = """
You are a role classifier for entities in a text. The entity and context will be provided, and you must predict:
1. The main role: Protagonist, Antagonist, or Innocent.
2. The fine-grained role based on the main role should STRICTLY be from the following list:     
    - if main role is "Protagonist", then fine-grained role must strictly be one or multiple from ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"] separated by tabs not commas
    - if main role is "Antagonist", then fine-grained role must strictly be one or multiple from ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"]  separated by tabs not commas
    - if main role is "Innocent", then fine-grained role must strictly be one or multiple from ["Forgotten", "Exploited", "Victim", "Scapegoat"] separated by tabs not commas
    
Here is the input:
Entity: {entity}
Context: {text}

Output the result as:
Main Role: <main role>
Fine-Grained Role: <fine-grained role>
"""


plan_prompt = PromptTemplate(template=PLANNER_PROMPT, input_variables=["entity", "text"])
plan_chain = LLMChain(llm=llm, prompt=plan_prompt)

# Helper Function: Extract Centered Context with Delimiters
def extract_relevant_context(text, start, end, window_size=200):
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

# Retry Logic for API Calls
@retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
def call_model_with_retry(entity, text):
    return plan_chain.run({"entity": entity, "text": text})



# Prediction Function with Tab-Separated Roles and Incremental Writing
def predict_roles(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:  # Open in write mode initially to clear old data
        pass
        # f.write("file_id\tentity\tstart\tend\tmain_role\tfine_grained_roles\n")  # Add header

    for idx, item in enumerate(data):
        print(f"Processing item {idx + 1}/{len(data)}: Entity {item['entity']}")
        try:
            # Call the model and get the response
            response = call_model_with_retry(item["entity"], item["context"])
            print(f"Response: {response}")

            # Parse response
            main_role = response.split("Main Role:")[1].split("\n")[0].strip()
            fine_grained_role = response.split("Fine-Grained Role:")[1].strip()
            
            # Ensure fine-grained roles are tab-separated
            fine_grained_roles_tab_separated = "\t".join(fine_grained_role.split(", "))

            # Append to file
            with open(output_file, 'a', encoding='utf-8') as f:  # Append mode
                f.write(f"{item['file_id']}\t{item['entity']}\t{item['start']}\t{item['end']}\t{main_role}\t{fine_grained_roles_tab_separated}\n")

        except Exception as e:
            print(f"Error processing item {idx + 1}: {e}")
            continue

        # Optional delay to avoid hitting API rate limits
        time.sleep(5)

    print(f"Predictions saved to {output_file}")

 

# Main Script
if __name__ == "__main__":
    entity_mentions_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
    documents_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
    output_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\output_predictions.txt"

    # Load and Process Data
    data = load_data(entity_mentions_file, documents_folder)
    predict_roles(data, output_file)