# # # # # import os
# # # # # import openai
# # # # # import json

# # # # # # Set up the API key and endpoint for Azure OpenAI GPT-4
# # # # # openai.api_key = 'your_azure_api_key'
# # # # # endpoint = "https://your_openai_endpoint.openai.azure.com/"
# # # # # deployment_id = "gpt-4-deployment-id"

# # # # # # Define the main roles and fine-grained roles
# # # # # main_roles = ["Protagonist", "Antagonist", "Innocent"]

# # # # # individual_fine_grained_roles = [
# # # # #     'Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous',
# # # # #     'Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 
# # # # #     'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot',
# # # # #     'Forgotten', 'Exploited', 'Victim', 'Scapegoat'
# # # # # ]

# # # # # fine_grained_roles = {
# # # # #     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
# # # # #     "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
# # # # #                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
# # # # #     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# # # # # }

# # # # # # Path to entity mentions file
# # # # # entity_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
# # # # # document_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"

# # # # # # Function to read entity mentions from the file
# # # # # def read_entity_mentions(file_path):
# # # # #     entities = []
# # # # #     with open(file_path, 'r') as file:
# # # # #         for line in file:
# # # # #             parts = line.strip().split("\t")
# # # # #             if len(parts) == 4:
# # # # #                 filename, entity, start, end = parts
# # # # #                 entities.append((filename, entity, int(start), int(end)))
# # # # #     return entities

# # # # # # Function to get text from a document based on filename
# # # # # def get_text_from_file(filename):
# # # # #     file_path = os.path.join(document_folder, filename)
# # # # #     with open(file_path, 'r', encoding='utf-8') as file:
# # # # #         return file.read()

# # # # # # Function to call GPT-4 to predict the main role and fine-grained role
# # # # # def predict_roles(entity_text, entity):
# # # # #     prompt = f"Given the text, classify the main role of the entity '{entity}' and assign a fine-grained role:\n\n{entity_text}\n\nMain role options: {', '.join(main_roles)}\nFine-grained roles options: {', '.join(individual_fine_grained_roles)}\nPlease provide both main and fine-grained roles."
    
# # # # #     try:
# # # # #         response = openai.Completion.create(
# # # # #             engine=deployment_id,
# # # # #             prompt=prompt,
# # # # #             max_tokens=150,
# # # # #             n=1,
# # # # #             stop=None,
# # # # #             temperature=0.7
# # # # #         )
# # # # #         result = response.choices[0].text.strip()
# # # # #         return result
# # # # #     except Exception as e:
# # # # #         print(f"Error while predicting roles for entity '{entity}': {e}")
# # # # #         return None

# # # # # # Main function to process entities and classify roles
# # # # # def process_entities_and_predict_roles():
# # # # #     entities = read_entity_mentions(entity_file_path)
    
# # # # #     for filename, entity, start, end in entities:
# # # # #         document_text = get_text_from_file(filename)
        
# # # # #         # Extract the entity's text from the document
# # # # #         entity_text = document_text[start:end]
        
# # # # #         # Predict the main role and fine-grained role
# # # # #         result = predict_roles(entity_text, entity)
        
# # # # #         if result:
# # # # #             # Process the result to separate main role and fine-grained role
# # # # #             try:
# # # # #                 main_role, fine_grained_role = result.split("\n")
# # # # #                 print(f"Entity: {entity}")
# # # # #                 print(f"Main Role: {main_role}")
# # # # #                 print(f"Fine-grained Role: {fine_grained_role}")
# # # # #                 print("-" * 50)
# # # # #             except Exception as e:
# # # # #                 print(f"Error processing the result for entity '{entity}': {e}")

# # # # # # Run the process
# # # # # process_entities_and_predict_roles()
# # # # import os
# # # # from langchain_community.chat_models.azure_openai import AzureChatOpenAI

# # # # # Define constants
# # # # BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/"
# # # # API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# # # # DEPLOYMENT_NAME = "gpt-4o-mini"
# # # # API_VERSION = "2024-02-15-preview"

# # # # # Initialize the LLM
# # # # llm = AzureChatOpenAI(
# # # #     openai_api_base=BASE_URL,
# # # #     openai_api_version=API_VERSION,
# # # #     deployment_name=DEPLOYMENT_NAME,
# # # #     openai_api_key=API_KEY,
# # # #     openai_api_type="azure",
# # # #     temperature=0,
# # # # )

# # # # # Define main roles and fine-grained roles
# # # # main_roles = ["Protagonist", "Antagonist", "Innocent"]

# # # # fine_grained_roles = {
# # # #     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
# # # #     "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
# # # #                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
# # # #     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# # # # }

# # # # # Define file paths
# # # # entity_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
# # # # documents_folder_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"

# # # # # Helper function to read entities from the entity file
# # # # def read_entities(entity_file_path):
# # # #     entities = []
# # # #     with open(entity_file_path, 'r', encoding='utf-8') as file:
# # # #         for line in file:
# # # #             parts = line.strip().split("\t")
# # # #             if len(parts) == 4:
# # # #                 filename, entity, start, end = parts
# # # #                 entities.append((filename, entity, int(start), int(end)))
# # # #     return entities

# # # # # Helper function to get the text of the entity from the document
# # # # def get_entity_text(filename, start, end):
# # # #     doc_path = os.path.join(documents_folder_path, filename)
# # # #     with open(doc_path, 'r', encoding='utf-8') as file:
# # # #         text = file.read()
# # # #         return text[start:end]

# # # # # Helper function to generate GPT-4o predictions for main and fine-grained roles
# # # # def get_roles_from_model(entity_text):
# # # #     # Prepare the prompt for GPT model
# # # #     prompt = f"Given the entity text: \"{entity_text}\", classify it into the following roles: {', '.join(main_roles)}. Also, provide fine-grained roles for each main role, as per the following categories: {fine_grained_roles}."
    
# # # #     # Make the API request
# # # #     print(prompt)  # Add this line before llm.generate

# # # #     response = llm.generate([prompt])
# # # #     return response['choices'][0]['text']

# # # # # Main function to process the dataset and predict roles
# # # # def process_entities_and_predict():
# # # #     # Read entities from the entity file
# # # #     entities = read_entities(entity_file_path)

# # # #     results = []
# # # #     for filename, entity, start, end in entities:
# # # #         # Get the entity text from the document
# # # #         entity_text = get_entity_text(filename, start, end)
        
# # # #         # Get the roles for the entity
# # # #         roles_prediction = get_roles_from_model(entity_text)
        
# # # #         # Append the result
# # # #         results.append((filename, entity, roles_prediction))
    
# # # #     # Output the results
# # # #     for filename, entity, prediction in results:
# # # #         print(f"Entity: {entity} in {filename} - Roles Prediction: {prediction}")

# # # # # Run the prediction process
# # # # process_entities_and_predict()


# # # # from langchain_openai import AzureChatOpenAI
# # # # from langchain_community.chat_models.azure_openai import AzureChatOpenAI
# # # # import os
# # # # import openai
# # # # import json


# # # # # Define constants
# # # # BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/openai"
# # # # API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# # # # DEPLOYMENT_NAME = "gpt-4o-mini"
# # # # API_VERSION = "2024-02-15-preview"

# # # # # Initialize the LLM
# # # # llm = AzureChatOpenAI(
# # # #     azure_endpoint=BASE_URL,
# # # #     azure_api_key=API_KEY,
# # # #     azure_deployment_name=DEPLOYMENT_NAME,
# # # #     azure_api_version=API_VERSION,
# # # #     temperature=0,
# # # # )

# # # # # Define main roles and fine-grained roles
# # # # main_roles = ["Protagonist", "Antagonist", "Innocent"]
# # # # fine_grained_roles = {
# # # #     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
# # # #     "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
# # # #                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
# # # #     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# # # # }

# # # # # Helper function to generate GPT-4o predictions for main and fine-grained roles
# # # # def get_roles_from_model(entity_text):
# # # #     # Prepare the prompt for GPT model
# # # #     prompt = f"Given the entity text: \"{entity_text}\", classify it into the following roles: {', '.join(main_roles)}. Also, provide fine-grained roles for each main role, as per the following categories: {fine_grained_roles}."
    
# # # #     # Make the API request
# # # #     print(prompt)  # Add this line before calling llm.chat

# # # #     response = llm.chat([{"role": "user", "content": prompt}])  # Use chat method if that's the API update
# # # #     return response['choices'][0]['message']['content']

# # # # # Main function to process the dataset and predict roles
# # # # def process_entities_and_predict():
# # # #     # Read entities from the entity file
# # # #     entities = read_entities(entity_file_path)

# # # #     results = []
# # # #     for filename, entity, start, end in entities:
# # # #         # Get the entity text from the document
# # # #         entity_text = get_entity_text(filename, start, end)
        
# # # #         # Get the roles for the entity
# # # #         roles_prediction = get_roles_from_model(entity_text)
        
# # # #         # Append the result
# # # #         results.append((filename, entity, roles_prediction))
    
# # # #     # Output the results
# # # #     for filename, entity, prediction in results:
# # # #         print(f"Entity: {entity} in {filename} - Roles Prediction: {prediction}")


# # # # # Define file paths
# # # # entity_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
# # # # documents_folder_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"

# # # # # Helper function to read entities from the entity file
# # # # def read_entities(entity_file_path):
# # # #     entities = []
# # # #     with open(entity_file_path, 'r', encoding='utf-8') as file:
# # # #         for line in file:
# # # #             parts = line.strip().split("\t")
# # # #             if len(parts) == 4:
# # # #                 filename, entity, start, end = parts
# # # #                 entities.append((filename, entity, int(start), int(end)))
# # # #     return entities

# # # # # Helper function to get the text of the entity from the document
# # # # def get_entity_text(filename, start, end):
# # # #     doc_path = os.path.join(documents_folder_path, filename)
# # # #     with open(doc_path, 'r', encoding='utf-8') as file:
# # # #         text = file.read()
# # # #         return text[start:end]

# # # # # Run the prediction process
# # # # process_entities_and_predict()


# # # # import os
# # # # import json
# # # # from langchain_community.chat_models.azure_openai import AzureChatOpenAI

# # # # # Define constants
# # # # BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/openai"
# # # # API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# # # # DEPLOYMENT_NAME = "gpt-4o-mini"
# # # # API_VERSION = "2024-02-15-preview"

# # # # # Initialize the LLM
# # # # llm = AzureChatOpenAI(
# # # #     azure_endpoint=BASE_URL,
# # # #     azure_api_key=API_KEY,
# # # #     azure_deployment_name=DEPLOYMENT_NAME,
# # # #     azure_api_version=API_VERSION,
# # # #     temperature=0,
# # # # )

# # # # # Define main roles and fine-grained roles
# # # # main_roles = ["Protagonist", "Antagonist", "Innocent"]
# # # # fine_grained_roles = {
# # # #     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
# # # #     "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
# # # #                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
# # # #     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# # # # }

# # # # # Helper function to generate GPT-4o predictions for main and fine-grained roles
# # # # def get_roles_from_model(entity_text):
# # # #     prompt = (
# # # #         f"Given the entity text: \"{entity_text}\", classify it into the following roles: "
# # # #         f"{', '.join(main_roles)}. Also, provide fine-grained roles for each main role, "
# # # #         f"as per the following categories: {fine_grained_roles}."
# # # #     )
# # # #     response = llm.call([{"role": "user", "content": prompt}])  # Update method call
# # # #     return response[0]['content']  # Assuming the response structure

# # # # # Main function to process the dataset and predict roles
# # # # def process_entities_and_predict():
# # # #     entities = read_entities(entity_file_path)
# # # #     results = []
# # # #     for filename, entity, start, end in entities:
# # # #         entity_text = get_entity_text(filename, start, end)
# # # #         roles_prediction = get_roles_from_model(entity_text)
# # # #         results.append({
# # # #             "filename": filename,
# # # #             "entity": entity,
# # # #             "roles_prediction": roles_prediction
# # # #         })
# # # #     save_results_to_file(results, output_file_path)

# # # # # Helper function to read entities from the entity file
# # # # def read_entities(entity_file_path):
# # # #     entities = []
# # # #     with open(entity_file_path, 'r', encoding='utf-8') as file:
# # # #         for line in file:
# # # #             parts = line.strip().split("\t")
# # # #             if len(parts) == 4:
# # # #                 filename, entity, start, end = parts
# # # #                 entities.append((filename, entity, int(start), int(end)))
# # # #     return entities

# # # # # Helper function to get the text of the entity from the document
# # # # def get_entity_text(filename, start, end):
# # # #     doc_path = os.path.join(documents_folder_path, filename)
# # # #     with open(doc_path, 'r', encoding='utf-8') as file:
# # # #         text = file.read()
# # # #         return text[start:end]

# # # # # Helper function to save results to a JSON file
# # # # def save_results_to_file(results, output_file_path):
# # # #     with open(output_file_path, 'w', encoding='utf-8') as file:
# # # #         json.dump(results, file, indent=4, ensure_ascii=False)
# # # #     print(f"Results saved to {output_file_path}")

# # # # # Define file paths
# # # # entity_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
# # # # documents_folder_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
# # # # output_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\predictions.json"

# # # # # Run the prediction process
# # # # process_entities_and_predict()


# # # import os
# # # import json
# # # from langchain.prompts import PromptTemplate
# # # from langchain.chains import LLMChain
# # # from langchain_community.chat_models.azure_openai import AzureChatOpenAI

# # # # Define constants
# # # BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/openai"
# # # API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# # # DEPLOYMENT_NAME = "gpt-4o-mini"
# # # API_VERSION = "2024-02-15-preview"

# # # # Initialize the LLM
# # # llm = AzureChatOpenAI(
# # #     azure_endpoint=BASE_URL,
# # #     azure_api_key=API_KEY,
# # #     azure_deployment_name=DEPLOYMENT_NAME,
# # #     azure_api_version=API_VERSION,
# # #     temperature=0,
# # # )

# # # # Define roles and fine-grained roles
# # # main_roles = ["Protagonist", "Antagonist", "Innocent"]
# # # fine_grained_roles = {
# # #     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
# # #     "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
# # #                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
# # #     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# # # }

# # # # Define the prompt template
# # # roles_prompt = PromptTemplate(
# # #     template=(
# # #         "Given the entity text: \"{entity_text}\", classify it into the following main roles: "
# # #         "{main_roles}. Additionally, provide fine-grained roles for each main role, as per these "
# # #         "categories: {fine_grained_roles}."
# # #     ),
# # #     input_variables=["entity_text", "main_roles", "fine_grained_roles"],
# # # )

# # # # Define the LLM chain
# # # roles_chain = LLMChain(llm=llm, prompt=roles_prompt)

# # # # Main function to process the dataset and predict roles
# # # def process_entities_and_predict():
# # #     entities = read_entities(entity_file_path)
# # #     results = []
# # #     for filename, entity, start, end in entities:
# # #         entity_text = get_entity_text(filename, start, end)
# # #         # Use the chain to get predictions
# # #         roles_prediction = roles_chain.run({
# # #             "entity_text": entity_text,
# # #             "main_roles": ", ".join(main_roles),
# # #             "fine_grained_roles": fine_grained_roles,
# # #         })
# # #         results.append({
# # #             "filename": filename,
# # #             "entity": entity,
# # #             "roles_prediction": roles_prediction
# # #         })
# # #     save_results_to_file(results, output_file_path)

# # # # Helper function to read entities from the entity file
# # # def read_entities(entity_file_path):
# # #     entities = []
# # #     with open(entity_file_path, 'r', encoding='utf-8') as file:
# # #         for line in file:
# # #             parts = line.strip().split("\t")
# # #             if len(parts) == 4:
# # #                 filename, entity, start, end = parts
# # #                 entities.append((filename, entity, int(start), int(end)))
# # #     return entities

# # # # Helper function to get the text of the entity from the document
# # # def get_entity_text(filename, start, end):
# # #     doc_path = os.path.join(documents_folder_path, filename)
# # #     with open(doc_path, 'r', encoding='utf-8') as file:
# # #         text = file.read()
# # #         return text[start:end]

# # # # Helper function to save results to a JSON file
# # # def save_results_to_file(results, output_file_path):
# # #     with open(output_file_path, 'w', encoding='utf-8') as file:
# # #         json.dump(results, file, indent=4, ensure_ascii=False)
# # #     print(f"Results saved to {output_file_path}")

# # # # Define file paths
# # # entity_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
# # # documents_folder_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
# # # output_file_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\predictions.json"

# # # # Run the prediction process
# # # process_entities_and_predict()


# # import os
# # from langchain_community.chat_models.azure_openai import AzureChatOpenAI
# # from langchain.prompts import PromptTemplate
# # from langchain.chains import LLMChain

# # # Azure GPT-4o Mini Configuration
# # BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/"
# # API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# # DEPLOYMENT_NAME = "gpt-4o-mini"
# # API_VERSION = "2024-02-15-preview"

# # llm = AzureChatOpenAI(
# #     openai_api_base=BASE_URL,
# #     openai_api_version=API_VERSION,
# #     deployment_name=DEPLOYMENT_NAME,
# #     openai_api_key=API_KEY,
# #     openai_api_type="azure",
# #     temperature=0,
# # )

# # # Paths
# # entity_mentions_path = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
# # documents_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
# # output_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\predictions.txt"

# # # Roles
# # main_roles = ["Protagonist", "Antagonist", "Innocent"]
# # fine_grained_roles = {
# #     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
# #     "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
# #                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
# #     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# # }

# # # Prompt Template
# # PLANNER_PROMPT = """
# # You are an AI model specialized in text analysis. Given the following text, extract the entity's main role (Protagonist, Antagonist, Innocent) 
# # and fine-grained role based on its context.

# # Entity: {entity}
# # Text: {text}

# # Provide your response in this JSON format:
# # {{
# #     "Main Role": "<main_role>",
# #     "Fine-Grained Role": "<fine_grained_role>"
# # }}
# # """

# # plan_prompt = PromptTemplate(template=PLANNER_PROMPT, input_variables=["entity", "text"])
# # plan_chain = LLMChain(llm=llm, prompt=plan_prompt)

# # # Helper Function: Process Data
# # def load_data(entity_mentions_file, documents_folder):
# #     data = []
# #     with open(entity_mentions_file, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             file_id, entity, start, end = line.strip().split("\t")
# #             file_path = os.path.join(documents_folder, file_id)
# #             if os.path.exists(file_path):
# #                 with open(file_path, 'r', encoding='utf-8') as doc_file:
# #                     text = doc_file.read()
# #                 entity_text = text[int(start):int(end)]
# #                 data.append({"file_id": file_id, "entity": entity, "context": text, "entity_text": entity_text})
# #     return data

# # # Helper Function: Predict Roles
# # def predict_roles(data):
# #     predictions = []
# #     for item in data:
# #         input_data = {
# #             "entity": item["entity"],
# #             "text": item["context"]
# #         }
# #         response = plan_chain.run(input_data)
# #         predictions.append({"file_id": item["file_id"], "entity": item["entity"], "response": response})
# #     return predictions

# # # Main Code
# # data = load_data(entity_mentions_path, documents_folder)
# # predictions = predict_roles(data)

# # # Save Predictions
# # with open(output_file, 'w', encoding='utf-8') as f:
# #     for prediction in predictions:
# #         f.write(f"File: {prediction['file_id']}\n")
# #         f.write(f"Entity: {prediction['entity']}\n")
# #         f.write(f"Prediction: {prediction['response']}\n")
# #         f.write("\n")
# # print(f"Predictions saved to {output_file}")
# import os
# import time
# from langchain.chains import LLMChain
# from langchain_community.chat_models.azure_openai import AzureChatOpenAI
# from langchain.prompts import PromptTemplate
# from tenacity import retry, wait_fixed, stop_after_attempt

# # Azure OpenAI Configuration
# BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/"
# API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# DEPLOYMENT_NAME = "gpt-4o-mini"
# API_VERSION = "2024-02-15-preview"

# llm = AzureChatOpenAI(
#     openai_api_base=BASE_URL,
#     openai_api_version=API_VERSION,
#     deployment_name=DEPLOYMENT_NAME,
#     openai_api_key=API_KEY,
#     openai_api_type="azure",
#     temperature=0,
# )

# # Prompt Template
# PLANNER_PROMPT = """
# You are a role classifier for entities in a text. The entity and context will be provided, and you must predict:
# 1. The main role: Protagonist, Antagonist, or Innocent.
# 2. The fine-grained role based on the main role strictly from the following list:     
#     - if main role is "Protagonist", then fine-grained role can be one or multiple from ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"]
#     - if main role is "Antagonist", then fine-grained role can be one or multiple from ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"]
#     - if main role is "Innocent", then fine-grained role can be one or multiple from ["Forgotten", "Exploited", "Victim", "Scapegoat"]
    
# Here is the input:
# Entity: {entity}
# Context: {text}

# Output the result as:
# Main Role: <main role>
# Fine-Grained Role: <fine-grained role>
# """
# plan_prompt = PromptTemplate(template=PLANNER_PROMPT, input_variables=["entity", "text"])
# plan_chain = LLMChain(llm=llm, prompt=plan_prompt)

# # Role Categories
# main_roles = ["Protagonist", "Antagonist", "Innocent"]
# fine_grained_roles = {
#     "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
#     "Antagonist": [
#         "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
#         "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"
#     ],
#     "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
# }

# # Helper Function: Extract Context
# def extract_relevant_context(text, start, end, window_size=200):
#     start_idx = max(0, int(start) - window_size)
#     end_idx = min(len(text), int(end) + window_size)
#     return text[start_idx:end_idx]

# # Load Dataset
# def load_data(entity_mentions_file, documents_folder):
#     data = []
#     with open(entity_mentions_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             file_id, entity, start, end = line.strip().split("\t")
#             file_path = os.path.join(documents_folder, file_id)
#             if os.path.exists(file_path):
#                 with open(file_path, 'r', encoding='utf-8') as doc_file:
#                     text = doc_file.read()
#                 context = extract_relevant_context(text, start, end)
#                 data.append({"file_id": file_id, "entity": entity, "context": context})
#     return data

# # Retry Logic for API Calls
# @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
# def call_model_with_retry(entity, text):
#     return plan_chain.run({"entity": entity, "text": text})

# # Prediction Function with Delay
# def predict_roles(data, output_file):
#     predictions = []
#     for idx, item in enumerate(data):
#         print(f"Processing item {idx+1}/{len(data)}: Entity {item['entity']}")
#         try:
#             response = call_model_with_retry(item["entity"], item["context"])
#             predictions.append({"file_id": item["file_id"], "entity": item["entity"], "response": response})
#             print(f"Response: {response}")
#         except Exception as e:
#             print(f"Error processing item {idx}: {e}")
#             continue
#         time.sleep(5)  # Delay to avoid hitting rate limits

#     # Save Results
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for prediction in predictions:
#             f.write(f"{prediction['file_id']}\t{prediction['entity']}\t{prediction['response']}\n")
#     print(f"Predictions saved to {output_file}")

# # Main Script
# if __name__ == "__main__":
#     entity_mentions_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
#     documents_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
#     output_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\output_predictions.txt"

#     # Load and Process Data
#     data = load_data(entity_mentions_file, documents_folder)
#     predict_roles(data, output_file)


# import os
# import time
# from langchain.chains import LLMChain
# from langchain_community.chat_models.azure_openai import AzureChatOpenAI
# from langchain.prompts import PromptTemplate
# from tenacity import retry, wait_fixed, stop_after_attempt

# # Azure OpenAI Configuration
# BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/"
# API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
# DEPLOYMENT_NAME = "gpt-4o-mini"
# API_VERSION = "2024-02-15-preview"

# llm = AzureChatOpenAI(
#     openai_api_base=BASE_URL,
#     openai_api_version=API_VERSION,
#     deployment_name=DEPLOYMENT_NAME,
#     openai_api_key=API_KEY,
#     openai_api_type="azure",
#     temperature=0,
# )

# # Prompt Template
# PLANNER_PROMPT = """
# You are a role classifier for entities in a text. The entity and context will be provided, and you must predict:
# 1. The main role: Protagonist, Antagonist, or Innocent.
# 2. The fine-grained role based on the main role should STRICTLY be from the following list:     
#     - if main role is "Protagonist", then fine-grained role can be one or multiple from ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"]
#     - if main role is "Antagonist", then fine-grained role can be one or multiple from ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"]
#     - if main role is "Innocent", then fine-grained role can be one or multiple from ["Forgotten", "Exploited", "Victim", "Scapegoat"]
    
# Here is the input:
# Entity: {entity}
# Context: {text}

# Output the result as:
# Main Role: <main role>
# Fine-Grained Role: <fine-grained role>
# """

# plan_prompt = PromptTemplate(template=PLANNER_PROMPT, input_variables=["entity", "text"])
# plan_chain = LLMChain(llm=llm, prompt=plan_prompt)

# # Helper Function: Extract Centered Context with Delimiters
# def extract_relevant_context(text, start, end, window_size=200):
#     start, end = int(start), int(end)
#     start_idx = max(0, start - window_size)
#     end_idx = min(len(text), end + window_size)

#     # Mark the entity with delimiters
#     entity_text = text[start:end]
#     marked_entity = f"<<<{entity_text}>>>"
#     context = text[start_idx:start] + marked_entity + text[end:end_idx]
    
#     return context

# # Load Dataset
# def load_data(entity_mentions_file, documents_folder):
#     data = []
#     with open(entity_mentions_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             file_id, entity, start, end = line.strip().split("\t")
#             file_path = os.path.join(documents_folder, file_id)
#             if os.path.exists(file_path):
#                 with open(file_path, 'r', encoding='utf-8') as doc_file:
#                     text = doc_file.read()
#                 context = extract_relevant_context(text, start, end)
#                 data.append({
#                     "file_id": file_id, 
#                     "entity": entity, 
#                     "start": start, 
#                     "end": end, 
#                     "context": context
#                 })
#     return data

# # Retry Logic for API Calls
# @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
# def call_model_with_retry(entity, text):
#     return plan_chain.run({"entity": entity, "text": text})

# # Prediction Function with Delay
# def predict_roles(data, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for idx, item in enumerate(data):
#             print(f"Processing item {idx+1}/{len(data)}: Entity {item['entity']}")
#             try:
#                 response = call_model_with_retry(item["entity"], item["context"])
#                 print(f"Response: {response}")
                
#                 # Parse response
#                 main_role = response.split("Main Role:")[1].split("\n")[0].strip()
#                 fine_grained_role = response.split("Fine-Grained Role:")[1].strip()
                
#                 # Write to file
#                 f.write(f"{item['file_id']}\t{item['entity']}\t{item['start']}\t{item['end']}\t{main_role}\t{fine_grained_role}\n")
#             except Exception as e:
#                 print(f"Error processing item {idx+1}: {e}")
#                 continue
#             time.sleep(5)  # Delay to avoid hitting rate limits

#     print(f"Predictions saved to {output_file}")

# # Main Script
# if __name__ == "__main__":
#     entity_mentions_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-entity-mentions_EN.txt"
#     documents_folder = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\subtask1_multilingual_dataset\dev-documents_25_October\subtask-1-documents"
#     output_file = r"D:\Malak Doc\Malak Education\MBZUAI\Academic years\fall 2024\NLP\Assignments\Assignment2\NLP701_assignment2_subtask1\output_predictions.txt"

#     # Load and Process Data
#     data = load_data(entity_mentions_file, documents_folder)
#     predict_roles(data, output_file)


import os
import time
from langchain.chains import LLMChain
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from tenacity import retry, wait_fixed, stop_after_attempt

# Azure OpenAI Configuration
BASE_URL = "https://ai-malakmansour0093ai939299253478.openai.azure.com/"
API_KEY = "8IGmt7raN2eCyC22OxPNX5xm4MYKPQdBZq0WkqHOWUDc2cqcCyA2JQQJ99AKACHYHv6XJ3w3AAAAACOGcuHo"
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2024-02-15-preview"

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