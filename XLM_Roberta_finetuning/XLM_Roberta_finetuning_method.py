# LIBRARIES
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModel,
)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import softmax


import numpy as np




# SECTION: TRAINING
train_annotations_file = "/kaggle/input/processed-5/assignment2_processed_dataset/training_data_16_October_release/subtask-1-annotations.txt"
train_raw_documents_folder = "/kaggle/input/processed-5/assignment2_processed_dataset/training_data_16_October_release/raw-documents"

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


def load_annotations(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    processed_lines = [
        extract_entity_and_fine_grained_roles(line.strip().split("\t"))
        for line in lines
    ]
    return pd.DataFrame(
        processed_lines,
        columns=[
            "article_id",
            "entity",
            "start_offset",
            "end_offset",
            "main_role",
            "fine_grained_roles",
        ],
    )


def load_document(article_id, folder_path):
    file_path = os.path.join(folder_path, article_id)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Data Preprocessing: Adding special token around entities
def preprocess_entity(entity, context, start_offset, end_offset, tokenizer):
    context_with_entity = context[:start_offset] + f"<ENTITY>{entity}</ENTITY>" + context[end_offset:]
    inputs = tokenizer(
        context_with_entity,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs


# Dataset Class with Hierarchical Classification
class RoleDataset(Dataset):
    def __init__(self, annotations, documents, tokenizer, max_length=512):
        self.annotations = annotations
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        article_id = annotation["article_id"]
        entity = annotation["entity"]
        start_offset = int(annotation["start_offset"])
        end_offset = int(annotation["end_offset"])
        main_role = annotation["main_role"]
        fine_grained_roles = annotation["fine_grained_roles"].split(" ")

        # Get the document text
        document = self.documents[article_id]

        # Preprocess document to add special entity tokens
        inputs = preprocess_entity(entity, document, start_offset, end_offset, self.tokenizer)

        # Label encoding
        all_main_roles = set(main_roles)
        all_fine_grained_roles = set(individual_fine_grained_roles)

        main_role_label = [main_role]
        fine_grained_roles_labels = fine_grained_roles

        mlb_main = MultiLabelBinarizer(classes=list(all_main_roles))
        mlb_fine = MultiLabelBinarizer(classes=list(all_fine_grained_roles))

        main_role_encoded = mlb_main.fit_transform([main_role_label])[0]
        fine_grained_encoded = mlb_fine.fit_transform([fine_grained_roles_labels])[0]

        # Combine labels
        labels = torch.cat(
            [
                torch.tensor(main_role_encoded, dtype=torch.float),
                torch.tensor(fine_grained_encoded, dtype=torch.float),
            ]
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


# Main Model with Hierarchical Classification
class RoleClassifier(nn.Module):
    def __init__(self, model_name, num_main_roles, num_fine_grained_roles):
        super(RoleClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.main_role_classifier = nn.Linear(hidden_size, num_main_roles)
        self.fine_grained_classifier = nn.Linear(hidden_size, num_fine_grained_roles)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation

        main_role_logits = self.main_role_classifier(cls_output)
        fine_grained_logits = self.fine_grained_classifier(cls_output)

        if labels is not None:
            main_role_labels = labels[:, :3]  # First 3 for main roles
            fine_grained_labels = labels[:, 3:]  # Remaining for fine-grained roles

            main_loss = BCEWithLogitsLoss()(main_role_logits, main_role_labels)
            fine_loss = BCEWithLogitsLoss()(fine_grained_logits, fine_grained_labels)

            loss = main_loss + fine_loss
            return loss, main_role_logits, fine_grained_logits

        return main_role_logits, fine_grained_logits


# Initialize the tokenizer and dataset
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load annotations and documents
annotations = load_annotations(train_annotations_file)
documents = {
    article_id: load_document(article_id, train_raw_documents_folder)
    for article_id in annotations["article_id"].unique()
}

train_dataset = RoleDataset(annotations, documents, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)




# Fine-grained roles and main roles
main_roles = ["Protagonist", "Antagonist", "Innocent"]

individual_fine_grained_roles = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous',
            'Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 
            'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot',
            'Forgotten', 'Exploited', 'Victim', 'Scapegoat']

fine_grained_roles = {
    "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", 
                   "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
    "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
}


# Initialize the model
num_main_roles = len(main_roles)
num_fine_grained_roles = len(individual_fine_grained_roles)


model = RoleClassifier(model_name, num_main_roles, num_fine_grained_roles)


# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)


# Train the model
trainer.train()


# Save the model
torch.save(model.state_dict(), "./role_classifier/pytorch_model.bin")

tokenizer.save_pretrained("./role_classifier")


# Save the model configuration
model_config = {
    "model_name": model_name,
    "num_main_roles": num_main_roles,
    "num_fine_grained_roles": num_fine_grained_roles,
}
torch.save(model_config, "./role_classifier/model_config.pth")


# SECTION: SAVING THE MODEL
# Example usage with model loading and prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RoleClassifier(model_name, num_main_roles, num_fine_grained_roles)
model.load_state_dict(torch.load("./role_classifier/pytorch_model.bin"))
model.to(device)



# SECTION: TESTING
dev_raw_documents_folder = "/kaggle/input/processed-5/assignment2_processed_dataset/dev-documents_25_October/subtask-1-documents"

sigmoid_threshold = 0.1

# Load dev annotations
def load_dev_annotations(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    processed_lines = [
        extract_entity_and_fine_grained_roles(line.strip().split("\t"))
        for line in lines
    ]
    return pd.DataFrame(
        processed_lines, columns=["article_id", "entity", "start_offset", "end_offset"]
    )


# Dev Dataset
class DevRoleDataset(Dataset):
    def __init__(self, annotations, documents, tokenizer, max_length=512):
        self.annotations = annotations
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        article_id = annotation["article_id"]
        entity = annotation["entity"]
        start_offset = int(annotation["start_offset"])
        end_offset = int(annotation["end_offset"])

        
        # Get the document text
        document = self.documents[article_id]


        # ENTITY CENTER
# Center the entity in the context window
        entity_start_char_idx = start_offset
        entity_end_char_idx = end_offset

        # Define a character window around the entity
        half_window_size = self.max_length // 2
        left_context_start = max(0, entity_start_char_idx - half_window_size)
        right_context_end = min(len(document), entity_end_char_idx + half_window_size)

        # Slice the document around the entity
        context = document[left_context_start:right_context_end]

        # Adjust the entity offsets for the new context
        adjusted_entity_start = entity_start_char_idx - left_context_start
        adjusted_entity_end = entity_end_char_idx - left_context_start




        context_with_entity = context[:start_offset] + f"<ENTITY>{entity}</ENTITY>" + context[end_offset:]

        
        
        # Tokenize and prepare inputs
        inputs = self.tokenizer(
            # document,
            # context,
            context_with_entity,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),

            "article_id": article_id,
            "entity": entity,
            "start_offset": start_offset,
            "end_offset": end_offset,
        }


def predict_roles_hierarchical(model, dataloader, tokenizer):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # Main role prediction (first model for coarse classification)
            main_role_logits, fine_grained_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Apply softmax for main role classification
            main_role_probs = softmax(main_role_logits, dim=-1).cpu().numpy()
            main_role_preds = np.argmax(main_role_probs, axis=-1)

            # Fine-grained role prediction (based on the predicted main role)
            for i in range(len(input_ids)):
                main_role = main_roles[main_role_preds[i]]
                fine_grained_logits_for_role = fine_grained_logits[i]
                fine_grained_preds = (torch.sigmoid(fine_grained_logits_for_role) > sigmoid_threshold).cpu().numpy()

                # Map fine-grained roles based on the predicted main role
 
                # Example: A mapping of main roles to their index ranges in fine_grained_preds
                # Replace these ranges with the actual ranges for your use case
                main_role_index_bounds = {
                    "Protagonist": (0, 6),  # Indices for Protagonist fine-grained roles
                    "Antagonist": (6, 12),  # Indices for Antagonist fine-grained roles
                    "Innocent": (12, 18),   # Indices for Innocent fine-grained roles
                    # Add other roles and their bounds here
                }
                
                # Get the start and end index for the main_role
                start_idx, end_idx = main_role_index_bounds[main_role]
                
                # Filter fine-grained roles for the main_role based on its index range
                fine_grained_roles_predicted = [
                    fine_grained_roles[main_role][idx - start_idx]  # Adjust index relative to main_role's fine-grained roles
                    for idx, val in enumerate(fine_grained_preds[start_idx:end_idx])  # Iterate only over the range for the main_role
                    if val == 1  # Pick only the roles predicted as True
                ]
                

                

                # Collect results
                predictions.append(
                    {
                        "article_id": batch["article_id"][i],
                        "entity": batch["entity"][i],
                        "start_offset": batch["start_offset"][i],
                        "end_offset": batch["end_offset"][i],
                        "main_roles": [main_roles[main_role_preds[i]]],
                        "fine_grained_roles": fine_grained_roles_predicted,
                    }
                )

    return predictions
    
# Update the prediction function to save predictions in the desired format
def save_predictions(predictions, output_file):
    with open(output_file, "w") as f:
        for prediction in predictions:
            article_id = prediction["article_id"]
            entity = prediction["entity"]
            start_offset = prediction["start_offset"]  # Start offset for the entity
            end_offset = prediction["end_offset"]  # End offset for the entity
            main_role = prediction["main_roles"][0]  # Assume we are picking only one main_role
            fine_grained_roles = "\t".join(prediction["fine_grained_roles"])  # Join fine-grained roles with tab

            # Prepare the line in the required format
            line = f"{article_id}\t{entity}\t{start_offset}\t{end_offset}\t{main_role}\t{fine_grained_roles}\n"
            f.write(line)

    print(f"Predictions saved to {output_file}")


# SECTION: TESTING ON ENGLISH
dev_annotations_file = "/kaggle/input/processed-5/assignment2_processed_dataset/dev-documents_25_October/subtask-1-entity-mentions_EN.txt"


# Load dev annotations and documents
dev_annotations = load_dev_annotations(dev_annotations_file)
dev_documents = {
    article_id: load_document(article_id, dev_raw_documents_folder)
    for article_id in dev_annotations["article_id"].unique()
}

# Dev Dataset and DataLoader
dev_dataset = DevRoleDataset(dev_annotations, dev_documents, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)



# Assuming `dev_loader` is the DataLoader for the dev set
predictions = predict_roles_hierarchical(model, dev_loader, tokenizer)

# Save predictions to .txt file
save_predictions(predictions,"submission_EN.txt")


# SECTION: TESTING ON BULGARIAN
dev_annotations_file = "/kaggle/input/processed-5/assignment2_processed_dataset/dev-documents_25_October/subtask-1-entity-mentions_BG.txt"


# Load dev annotations and documents
dev_annotations = load_dev_annotations(dev_annotations_file)
dev_documents = {
    article_id: load_document(article_id, dev_raw_documents_folder)
    for article_id in dev_annotations["article_id"].unique()
}

# Dev Dataset and DataLoader
dev_dataset = DevRoleDataset(dev_annotations, dev_documents, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)




# Assuming `dev_loader` is the DataLoader for the dev set
predictions = predict_roles_hierarchical(model, dev_loader, tokenizer)

# Save predictions to .txt file
save_predictions(predictions,"submission_BG.txt")


# SECTION: TESTING ON HINDI
dev_annotations_file = "/kaggle/input/processed-5/assignment2_processed_dataset/dev-documents_25_October/subtask-1-entity-mentions_HI.txt"


# Load dev annotations and documents
dev_annotations = load_dev_annotations(dev_annotations_file)
dev_documents = {
    article_id: load_document(article_id, dev_raw_documents_folder)
    for article_id in dev_annotations["article_id"].unique()
}

# Dev Dataset and DataLoader
dev_dataset = DevRoleDataset(dev_annotations, dev_documents, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)



# Assuming `dev_loader` is the DataLoader for the dev set
predictions = predict_roles_hierarchical(model, dev_loader, tokenizer)

# Save predictions to .txt file
save_predictions(predictions,"submission_HI.txt")


# SECTION: TESTING ON PORTUGUESE
dev_annotations_file = "/kaggle/input/processed-5/assignment2_processed_dataset/dev-documents_25_October/subtask-1-entity-mentions_PT.txt"


# Load dev annotations and documents
dev_annotations = load_dev_annotations(dev_annotations_file)
dev_documents = {
    article_id: load_document(article_id, dev_raw_documents_folder)
    for article_id in dev_annotations["article_id"].unique()
}

# Dev Dataset and DataLoader
dev_dataset = DevRoleDataset(dev_annotations, dev_documents, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)


# Assuming `dev_loader` is the DataLoader for the dev set
predictions = predict_roles_hierarchical(model, dev_loader, tokenizer)

# Save predictions to .txt file
save_predictions(predictions,"submission_PT.txt")