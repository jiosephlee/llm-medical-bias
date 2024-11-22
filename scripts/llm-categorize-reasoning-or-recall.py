import json
import os
from tqdm import tqdm

# Path to MedMCQA .jsonl file and metadata file
input_file = "/data/medMCQA/dev.jsonl"
metadata_file = "/data/medMCQA/category_metadata.json"

# Function to determine question type based on question text
def categorize_question(question_text):
    # Basic keyword check to determine if the question involves a patient scenario
    patient_keywords = ["patient", "boy", "girl", "man", "woman", "pregnancy", "old"]
    if any(keyword in question_text.lower() for keyword in patient_keywords):
        return "clinical reasoning"  # likely involves a patient scenario
    return "clinical knowledge recall"  # likely factual or recall-based

# Load or initialize metadata file
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
else:
    print(f"Metadata file not found. Creating {metadata_file}.")
    metadata = {}

# Process each question in the MedMCQA dataset
with open(input_file, 'r') as f:
    for line in tqdm(f, desc="Processing questions"):
        question_data = json.loads(line.strip())
        
        # Categorize the question
        question_type = categorize_question(question_data["question"])
        
        # Update question data with new metadata
        question_data["category"] = question_type
        
        # Update the metadata dictionary with question id as the key
        metadata[question_data["id"]] = {
            "question": question_data["question"],
            "category": question_type
        }

# Save updated metadata to category_metadata.json
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
    
print(f"Updated metadata saved to {metadata_file}")