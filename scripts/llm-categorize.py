import os
import json
import argparse
import time
from tqdm import tqdm
import utils.utils as utils
from datetime import datetime

# Global pools for tracking dynamic categories
global_pools = {
    "diseases": set(),
    "question_types": set(),
    "medical_specialties": set(),
    "severity_urgency": set(),
    "patient_demographics": set(),
    "concepts": set()
}

# Expected schema for validation
expected_schema = {
    "diseases": list,
    "question_types": list,
    "medical_specialties": list,
    "severity_urgency": list,
    "patient_demographics": list
}

# Global pool for tracking extracted concepts
global_concepts_pool = set()

def extract_concepts_with_gpt(question_text, max_try_num=3, model="gpt-4o-mini", debug=False):
    """Query GPT to extract knowledge concepts from the question."""
    
    prompt = f"""

    {question_text}.\n For the above question, extract the knowledge concepts that this question is trying to test. Give me the biomedical concept names alone. Answer in json format under the key 'concepts'.
    """
    prompt = f"""

    {question_text}.\n For the above question, extract the key knowledge concepts that this question is trying to test. Give me the biomedical concept names alone in their most concise form. Answer in json format under the key 'concepts'.
    """
    for attempt in range(max_try_num):
        response = utils.query_gpt(prompt, max_tokens=200, temperature=0, model=model, debug=debug, json=True)
        if response == -1:
            if debug:
                print(f"Attempt {attempt + 1}: GPT response failed.")
            time.sleep(1)
            continue
        
        try:
            data = json.loads(response)
            # Validate the response structure
            if "concepts" in data and isinstance(data["concepts"], list):
                # Clean and standardize the concepts
                concepts = [concept.strip().lower() for concept in data["concepts"]]
                # Update global pool
                global_concepts_pool.update(concepts)
                return concepts
            else:
                if debug:
                    print(f"Attempt {attempt + 1}: Validation failed. Response: {response}")
        except json.JSONDecodeError as e:
            if debug:
                print(f"Response was: {response}")
                print(f"Attempt {attempt + 1}: JSON decoding error: {e}")
        # Delay before retry
        time.sleep(2)
    
    # Return empty list if retries exhaust
    if debug:
        print("Failed to retrieve valid concepts after maximum retries.")
    return []

def add_concepts(data, start_index=0, max_try_num=3, model="gpt-4o-mini", debug=False):
    """Add extracted concepts to each question in the dataset starting from the given index using GPT."""
    print(f"Calling {model} to extract concepts...")
    for i, item in enumerate(tqdm(data[start_index:], desc="Adding concepts")):
        question_text = item.get('question', '')
        concepts = extract_concepts_with_gpt(question_text, max_try_num=max_try_num, model=model, debug=debug)
        item["concepts"] = concepts
    return data

def validate_metadata(metadata):
    """Validate the metadata structure against the expected schema."""
    if not isinstance(metadata, dict):
        return False
    for key, expected_type in expected_schema.items():
        if key not in metadata or not isinstance(metadata[key], expected_type):
            return False
        metadata[key] = [item.strip().lower() for item in metadata[key]]
    return True

def update_global_pools(metadata):
    """Update global pools with new categories from metadata."""
    for key, values in metadata.items():
        if key in global_pools:
            global_pools[key].update(values)

def extract_metadata_with_gpt(question_text, max_try_num=3, model="gpt-4o-mini", debug=False):
    """Query GPT to extract metadata with retries if validation fails."""
    
    prompt = f"""
    Given the following medical question, extract the following metadata and provide it in JSON format:

    1. Diseases involved: List all diseases or medical conditions mentioned or implied.
    2. Question types: Identify the types of questions involved (e.g., diagnostic, treatment, prognosis, prevention, etiology, epidemiology, anatomy, physiology).
    3. Medical specialties: List all relevant medical specialties related to the question.
    4. Severity and urgency: Assess the severity and urgency level (e.g., high, medium, low).
    5. Patient demographics: Extract any patient demographic information (e.g., age, gender).

    Answer in JSON format with keys: "diseases", "question_types", "medical_specialties", "severity_urgency", "patient_demographics". Each value should be a list of strings.

    Question:
    \"\"\"
    {question_text}
    \"\"\"
    """
    for attempt in range(max_try_num):
        response = utils.query_gpt(prompt, max_tokens=200, temperature=0, model=model, debug=debug, json=True)
        if response == -1:
            if debug:
                print(f"Attempt {attempt + 1}: GPT response failed.")
            time.sleep(1)
            continue
        
        try:
            metadata = json.loads(response)
            
            # Validate schema and format response if validation passes
            if validate_metadata(metadata):
                for key in metadata:
                    metadata[key] = [item.strip().lower() for item in metadata[key]]
                
                # Update global pools and return valid metadata
                update_global_pools(metadata)
                return metadata
            
            else:
                if debug:
                    print(f"Attempt {attempt + 1}: Validation failed. Response: {response}")
        
        except json.JSONDecodeError as e:
            if debug:
                print(f"Response was: {response}")
                print(f"Attempt {attempt + 1}: JSON decoding error: {e}")
        
        # Delay before retry
        time.sleep(2)
    
    # Return default structure if retries exhaust
    if debug:
        print("Failed to retrieve valid metadata after maximum retries.")
    return {key: [] for key in expected_schema}

def add_metadata(data, start_index=0, max_try_num=3, model="gpt-4o-mini", debug=False):
    """Add metadata to each question in the dataset starting from the given index using GPT."""
    print(f"Calling {model}...")
    for i, item in enumerate(tqdm(data[start_index:], desc="Adding metadata")):
        question_text = item.get('question', '')
        metadata = extract_metadata_with_gpt(question_text, max_try_num=max_try_num, model=model, debug=debug)
        item.update(metadata)
    return data

def save_dataset(data, input_filepath, date):
    """Save the augmented dataset to a JSONL file with '_with_category_metadata.jsonl' appended to the filename."""
    base_name = os.path.basename(input_filepath)
    dir_name = os.path.dirname(input_filepath)
    output_filename = f"{os.path.splitext(base_name)[0]}_with_concepts_metadata_{date}.jsonl"
    output_filepath = os.path.join(dir_name, output_filename)
    
    with open(output_filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"New dataset with metadata saved to {output_filepath}")

def save_global_pools(input_filepath, date):
    """Save global pools to a JSON file for record-keeping in the same directory as the dataset."""
    dir_name = os.path.dirname(input_filepath)
    output_filepath = os.path.join(dir_name, f"global_pools_{date}.json")
    global_pools_serializable = {k: list(v) for k, v in global_pools.items()}
    with open(output_filepath, 'w') as f:
        json.dump(global_pools_serializable, f, indent=2)
    print(f"Global pools saved to {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Add metadata to medical question dataset using GPT.')
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to evaluate (e.g., MedQA)")
    parser.add_argument("--max_questions", type=int, default=200, help="Max number of questions to evaluate")
    parser.add_argument("--question_start", type=int, default=0, help="Starting question index to skip already categorized questions")
    parser.add_argument('--max_try_num', type=int, default=3, help='Maximum number of GPT query attempts.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='GPT model to use.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    args = parser.parse_args()

    # Get dataset path from _DATASETS
    input_filepath = utils._DATASETS.get(args.dataset)
    if not input_filepath:
        print(f"Dataset '{args.dataset}' not found in _DATASETS.")
        return

    # Load the dataset
    data = utils.load_dataset(input_filepath, args.max_questions)

    # Add metadata
    data_with_metadata = add_concepts(data, start_index=args.question_start, max_try_num=args.max_try_num, model=args.model, debug=args.debug)

    # Save the new dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dataset(data_with_metadata, input_filepath, timestamp)
    
    # Save global pools of each metadata category
    save_global_pools(input_filepath, timestamp)

if __name__ == '__main__':
    main()