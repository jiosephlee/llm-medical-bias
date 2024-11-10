import os
import json
import argparse
import time
from tqdm import tqdm
import utils

# Predefined general categories
general_categories = {
    "diseases": [
        "Cardiovascular",
        "Respiratory",
        "Gastrointestinal",
        "Endocrine/Metabolic",
        "Infectious Diseases",
        "Neurological",
        "Musculoskeletal",
        "Oncological",
        "Mental Health",
        "Dermatological",
        "Hematological",
        "Renal/Urinary",
        "Reproductive Health",
        "Autoimmune",
        "Other"
    ],
    "high_impact_diseases": [
        "Cardiovascular Disease",
        "Hypertension",
        "Coronary Artery Disease",
        "Myocardial Infarction (Heart Attack)",
        "Stroke",
        "Chronic Obstructive Pulmonary Disease (COPD)",
        "Asthma",
        "Type 2 Diabetes Mellitus",
        "Chronic Kidney Disease",
        "Liver Cirrhosis",
        "Hepatitis C",
        "Sepsis",
        "Breast Cancer",
        "Lung Cancer",
        "Colorectal Cancer",
        "Prostate Cancer",
        "Alzheimer's Disease",
        "Parkinson's Disease",
        "Major Depression",
        "Schizophrenia",
        "Osteoarthritis",
        "Rheumatoid Arthritis",
        "Human Immunodeficiency Virus (HIV)",
        "Tuberculosis",
        "Influenza",
        "Obesity",
        "Anemia",
        "Dementia",
        "Congestive Heart Failure",
        "Atrial Fibrillation",
        "Acute Kidney Injury",
        "Other"
    ],
    "question_types": [
        "Diagnostic",
        "Treatment",
        "Prognosis",
        "Prevention",
        "Etiology",
        "Pathophysiology",
        "Adverse Effects",
        "Epidemiology",
        "Anatomy & Physiology",
        "Other"
    ],
    "medical_specialties": [
        "Cardiology",
        "Pulmonology",
        "Gastroenterology",
        "Endocrinology",
        "Neurology",
        "Oncology",
        "Psychiatry",
        "Dermatology",
        "Nephrology",
        "Obstetrics & Gynecology",
        "Pediatrics",
        "Surgery",
        "Emergency Medicine",
        "General Medicine",
        "Other"
    ],
    "severity_urgency": [
        "High",
        "Moderate",
        "Low"
    ],
    "age": [
        "Infants (0-1 year)",
        "Children (2-12 years)",
        "Adolescents (13-18 years)",
        "Young Adults (19-40 years)",
        "Adults (41-64 years)",
        "Elderly (65+ years)"
    ],
    "gender": [
        "Male",
        "Female",
        "Other"
    ],
    "ethnicity": [
        "Caucasian",
        "African American",
        "Asian",
        "Hispanic",
        "Other"
    ]
}

def remap_labels(data, general_categories, max_try_num=3, model="gpt-4o-mini", debug=False):
    """Remap existing specific labels for each question to general category labels using GPT."""
    for item in tqdm(data, desc="Remapping labels"):
        # Collect existing metadata labels for the item
        metadata_labels = {}
        for key in ['diseases', 'question_types', 'medical_specialties', 'severity_urgency', 'patient_demographics']:
            metadata_labels[key] = item.get(key, [])

        # Prepare the prompt
        prompt = f"""
Given the following specific metadata labels for a medical question, remap them to the predefined general categories. Some specific labels may map to multiple general categories.

Specific Metadata Labels:
"""
        # Add the metadata labels to the prompt
        for category, labels in metadata_labels.items():
            labels_text = ', '.join(labels) if labels else 'None'
            prompt += f"- {category}: {labels_text}\n"

        prompt += "\nPredefined General Categories:\n"

        # Add the general categories to the prompt
        for category, general_labels in general_categories.items():
            labels_text = ', '.join(general_labels)
            prompt += f"- {category}: {labels_text}\n"

        prompt += """
Please map the specific labels to the general categories and provide the result in JSON format.

The JSON format should have keys corresponding to the categories, and the values should be lists of the general labels that the specific labels map to.

Example Output:
{
  "diseases": ["Cardiovascular", "Endocrine/Metabolic"],
  "high_impact_diseases": ["Hypertension", "Type 2 Diabetes Mellitus"],
  "question_types": ["Treatment", "Etiology"],
  "medical_specialties": ["Cardiology", "Endocrinology"],
  "severity_urgency": ["High"],
  "age": ["Adults (41-64 years)"],
  "gender": ["Male"],
  "ethnicity": ["African American"]
}
"""

        # Query the GPT model
        for attempt in range(max_try_num):
            response = utils.query_gpt(prompt, max_tokens=500, temperature=0, model=model, debug=debug)
            if response == -1:
                if debug:
                    print(f"Attempt {attempt + 1}: GPT response failed.")
                time.sleep(1)
                continue

            try:
                remapped_metadata = json.loads(response)
                # Validate the remapped metadata
                if validate_remapped_metadata(remapped_metadata, general_categories):
                    # Update the item with the remapped metadata
                    for category in general_categories.keys():
                        item[category] = remapped_metadata.get(category, [])
                    break  # Break out of the retry loop
                else:
                    if debug:
                        print(f"Attempt {attempt + 1}: Invalid remapped metadata. Response: {response}")
            except json.JSONDecodeError as e:
                if debug:
                    print(f"Attempt {attempt + 1}: JSON decoding error: {e}. Response: {response}")

            time.sleep(1)
        else:
            if debug:
                print(f"Failed to remap labels for item after {max_try_num} attempts.")
            # If failed after retries, assign empty lists
            for category in general_categories.keys():
                item[category] = []

    return data

def validate_remapped_metadata(metadata, general_categories):
    """Validate that the remapped metadata has the correct structure and values."""
    if not isinstance(metadata, dict):
        return False
    for category in general_categories.keys():
        if category not in metadata:
            return False
        if not isinstance(metadata[category], list):
            return False
        # Check that each label in the list is among the general labels
        for label in metadata[category]:
            if label not in general_categories[category]:
                return False
    return True

def save_dataset(data, input_filepath):
    """Save the remapped dataset to a JSONL file with '_remapped_labels.jsonl' appended to the filename."""
    base_name = os.path.basename(input_filepath)
    dir_name = os.path.dirname(input_filepath)
    output_filename = f"{os.path.splitext(base_name)[0]}_remapped_labels.jsonl"
    output_filepath = os.path.join(dir_name, output_filename)
    
    with open(output_filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Remapped dataset saved to {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Remap existing labels to general categories using GPT.')
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to remap labels for (e.g., MedQA)")
    parser.add_argument("--max_questions", type=int, default=200, help="Max number of questions to evaluate")
    parser.add_argument('--max_try_num', type=int, default=3, help='Maximum number of GPT query attempts.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='GPT model to use.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    args = parser.parse_args()

    # Get dataset path from _DATASETS, the one with metadata
    input_filepath = utils._DATASETS.get(args.dataset).split(".jsonl")[0]+ "_with_category_metadata.jsonl"
    if not input_filepath:
        print(f"Dataset '{args.dataset}' not found in _DATASETS.")
        return

    # Load the dataset
    data = utils.load_dataset(input_filepath, args.max_questions)

    # Remap labels in the dataset
    data_with_remapped_labels = remap_labels(data, general_categories, max_try_num=args.max_try_num, model=args.model, debug=args.debug)

    # Save the remapped dataset
    save_dataset(data_with_remapped_labels, input_filepath)

if __name__ == '__main__':
    main()