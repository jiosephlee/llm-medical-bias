import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import utils

def format_prompt(question_text, answer_options):
    options_text = "\n".join([f"{key}: {value}" for key, value in answer_options.items()])
    prompt = f"Question: {question_text}\n\nOptions:\n{options_text}\n\nPlease select the best answer. Only output the letter."
    return prompt

# Save predictions to .txt
def save_predictions(predictions, output_dir, model, num_of_questions,timestamp):
    output_file = os.path.join(output_dir, f"{timestamp}_{model}_{num_of_questions}_predictions.txt")
    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on medical QA dataset using LLMs.")
    parser.add_argument("--dataset", type=str, required=True, choices=utils._DATASETS.keys(), help="Name of the dataset to evaluate")
    parser.add_argument("--max_questions", type=int, default=200, help="Max number of questions to evaluate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='GPT model to use.')
    parser.add_argument("--load_predictions", type=str, help="Timestamp & model_name & # of questions to existing predictions to load")
    args = parser.parse_args()

    # Get dataset filepath
    print("Loading Dataset...")
    dataset_path = utils._DATASETS.get(args.dataset)
    if not dataset_path:
        raise ValueError(f"Dataset {args.dataset} not found in _DATASETS.")

    # Load dataset
    dataset = utils.load_dataset(dataset_path, max_questions=args.max_questions)
    predictions = []
    
    if args.load_predictions:
        # Load existing predictions
        print("Loading existing predictions...")
        predictions = utils.load_predictions(args.load_predictions)
        num_existing_predictions = len(predictions)
        print(f"Loaded {num_existing_predictions} existing predictions.")
    else:
        num_existing_predictions = 0

    # Determine how many more predictions we need to make
    total_predictions_needed = args.max_questions
    num_new_predictions_needed = total_predictions_needed - num_existing_predictions

    if num_new_predictions_needed > 0:
        print(f"Making {num_new_predictions_needed} new predictions...")
        # Skip the first num_existing_predictions in the dataset
        dataset_to_predict = dataset[num_existing_predictions:num_existing_predictions + num_new_predictions_needed]

        for item in tqdm(dataset_to_predict):
            question_text = item['question']
            answer_options = item['options']
            correct_answer = item.get('answer_idx', None)  # Include if needed

            prediction = utils.query_gpt(format_prompt(question_text, answer_options), model=args.model, debug=args.debug)
            prediction_entry = {
                "question": question_text,
                "predicted_answer": prediction.strip().replace('.',''),
                "correct_answer": correct_answer.strip().replace('.','')
            }
            predictions.append(prediction_entry)
    else:
        print("No new predictions needed.")

    # Save combined predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = "./results/"
    os.makedirs(save_path, exist_ok=True)
    save_predictions(predictions, save_path, args.model, args.max_questions, timestamp)

    print("Processing complete. Predictions saved.")