import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import utils.utils as utils
import re

# Prediction strategies
class Predictor:
    """Base class for prediction strategy."""
    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug
    
    def predict(self, prompt):
        """Override this method in subclasses to perform prediction."""
        raise NotImplementedError("This method should be overridden in subclasses")

class StandardPredictor(Predictor):
    """Standard prediction without specific reasoning techniques."""
    def predict(self, prompt):
        response = utils.query_gpt(prompt + " Only output the letter.", model=self.model, debug=self.debug)
        return response
    
class CoTPredictor(Predictor):
    """Prediction using Chain of Thought reasoning."""
    def predict(self, prompt):
        cot_prompt = prompt + "Please explain your reasoning step-by-step before selecting the best answer. Then, output the letter alone in parenthesis."
        response = utils.query_gpt(cot_prompt, model=self.model, debug=self.debug)
        
        # Define a regex pattern to capture a letter inside parentheses at the end of the response
        match = re.search(r"\( *([A-D]) *\)", response.strip())
        
        if match:
            # If a valid answer letter is found within parentheses, return it in uppercase form
            return match.group(1).upper()
        else:
            # Handle cases where no valid answer letter is detected in the expected format
            if self.debug:
                print("Warning: No valid answer option found in the response. Response was:", response)
            return "Unknown"  # or handle as per your logic, e.g., retry or return a default

class USCPredictor(Predictor):
    """Prediction using Universal Self-Consistency (multiple predictions and majority vote)."""
    def __init__(self, model, num_trials=5, debug=False):
        super().__init__(model, debug)
        self.num_trials = num_trials

    def predict(self, prompt):
        responses = [utils.query_gpt(prompt, model=self.model, debug=self.debug) for _ in range(self.num_trials)]
        final_answers = [response.splitlines()[-1] for response in responses]
        return max(set(final_answers), key=final_answers.count)  # Majority vote


# Formatter for prompt
def format_prompt(question_text, answer_options):
    options_text = "\n".join([f"{key}: {value}" for key, value in answer_options.items()])
    prompt = f"Question: {question_text}\n\nOptions:\n{options_text}\n\nPlease select the best answer."
    return prompt

# Save predictions to .txt
def save_predictions(predictions, output_dir, model, num_of_questions, strategy, timestamp):
    output_file = os.path.join(output_dir, f"{timestamp}_{model}_{strategy}_{num_of_questions}_predictions.txt")
    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on medical QA dataset using LLMs.")
    parser.add_argument("--dataset", type=str, required=True, choices=utils._DATASETS.keys(), help="Name of the dataset to evaluate")
    parser.add_argument("--max_questions", type=int, default=200, help="Max number of questions to evaluate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="GPT model to use.")
    parser.add_argument("--strategy", type=str, choices=["standard", "cot", "usc"], default="standard", help="Prediction strategy to use")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for USC strategy")
    parser.add_argument("--load_predictions", type=str, help="Timestamp & model_name & # of questions to existing predictions to load")
    args = parser.parse_args()

    # Get dataset filepath
    print("Loading Dataset...")
    dataset_path = utils._DATASETS.get(args.dataset)
    if not dataset_path:
        raise ValueError(f"Dataset {args.dataset} not found in _DATASETS.")

    # Load dataset
    dataset = utils.load_jsonl(dataset_path, max_questions=args.max_questions)
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

    # Instantiate predictor based on selected strategy
    if args.strategy == "standard":
        predictor = StandardPredictor(model=args.model, debug=args.debug)
    elif args.strategy == "cot":
        predictor = CoTPredictor(model=args.model, debug=args.debug)
    elif args.strategy == "usc":
        predictor = USCPredictor(model=args.model, num_trials=args.num_trials, debug=args.debug)

    if num_new_predictions_needed > 0:
        print(f"Making {num_new_predictions_needed} new predictions...")
        dataset_to_predict = dataset[num_existing_predictions:num_existing_predictions + num_new_predictions_needed]

        for item in tqdm(dataset_to_predict):
            question_text = item['question']
            answer_options = item['options']
            correct_answer = item.get('answer_idx', None)

            prompt = format_prompt(question_text, answer_options)
            prediction = predictor.predict(prompt)

            prediction_entry = {
                "question": question_text,
                "predicted_answer": prediction.strip().replace(".", ""),
                "correct_answer": correct_answer.strip().replace(".", "") if correct_answer else None
            }
            predictions.append(prediction_entry)
    else:
        print("No new predictions needed.")

    # Save combined predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "./results/"
    os.makedirs(save_path, exist_ok=True)
    save_predictions(predictions, save_path, args.model, args.max_questions, args.strategy, timestamp)

    print("Processing complete. Predictions saved.")