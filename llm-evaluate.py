import json
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import utils
from collections import defaultdict

# Evaluate predictions with accuracy, precision, recall, and F1-score
def evaluate_predictions(predictions, ground_truths):
    correct_answers = [item['answer_idx'] for item in ground_truths]
    predicted_answers = [pred['predicted_answer'] for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(correct_answers, predicted_answers)
    precision = precision_score(correct_answers, predicted_answers, average='weighted')
    recall = recall_score(correct_answers, predicted_answers, average='weighted')
    f1 = f1_score(correct_answers, predicted_answers, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def evaluate_by_category(predictions, ground_truths, category_key):
    """Evaluate metrics for each unique label in the specified category."""

    category_metrics = {}
    # Group predictions and ground truths by category label
    grouped_data = defaultdict(list)
    

    for pred, truth in zip(predictions, ground_truths):
        # We grab several labels for each question category as a question may be multi-class
        labels = truth.get(category_key, [])
        for label in labels:
            grouped_data[label].append((pred, truth))
    
    # Calculate metrics for each label in the category
    for label, items in grouped_data.items():
        label_predictions = [item[0]['predicted_answer'] for item in items]
        label_ground_truths = [item[1]['answer_idx'] for item in items]
        
        # Calculate metrics for this specific label group
        accuracy = accuracy_score(label_ground_truths, label_predictions)
        precision = precision_score(label_ground_truths, label_predictions, average='weighted')
        recall = recall_score(label_ground_truths, label_predictions, average='weighted')
        f1 = f1_score(label_ground_truths, label_predictions, average='weighted')
        
        category_metrics[label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    return category_metrics

def plot_f1_scores_by_category(category_metrics, category_name, output_dir):
    """Plot F1 scores for each label in the specified category."""
    labels = list(category_metrics.keys())
    f1_scores = [metrics["f1_score"] for metrics in category_metrics.values()]
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, f1_scores)
    plt.xlabel("F1 Score")
    plt.title(f"F1 Scores by {category_name.capitalize()}")
    plt.tight_layout()
    plt.savefig(output_dir + f"_{category_name}_f1_scores.png")
    plt.close()

# Save evaluation metrics to JSON
def save_metrics(metrics, output_dir, timestamp):
    output_file = os.path.join(output_dir, f"{timestamp}_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs' predictions on medical datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp of the predictions file")
    parser.add_argument("--has_category_metadata", action="store_true", help="Enable evaluation by category metadata")
    parser.add_argument("--plot", action="store_true", help="Enable plotting F1 scores by category")
    args = parser.parse_args()

    # Get dataset with metadata & predictions
    if args.has_category_metadata:
        dataset_path = utils._DATASETS.get(args.dataset).split(".json")[0]+"_with_category_metadata.jsonl"
    else:
        dataset_path = utils._DATASETS.get(args.dataset)
    if not dataset_path:
        raise ValueError(f"Dataset {args.dataset} not found in paths.json.")

    print("Loading Dataset...")
    ground_truths = utils.load_dataset(dataset_path)
    
    print("Loading Predictions...")
    predictions = utils.load_predictions(args.timestamp)

    # Basic Evaluate
    metrics = evaluate_predictions(predictions, ground_truths)
    print("Overall Metrics:", metrics)

    output_dir = "./results/"
    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, output_dir, f"{args.timestamp}_{args.model}_{args.max_questions}_metrics")

    # Category-based evaluation
    if args.has_category_metadata:
        category_metrics = {}
        category_keys = ["high_impact_diseases", "question_types", "medical_specialties", "severity_urgency", "age","gender","ethnicity"]
        
        for category in category_keys:
            print(f"Evaluating by category: {category}")
            category_metrics[category] = evaluate_by_category(predictions, ground_truths, category)
            #print(f"Metrics by {category}:", category_metrics[category])
        
        # Save all category metrics in a single JSON file
        all_metrics_file = os.path.join(output_dir, f"{args.timestamp}_{args.model}_{args.max_questions}_category_metrics.json")
        with open(all_metrics_file, 'w') as f:
            json.dump(category_metrics, f, indent=2)
        
        # Plot F1-scores if --plot is enabled
        if args.plot:
            plot_output_dir = os.path.join(output_dir, "plots/", args.timestamp)
            for category, metrics in category_metrics.items():
                plot_f1_scores_by_category(metrics, category, plot_output_dir)
    
    print("Evaluation complete. Metrics and plots saved.")