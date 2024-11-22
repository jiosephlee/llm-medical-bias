import json
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import utils.utils as utils
from collections import defaultdict
import numpy as np
import time
import requests

def get_pubmed_article_count(disease_term):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": disease_term,
        "retmode": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    time.sleep(0.2)
    return int(data["esearchresult"]["count"])

def get_bing_search_count(query):
    url = "https://api.bing.microsoft.com/v7.0/search"
    api_key = "28f95a9243864997a126f5fd6a4cdb6d"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    time.sleep(0.35)
    return data.get("webPages", {}).get("totalEstimatedMatches", 0)

def evaluate_and_analyze_correlation(category_metrics, category_name, metric_name, output_dir):
    """
    Analyze correlations between article counts and their metrics for a given category.
    Plot the correlations and save the plots.

    Args:
        category_metrics (dict): Dictionary with label names as keys and metric dictionaries as values.
        category_name (str): Name of the category for title and output file naming.
        metric_name (str): The metric to correlate with article counts (e.g., F1-score or accuracy).
        output_dir (str): Directory to save the plots.
    """
    labels = list(category_metrics.keys())
    metric_values = [metrics[metric_name] for metrics in category_metrics.values()]
    
    # Sort labels and metric values by metric values
    sorted_pairs = sorted(zip(metric_values, labels))
    sorted_metric_values, sorted_labels = zip(*sorted_pairs)
    
    # Get article counts for each label using the PubMed API
    article_counts = np.array([get_bing_search_count(label) for label in sorted_labels])
    print(article_counts)
    # Compute correlation between article counts and the metric values
    if len(article_counts) > 1 and len(sorted_metric_values) > 1:  # Ensure sufficient data for correlation
        correlation = np.corrcoef(article_counts, sorted_metric_values)[0, 1]
        print(f"Correlation between article count and {metric_name} for {category_name}: {correlation:.2f}")
    else:
        print(f"Not enough data to compute correlation for {category_name}.")
        correlation = np.nan
    
    # Plot correlation
    plt.figure(figsize=(12, 6))
    plt.scatter(article_counts, sorted_metric_values, alpha=0.7)
    plt.xlabel("Article Count")
    plt.xlim(0, max(article_counts) * 1.1)  # Add 10% margin to the maximum count
    plt.ylabel(metric_name.replace("_", " ").capitalize())
    plt.title(f"Correlation: {correlation:.2f} ({category_name.capitalize()})")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{category_name}_correlation.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

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

def evaluate_by_category(predictions, ground_truths, category_key, min_num_per_category=1):
    """Evaluate metrics for each unique label in the specified category."""

    category_metrics = {}
    # Group predictions and ground truths by category label
    grouped_data = defaultdict(list)

    for pred, truth in zip(predictions, ground_truths):
        # We grab several labels for each question category as a question may be multi-class
        
        labels = truth.get(category_key, [])
        for label in labels:
            grouped_data[label].append((pred, truth))
    
    # Filter out labels with fewer than min_num_per_category elements
    grouped_data = {label: items for label, items in grouped_data.items() if len(items) >= min_num_per_category}

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

# def plot_metric_by_category(category_metrics, category_name, metric_name, output_dir):
#     """
#     Plot specified metric for each label in the given category, sorted by metric value.

#     Args:
#         category_metrics (dict): Dictionary with label names as keys and metric dictionaries as values.
#         category_name (str): Name of the category for title and output file naming.
#         metric_name (str): The metric to plot (e.g., "f1_score" or "accuracy").
#         output_dir (str): Directory to save the plot.
#     """
#     labels = list(category_metrics.keys())
#     metric_values = [metrics[metric_name] for metrics in category_metrics.values()]
    
#     # Sort labels and metric values by metric values
#     sorted_pairs = sorted(zip(metric_values, labels))
#     sorted_metric_values, sorted_labels = zip(*sorted_pairs)
    
#     plt.figure(figsize=(10, 6))
#     plt.barh(sorted_labels, sorted_metric_values)
#     plt.xlabel(metric_name.replace("_", " ").capitalize())
#     plt.title(f"{metric_name.replace('_', ' ').capitalize()} by {category_name.capitalize()}")
#     plt.tight_layout()
    
#     # Create output directory if it does not exist
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"{category_name}_{metric_name}.png"))
#     plt.close()

def plot_metric_by_category(category_metrics, category_name, metric_name, output_dir):
    """
    Plot specified metric for each label in the given category, sorted by metric value.

    Args:
        category_metrics (dict): Dictionary with label names as keys and metric dictionaries as values.
        category_name (str): Name of the category for title and output file naming.
        metric_name (str): The metric to plot (e.g., "f1_score" or "accuracy").
        output_dir (str): Directory to save the plot.
    """
    labels = list(category_metrics.keys())
    metric_values = [metrics[metric_name] for metrics in category_metrics.values()]
    
    # Sort labels and metric values by metric values
    sorted_pairs = sorted(zip(metric_values, labels))
    sorted_metric_values, sorted_labels = zip(*sorted_pairs)
    
    # Dynamically adjust figure size based on number of categories
    num_categories = len(sorted_labels)
    height_per_category = 0.15  # Adjust this value based on how spaced out you want the bars
    figure_height = max(5, num_categories * height_per_category)  # Ensure a minimum height for small plots
    
    plt.figure(figsize=(10, figure_height))
    plt.barh(sorted_labels, sorted_metric_values)
    plt.xlabel(metric_name.replace("_", " ").capitalize())
    plt.title(f"{metric_name.replace('_', ' ').capitalize()} by {category_name.capitalize()}")
    plt.tight_layout()
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{category_name}_{metric_name}.png"))
    plt.close()

# Save evaluation metrics to JSON
def save_metrics(metrics, output_dir, timestamp):
    output_file = os.path.join(output_dir, f"{timestamp}_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs' predictions on medical datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--parameters", type=str, required=True, help="Parameters of the predictions file")
    parser.add_argument("--metadata_file_name", help="Enable evaluation by category metadata")
    parser.add_argument("--metadata", nargs="+", type=str, help="List of metadata categories to analyze")
    parser.add_argument("--plot", action="store_true", help="Enable plotting F1 scores by category")
    parser.add_argument("--analyze_correlation", action="store_true", help="Analyze correlation between term frequency and F1-score")

    args = parser.parse_args()

    # Get dataset with metadata & predictions
    if args.metadata_file_name:
        dataset_path = utils._DATASETS.get(args.dataset).split(".json")[0]+f"_{args.metadata_file_name}.jsonl"
    else:
        dataset_path = utils._DATASETS.get(args.dataset)
    if not dataset_path:
        raise ValueError(f"Dataset {args.dataset} not found in paths.json.")

    print("Loading Dataset...")
    ground_truths = utils.load_dataset(dataset_path, max_questions=int(args.parameters.split('_')[-1]))
    
    print("Loading Predictions...")
    predictions = utils.load_predictions(args.parameters)

    # Basic Evaluate
    metrics = evaluate_predictions(predictions, ground_truths)
    print("Overall Metrics:", metrics)

    output_dir = "./results/"
    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, output_dir, f"{args.parameters}")

    # Category-based evaluation
    if args.metadata_file_name:
        if args.metadata:
            category_keys = args.metadata
        else:
            raise Exception("No metadata categories provided.")
        category_metrics = {}
        for category in category_keys:
            print(f"Evaluating by category: {category}")
            category_metrics[category] = evaluate_by_category(predictions, ground_truths, category, min_num_per_category=10)
            #print(f"Metrics by {category}:", category_metrics[category])
        
        # Save all category metrics in a single JSON file
        all_metrics_file = os.path.join(output_dir, f"{args.parameters}_category_metrics.json")
        with open(all_metrics_file, 'w') as f:
            json.dump(category_metrics, f, indent=2)
        
        # Plot F1-scores if --plot is enabled
        if args.plot:
            plot_output_dir = os.path.join(output_dir, "plots/", args.parameters)
            for category, metrics in category_metrics.items():
                plot_metric_by_category(metrics, category, "f1_score", plot_output_dir)
                
        # Analyze correlation if enabled
        if args.analyze_correlation:
            for category, metrics in category_metrics.items():
                evaluate_and_analyze_correlation(metrics, category, "f1_score", plot_output_dir)
    
    print("Evaluation complete. Metrics and plots saved.")