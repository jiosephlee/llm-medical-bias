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

# Save evaluation metrics to JSON
def save_metrics(metrics, output_dir, parameters):
    output_file = os.path.join(output_dir, f"{parameters}_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs' predictions on medical datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--start", type=int, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--end", type=int, required=True, help="Name of the dataset to evaluate")
    parser.add_argument("--parameters", type=str, required=True, help="Parameters of the predictions file")
    parser.add_argument("--ordinal", action="store_true", help="Enable quadratic kappa")
    parser.add_argument("--by_class", action="store_true", help="Enable class analysis")

    args = parser.parse_args()


    print("Loading Dataset...")
    filepath= utils._DATASETS[args.dataset]['test_set_filepath']
    format = utils._DATASETS[args.dataset]['format']
    dataset = utils.load_dataset(filepath, format, args.start, args.end)
    print(len(dataset))
    ground_truths = dataset[utils._DATASETS[args.dataset]['target']]
    output_dir = f"./results/{args.dataset}/"

    print("Loading Predictions...")
    predictions = utils.load_predictions(args.parameters, format='csv', save_path=output_dir)['Estimated_Acuity']

    # Evaluate
    metrics = utils.evaluate_predictions(predictions, ground_truths, ordinal = args.ordinal, by_class=args.by_class)
    print("Overall Metrics:", metrics)
    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, output_dir, f"{args.parameters}")
    
    print("Evaluation complete. Metrics and plots saved.")
    