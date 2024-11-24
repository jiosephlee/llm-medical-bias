from utils.config import OPENAI_API_KEY, TOGETHER_API_KEY, DATABRICKS_TOKEN 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from openai import OpenAI
import pandas as pd
from together import Together
import os
import time 
import json 
import marshal

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

client = OpenAI()
client_tog = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

client_safe = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
)

def format_prompt_triage(row):
    return """f"""

# We store unanimously required & unique characteristics of the datasets here. Otherwise, edge cases will be handled elsewhere.
_DATASETS = {
    "MedQA": 
        {'filepath': "./data/medqa/questions/US/4_options/phrases_no_exclude_test.jsonl",
        'format': 'jsonl',
        'hippa': False,
        },
    "Triage-Public": 
        {'filepath': "./data/mimic-iv-public/triage_public.csv",
         'format': 'csv',
         'target': 'acuity',
         'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
         'hippa': False,
        },
    "Triage-Counterfactual": 
        {'filepath': "./data/mimic-iv-public/triage_counterfactual.csv",
        'format': 'csv',
        'target': 'acuity',
        'hippa': False,
        'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
        },
    "Triage-Private": 
        {'filepath': "./data/mimic-iv-private/triage.csv",
        'format': 'csv',
        'target': 'acuity',
        'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
        'hippa': True,
        },
    "Triage-Private-Stratified": 
        {'filepath': "./data/mimic-iv-private/triage_stratified_2500.csv",
         'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
        'format': 'csv',
        'target': 'acuity',
        'hippa': True,
        },
    }
        
def load_dataset(filepath, format, start_index, end_index):
    if not filepath:
        raise ValueError("Dataset not found in _DATASETS.")
    if format == 'jsonl':
        data = load_jsonl(filepath, start_index, end_index)
    elif format == 'csv':
        dataset = pd.read_csv(filepath).loc[start_index:end_index]
        data = dataset.dropna()
    else:
        raise ValueError(f"Unsupported format: {format}")
    return data
    
def load_jsonl(filepath, start_index, end_index):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if i <= end_index and i >= start_index]
    return data

def load_predictions(filename, save_path="./results/"):
    predictions_file = os.path.join(save_path, f"{filename}_predictions.txt")
    with open(predictions_file, 'r') as f:
        predictions = [json.loads(line.strip()) for line in f]
    return predictions

def stratified_df(df, target_col, test_size, seed=0):
    # Define stratified shuffle split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # Get the stratified sample indices
    for train_indices, test_indices in stratified_split.split(df, df[target_col]):
        stratified_train_df = df.iloc[train_indices]
        stratified_test_df = df.iloc[test_indices]
    
    return stratified_train_df.reset_index(drop=True), stratified_test_df.reset_index(drop=True)

def evaluate_predictions(predicted_answers, correct_answers, ordinal=False,flexibility=1, by_class=False):
    """
    Evaluate predictions with standard metrics, Quadratic Weighted Kappa, and optionally per-class analysis.

    Args:
        predicted_answers (list): List of predicted class labels.
        correct_answers (list): List of true class labels.
        flexibility (int): Tolerance for flexibility in metrics (e.g., ±1).
        by_class (bool): Whether to compute metrics by class.

    Returns:
        dict: A dictionary containing overall and optionally per-class evaluation metrics.
    """
    # Standard metrics
    accuracy = accuracy_score(correct_answers, predicted_answers)
    precision = precision_score(correct_answers, predicted_answers, average='weighted')
    recall = recall_score(correct_answers, predicted_answers, average='weighted')
    f1 = f1_score(correct_answers, predicted_answers, average='weighted')
    
    # Adjust predictions for flexibility
    adjusted_predictions = [
        true if abs(true - pred) <= flexibility else pred
        for true, pred in zip(correct_answers, predicted_answers)
    ]

    adjusted_accuracy = accuracy_score(correct_answers, adjusted_predictions)
    adjusted_precision = precision_score(correct_answers, adjusted_predictions, average='weighted')
    adjusted_recall = recall_score(correct_answers, adjusted_predictions, average='weighted')
    adjusted_f1 = f1_score(correct_answers, adjusted_predictions, average='weighted')

    # Metrics by class
    class_metrics = defaultdict(dict)
    if by_class:
        unique_classes = set(correct_answers)
        for cls in unique_classes:
            # Filter predictions and correct answers for the current class
            cls_indices = [i for i, true in enumerate(correct_answers) if true == cls]
            cls_correct = [correct_answers[i] for i in cls_indices]
            cls_predicted = [predicted_answers[i] for i in cls_indices]
            
            if cls_correct and cls_predicted:
                class_metrics[cls] = {
                    "accuracy": accuracy_score(cls_correct, cls_predicted),
                    "precision": precision_score(cls_correct, cls_predicted, average='binary', zero_division=0),
                    "recall": recall_score(cls_correct, cls_predicted, average='binary', zero_division=0),
                    "f1_score": f1_score(cls_correct, cls_predicted, average='binary', zero_division=0)
                }

    # Consolidate results
    results = {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "adjusted_accuracy": adjusted_accuracy,
            "adjusted_precision": adjusted_precision,
            "adjusted_recall": adjusted_recall,
            "adjusted_f1": adjusted_f1,
        }
    }
    if ordinal:
        qwk = cohen_kappa_score(correct_answers, predicted_answers, weights="quadratic")
        results['overall']["quadratic_kappa"] = qwk
    if by_class:
        results["by_class"] = class_metrics

    return results

def query_gpt_safe(prompt, model="openai-gpt-4o-chat", return_json=False,temperature=0.0, max_tokens=1000, debug=False):
    if debug:
        print(prompt)
    if return_json:
        response = client_safe.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0,
            response_format={"type": "json_object"}
        )
    else:
        response = client_safe.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0,
        )
    response = response.choices[0].message.content.strip()
    if debug:
        print(response)
    return response

def query_gpt(prompt, max_tokens=100, temperature=0, top_p = 0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, logprobs=False):
    if debug:
        print(prompt)
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            # Together API
            if 'gpt' not in model and 'o1' not in model:
                response = client_tog.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=1,
                repetition_penalty=1,
                stop=["<|eot_id|>","<|eom_id|>"],
                )
            else:
                if return_json:
                    response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"}
                    )  
                else:
                    if logprobs:
                        response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        top_p = top_p,
                        logprobs=logprobs,
                        top_logprobs=3,
                        seed = 0
                        )   
                    else: 
                        response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        top_p = top_p,
                        seed = 0)
            if debug:
                print(response.choices[0].message.content.strip())
            if logprobs:
                return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response.choices[0].message.content.strip()
        except Exception as e:
            if 'gpt' in model:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making Together API call: {e}")
            curr_try_num += 1
            if curr_try_num >= max_try_num:
                return (-1)
            time.sleep(10)
            
