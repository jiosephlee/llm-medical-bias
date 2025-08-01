import re
import pandas as pd
import json
from tqdm import tqdm
import utils.predictors as predictors
import argparse
import utils.utils as utils
import utils.prompts as prompts
from datetime import datetime
import time
import os
  
def extract_num(answer_text):
    # Use regex to find all floats and integers in the answer text
    matches = re.findall(r'\d+\.\d+|\d+', answer_text)
    
    # Convert all matches to floats for consistent processing
    matches = [float(num) for num in matches]

    if len(matches) == 1:
        return matches[0]  # Return the single found number
    elif len(matches) > 1:
        return f"Error: Multiple numbers found. Please verify the data: {answer_text}"
    else:
        return "Error: No acuity number found" 
    
def extract_acuity_from_text(text, debug):
    # Call another model to extract the acuity if necessary
    # Split the text into lines
    lines = text.splitlines()
    last_five_lines = lines[-5:]
    text = "\n".join(last_five_lines)
    answer_text = utils.query_llm(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", debug=debug)
    num = extract_num(answer_text)
    time.sleep(1)
    if type(num) == str and 'Error' in num:
        return utils.query_llm(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", debug=debug)
    else:
        return num
    
# def create_prompt_bias(row,strategy=None, return_json=False, vitals_off=False):
#     task_description = "Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority"
#     starting_prompt = f"Here is the profile of a {row['Race']} {row['Sex']} patient" 
#     cot, reasoning = "", ""
#     if strategy=='CoT':
#         cot, reasoning = "Let's think step by step", "your step-by-step reasoning in the key 'reasoning' and "
#     if vitals_off:
#         serialization_prompt=f"""pain   chiefcomplaint
# {row['pain']}   {row['chiefcomplaint']}"""
#     else:
#         serialization_prompt=f"""temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint
# {row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}"""
#     if return_json:
#         return f"""{starting_prompt}:

# {serialization_prompt}

# Estimate their acuity from 1 to 5 based on the following guidelines: {task_description}. {cot}

# Answer in valid JSON format, providing {reasoning}acuity as a single numeric value in the key 'acuity'."""
#     else:    
#         return f"""{starting_prompt}:

# {serialization_prompt}

# Estimate their acuity from 1-5 based on the following guidelines: {task_description}. {cot}
#         """
        
def predict(*, index, row, predictor, model, strategy, return_json, k_shots, serialization, vitals_off, bias, debug=False):
    prompt, response = predictor.predict(
        row=row,
        model=model,
        k_shots=k_shots,
        return_json=return_json,
        serialization_strategy=serialization,
        vitals_off=vitals_off,
        bias=bias,
        debug=debug)
    if return_json:
        try:
            response_data = json.loads(response)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            print("Raw response causing error:", response)
            response = utils.query_llm(response + "\n\nCan you format the above in proper JSON", model ='gpt-4o-mini',return_json=True)
            response_data = json.loads(response)
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data['Reasoning'] if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data['Reasoning'] if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
    else: 
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": extract_acuity_from_text(response, debug=debug),
                "Reasoning": response if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": extract_acuity_from_text(response, debug=debug),
                "Reasoning": response if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        
def save_csv(df, savepath, model, predictive_strategy, serialization,json_param, detailed_instructions,  start_index, end_index, timestamp, k_shots=None, k_shots_ablation=False):
    output_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        output_filepath = output_filepath + "_json"
    if detailed_instructions:
        output_filepath = output_filepath + "_detailed"
    if 'FewShot' in predictive_strategy or 'KATE' in predictive_strategy:
        output_filepath = output_filepath + f"_{k_shots}_shots"
    if k_shots_ablation:
        output_filepath = output_filepath + "_ablation"
    output_filepath = output_filepath + f"_{serialization}_{start_index}_{end_index}_{timestamp}.csv"
    # Save the DataFrame to a CSV file
    df.to_csv(output_filepath, index=False)
    print(f"DataFrame saved to {output_filepath}")
    return output_filepath
    
def load_csv(savepath, model, predictive_strategy, serialization,  json_param, detailed_instructions, start_index, end_index, timestamp, k_shots =None, k_shots_ablation=False):
    input_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        input_filepath = input_filepath + "_json"
    if detailed_instructions:
        input_filepath = input_filepath + "_detailed"
    if k_shots is not None:
        input_filepath = input_filepath + f"_{k_shots}"
    if k_shots_ablation:
        input_filepath = input_filepath + "_ablation"
    input_filepath = input_filepath + f"_{serialization}_{start_index}_{end_index}_{timestamp}.csv"
    
    try:
        df = pd.read_csv(input_filepath)
        print(f"DataFrame loaded from {input_filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {input_filepath}")
        return None
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions on medical QA dataset using LLMs.")
    parser.add_argument("--dataset", type=str, required=True, choices=utils._DATASETS.keys(), help="Name of the dataset to evaluate")
    parser.add_argument("--start", default=0,type=int, help="Start index of the samples to evaluate")
    parser.add_argument("--end", default=1000, type=int, help="End index of the samples to evaluate")
    parser.add_argument("--model", required=True, type=str, default="gpt-4o-mini", help="LLM model to use.")
    parser.add_argument("--strategy", required=True, type=str, choices=prompts.INSTRUCTIONS['triage-mimic'].keys(), default="standard", help="Prediction strategy to use")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for USC strategy")
    parser.add_argument("--k_shots", type=int, default=5, help="Number of shots for Few-Shot")
    parser.add_argument("--load_predictions", type=str, help="Timestamp & model_name & # of questions to existing predictions to load")
    parser.add_argument("--json", action="store_true", help="Turns on internal usage of json formats for the LLM API")
    parser.add_argument("--detailed_instructions", action="store_true", help="Turns on detailed instructions")
    parser.add_argument("--bias", action="store_true", help="Enables bias prompt")
    parser.add_argument("--serialization", default='natural',type=str, choices=["spaces", 'commas', "newline","json","natural","natural_sex_race","natural_full"], help="serialization prompt to use")
    parser.add_argument("--vitals_off", action="store_true", help="Turns on vitals off ablation prompt")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--k_shots_ablation", action="store_true", help="Change the parameters for k_shots ablation")

    args = parser.parse_args()

    print("Loading Dataset...")
    test_filepath= utils._DATASETS[args.dataset]['test_set_filepath']
    format = utils._DATASETS[args.dataset]['format']
    test_dataset = utils.load_dataset(test_filepath, format, args.start, args.end)

    print("Potentially loading existing predictions...")
    predictions = []
    if args.load_predictions:
        predictions = utils.load_predictions(args.load_predictions, format=utils._DATASETS[args.dataset]['format'], save_path="./results/checkpoints/" + args.dataset + '/')
        predictions = predictions.to_dict('records')
        num_existing_predictions = len(predictions)
        print(f"Loaded {num_existing_predictions} existing predictions.")
    else:
        num_existing_predictions = 0
    num_new_predictions_needed = (args.end - args.start) + 1 - num_existing_predictions
    
    if args.dataset == 'Triage-Counterfactual':
        args.k_shots = False
    if args.k_shots:
        if args.k_shots_ablation:
            training_df = utils.load_dataset(utils._DATASETS[args.dataset]['full_training_set_filepath'], format, 0, 1000000)
        else:
            training_df = utils.load_dataset(utils._DATASETS[args.dataset]['training_set_filepath'], format, 0, 1000000)

    print(f"Making {num_new_predictions_needed} new predictions...")
    predictor = predictors.Predictor(args.dataset,
                                     strategy=args.strategy, 
                                     hippa=utils._DATASETS[args.dataset]['hippa'],
                                     target = utils._DATASETS[args.dataset]['target'],
                                     training_set= training_df if args.k_shots else None,
                                     k_shots_ablation=args.k_shots_ablation)

    # Initialize counter for saving progress
    new_predictions_since_last_save = 0
    total_predictions_made = num_existing_predictions

    if num_new_predictions_needed > 0:
        if utils._DATASETS[args.dataset]['format'] == 'csv':
            for i, row in tqdm(test_dataset.loc[num_existing_predictions:args.end].iterrows()):
                # Prompting and LLM Logic is all handled here
                prediction = predict(
                    index=i,
                    row=row,
                    predictor=predictor,
                    model=args.model,
                    strategy=args.strategy,
                    return_json=args.json,
                    k_shots=args.k_shots,
                    serialization=args.serialization,
                    vitals_off=args.vitals_off,
                    bias=args.bias,
                    debug=args.debug
                )
                predictions.append(prediction)
                new_predictions_since_last_save += 1
                total_predictions_made += 1

                if new_predictions_since_last_save >= 250:
                    # Save predictions to disk
                    predictions_df = pd.DataFrame(predictions)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    save_path = "./results/checkpoints/" + args.dataset + '/'
                    os.makedirs(save_path, exist_ok=True)
                    save_path = save_path + args.dataset
                    save_csv(predictions_df, save_path, 
                             args.model, 
                             args.strategy, 
                             args.serialization,
                             args.json, 
                             args.detailed_instructions,
                             args.start,
                             args.start + total_predictions_made , 
                             timestamp, k_shots=args.k_shots)
                    new_predictions_since_last_save = 0
                    print(f"Saved progress after {len(predictions)} predictions.")
            predictions_df = pd.DataFrame(predictions)
    else:
        print("No new predictions needed.")

    # Save combined predictions one last time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = "./results/" + args.dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + args.dataset
    output_filepath = save_csv(predictions_df, save_path, 
                     args.model, 
                     args.strategy, 
                     args.serialization,
                     args.json, 
                     args.detailed_instructions,
                     args.start,
                     args.end,
                    timestamp,k_shots=args.k_shots,
                    k_shots_ablation=args.k_shots_ablation)

    print("Processing complete. Predictions saved.")
    
    # Evaluate predictions
    print("Evaluating predictions...")
    ground_truths = test_dataset[utils._DATASETS[args.dataset]['target']]
    predictions = predictions_df['Estimated_Acuity']

    metrics = utils.evaluate_predictions(predictions, ground_truths, ordinal=True, by_class=True)
    print("Overall Metrics:", metrics)
    utils.save_metrics(metrics, output_filepath.split('.csv')[0])
    print("Evaluation complete. Metrics and plots saved.")
    