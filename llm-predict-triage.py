import re
import pandas as pd
import json
from tqdm import tqdm
import utils.predictors as predictors
import argparse
import utils.utils as utils
import marshal
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
    time.sleep(1)
    answer_text = utils.query_gpt_safe(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "openai-gpt-35-turbo-chat", debug=debug)
    num = extract_num(answer_text)
    time.sleep(1)
    if type(num) == str and 'Error' in num:
        return utils.query_gpt_safe(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "openai-gpt-35-turbo-chat", debug=debug)
    else:
        return num

def serialization(dataset, row):
    if 'Triage' in dataset:
        return f"""temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint
{row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}"""

def instruction_prompt(dataset, strategy, return_json=False):
    cot_instruction = "your reasoning in the key 'reasoning' and "
    json_instruction = f"Answer in valid JSON format, providing {cot_instruction if 'CoT' in strategy else ''}acuity as a single numeric value in the key 'acuity'."
    if 'Triage' in dataset:
        if 'CoT' in strategy:
            return f"Estimate the patient's acuity from 1-5 based on the following guidelines: Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority. Let's think step by step. {json_instruction if return_json else ''}"
        return f"Estimate the patient's acuity from 1-5 based on the following guidelines: Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority. {json_instruction if return_json else ''}"

def create_prompt(row,strategy=None, return_json=False, detailed_instructions = False, bias=False):
    if 'FewShot' in strategy or strategy=='KATE':
        return f"""temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint
{row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}"""
    if detailed_instructions:
        task_description = """Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority. The levels are: 
1: When Level 1 condition or patient meets ED Trigger Criteria, the triage process stops, the patient is taken directly to a room and immediate physician intervention requested. Patient conditions which trigger level 1 include being unresponsive, intubated, apneic, pulseless, requiring a medication/intervention to alter ESI level e.g. narcan/adenosine/cardioversion, trauma, stroke, stemi
2: When a Level 2 condition is identified, the triage nurse notifies the resource nurse and appropriate placement will be determined. Patient conditions which trigger level 2 include high risk situations, new onset confusion, suicidal/homicidal ideation, lethargy, seizures or disorientation, possible ectopic pregnancy, an immunocompromised patient with a fever, severe pain/distress, or vital sign instability
3: Includes patients requiring two or more resources (labs, EKG, x-rays, IV fluids, etc) with stable vital signs
4: Patients requiring one resource only (labs, EKG, etc)
5: Patients not requiring any resources"""
    else: 
        task_description = "Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority"
    if bias:
        starting_prompt = f"Here is the profile of a {row['Race']} {row['Sex']} patient" 
    else:
        starting_prompt = "Here is the profile of a patient"
    cot, reasoning = "", ""
    if strategy=='CoT':
        cot, reasoning = "Let's think step by step", "your step-by-step reasoning in the key 'reasoning' and "
    if return_json:
        return f"""{starting_prompt}:

temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint
{row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}

Estimate their acuity from 1 to 5 based on the following guidelines: {task_description}. {cot}

Answer in valid JSON format, providing {reasoning}acuity as a single numeric value in the key 'acuity'."""
    else:    
        return f"""{starting_prompt}:

temperature   heartrate   resprate    o2sat   sbp   dbp   pain chiefcomplaint
{row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}

Estimate their acuity from 1-5 based on the following guidelines: {task_description}. {cot}
        """
        
# Handles structuring the response
def predict(index, row, prompt, predictor, model, strategy, return_json, k_shots, serialization_func, instruction_prompt, debug):
    # The logic for splitting strategy is partially handled here by passing the right parameters
    if strategy == 'FewShot' or strategy=='FewShotCoT' or strategy =='KATE':
        response = predictor.predict(prompt, row, model=model, k_shots = k_shots, return_json=return_json, serialization_func = serialization_func, instruction_prompt=instruction_prompt, debug=debug)
    elif strategy == 'ZeroShot' or strategy == 'CoT':
        response = predictor.predict(prompt, model=model, return_json=return_json, debug=debug)
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
                "Estimated_Acuity": response_data['acuity'],
                "Reasoning": response_data['reasoning'] if strategy=='CoT' else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": response_data['acuity'],
                "Reasoning": response_data['reasoning'] if strategy=='CoT' else None,
                **row.to_dict()  # Include the original row's data for reference
            }
    else: 
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": extract_acuity_from_text(response, debug=debug),
                "Reasoning": response_data['reasoning'] if strategy=='CoT' else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": extract_acuity_from_text(response, debug=debug),
                "Reasoning": response_data['reasoning'] if strategy=='CoT' else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        
def save_csv(df, savepath, model, predictive_strategy, json_param, detailed_instructions,  start_index, end_index, timestamp, k_shots=None):
    output_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        output_filepath = output_filepath + "_json"
    if detailed_instructions:
        output_filepath = output_filepath + "_detailed"
    if k_shots is not None:
        output_filepath = output_filepath + f"_{k_shots}"
    output_filepath = output_filepath + f"{start_index}_{end_index}_{timestamp}.csv"
    # Save the DataFrame to a CSV file
    df.to_csv(output_filepath, index=False)
    print(f"DataFrame saved to {output_filepath}")
    
def load_csv(savepath, model, predictive_strategy, start_index, end_index, json_param, detailed_instructions):
    input_filepath = f"{savepath}_{predictive_strategy}_{model}"
    if json_param:
        input_filepath = input_filepath + "_json"
    if detailed_instructions:
        input_filepath = input_filepath + "_detailed"
    input_filepath = input_filepath + f"{start_index}_{end_index}_{timestamp}.csv"
    
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
    parser.add_argument("--start", required=True, type=int, help="Start index of the samples to evaluate")
    parser.add_argument("--end", required=True, type=int, help="End index of the samples to evaluate")
    parser.add_argument("--model", required=True, type=str, default="gpt-4o-mini", help="LLM model to use.")
    parser.add_argument("--strategy", required=True, type=str, choices=["ZeroShot", 'FewShot', "CoT","FewShotCoT", "USC", "KATE"], default="standard", help="Prediction strategy to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for USC strategy")
    parser.add_argument("--k_shots", type=int, default=5, help="Number of shots for Few-Shot")
    parser.add_argument("--load_predictions", type=str, help="Timestamp & model_name & # of questions to existing predictions to load")
    parser.add_argument("--json", action="store_true", help="Turns on internal usage of json formats for the LLM API")
    parser.add_argument("--detailed_instructions", action="store_true", help="Turns on detailed instructions")
    parser.add_argument("--bias", action="store_true", help="Enables bias prompt")
    args = parser.parse_args()

    print("Loading Dataset...")
    filepath= utils._DATASETS[args.dataset]['filepath']
    format = utils._DATASETS[args.dataset]['format']
    dataset = utils.load_dataset(filepath, format, args.start, args.end)

    print("Potentially loading existing predictions...")
    predictions = []
    if args.load_predictions:
        predictions = utils.load_predictions(args.load_predictions)
        num_existing_predictions = len(predictions)
        print(f"Loaded {num_existing_predictions} existing predictions.")
    else:
        num_existing_predictions = 0
    num_new_predictions_needed = args.end + 1 - num_existing_predictions

    if args.k_shots:
        training_df = utils.load_dataset(utils._DATASETS[args.dataset]['training_set_filepath'], format, 0, 1000000)

    print(f"Making {num_new_predictions_needed} new predictions...")
    predictor = predictors.Predictor(args.dataset,
                                     strategy=args.strategy, 
                                     hippa=utils._DATASETS[args.dataset]['hippa'],
                                     target = utils._DATASETS[args.dataset]['target'],
                                     training_set= training_df if args.k_shots else None)

    # Initialize counter for saving progress
    new_predictions_since_last_save = 0
    total_predictions_made = num_existing_predictions

    if num_new_predictions_needed > 0:
        if utils._DATASETS[args.dataset]['format'] == 'csv':
            for i, row in tqdm(dataset.loc[num_existing_predictions:args.end].iterrows()):
                prompt = create_prompt(row, 
                                       strategy=args.strategy,
                                       return_json=args.json, 
                                       detailed_instructions=args.detailed_instructions, 
                                       bias=args.bias)
                prediction = predict(i, 
                                     row,
                                     prompt, 
                                     predictor, 
                                     args.model, 
                                     args.strategy, 
                                     args.json, 
                                     args.k_shots, 
                                     serialization,
                                     instruction_prompt(args.dataset, args.strategy, args.json),
                                     args.debug)
                predictions.append(prediction)
                new_predictions_since_last_save += 1
                total_predictions_made += 1

                if new_predictions_since_last_save >= 100:
                    # Save predictions to disk
                    predictions_df = pd.DataFrame(predictions)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = "./results/" + args.dataset + '/' 
                    os.makedirs(save_path, exist_ok=True)
                    save_path = save_path + f'{args.dataset}'
                    save_csv(predictions_df, save_path, 
                             args.model, 
                             args.strategy, 
                             args.json, 
                             args.detailed_instructions,
                             args.start,
                             args.start + total_predictions_made , 
                             timestamp)
                    new_predictions_since_last_save = 0
                    print(f"Saved progress after {len(predictions)} predictions.")
            predictions_df = pd.DataFrame(predictions)
    else:
        print("No new predictions needed.")

    # Save combined predictions one last time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "./results/" + args.dataset + f'/{args.dataset}'
    os.makedirs(save_path, exist_ok=True)
    save_csv(predictions_df, save_path, 
                     args.model, 
                     args.strategy, 
                     args.json, 
                     args.detailed_instructions,
                     args.start,
                     args.end, 
                    timestamp,k_shots=args.k_shots)

    print("Processing complete. Predictions saved.")