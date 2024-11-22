import os
import re
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from collections import Counter
import utils.utils as utils
  
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
    
def extract_acuity_from_text(text):
    # Call another model to extract the acuity if necessary
    answer = utils.client_safe.chat.completions.create(
        model="openai-gpt-4o-chat",
        messages=[{"role": "user", "content": f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, provide the mean of the two numbers it's considering and provide the mean alone.\n\n\"\"\"{text}.\"\"\""}],
        temperature=0,
        top_p=0
    )
    answer_text = answer.choices[0].message.content.strip()
    num = extract_num(answer_text)
    if type(num) == str and 'Error' in num:
        answer = utils.client_safe.chat.completions.create(
        model="openai-gpt-4o-chat",
        messages=[{"role": "user", "content": f"Extract the final estimated acuity from the following information and output the number alone. If the estimate is uncertain, provide the mean of the two numbers it's considering and provide the mean alone.\n\n\"\"\"{answer_text}.\"\"\""}],
        temperature=0,
        top_p=0
        )  
        return extract_num(answer.choices[0].message.content.strip()) 
    else:
        return num


# Function to create the prompt based on the patient's data
def create_prompt(row,strategy=None, json=False, detailed_instructions = False, bias=False):
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
        starting_prompt = "Here is the profile of a patient"
    else:
        starting_prompt = "Here is the profile of a {row['Race']} {row['Sex']} patient" 
    cot = ""
    if strategy=='CoT':
        cot = "Let's think step by step" 
    if json:
        return f"""{starting_prompt}:

temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint
{row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}

Estimate their acuity from 1 to 5 based on the following guidelines: {task_description}. {cot}

Answer in valid JSON format, providing acuity as a single numeric value in the key "acuity" and a concise explanation of your reasoning in the key "reasoning"."""
    else:    
        return f"""{starting_prompt}:

temperature   heartrate   resprate    o2sat   sbp   dbp   pain chiefcomplaint
{row['temperature']}   {row['heartrate']}   {row['resprate']}   {row['o2sat']}   {row['sbp']}   {row['dbp']}   {row['pain']}   {row['chiefcomplaint']}

Estimate their acuity from 1-5 based on the following guidelines: {task_description}. {cot}
        """

def predict(df, model, predictive_strategy, start_index=0, end_index=500, json_param=False, detailed_instructions=False, bias=False, debug=False):
    print(f"Calling {model}...")
    predictions = []
    for index, row in tqdm(df.loc[start_index:end_index].iterrows(), desc="Triaging Patients"):
        # Generate the prompt & query the model
        prompt = create_prompt(row, strategy=predictive_strategy, json=json_param, detailed_instructions=detailed_instructions, bias=bias)
        response = utils.query_gpt_safe(prompt, model=model, json=json_param, debug=debug)
        if json_param:
            response_data = json.loads(response)
            predictions.append({
                "Estimated_Acuity": response_data['acuity'],
                "Reasoning": response_data['reasoning'],
                **row.to_dict()  # Include the original row's data for reference
            })
        else: 
            predictions.append({
                "Estimated_Acuity": extract_acuity_from_text(response),
                "Reasoning": response,
                **row.to_dict()  # Include the original row's data for reference
            })
            
 
    # Create a DataFrame from the predictions list
    predictions_df = pd.DataFrame(predictions)
    return predictions_df
  

## These functions' argument conventions are specific to this experiment
def save_csv(df, predictive_strategy, model, start_index, end_index, json_param, detailed_instructions):
    output_filepath = f"./data/triage_dataset_{predictive_strategy}_{model}_{start_index}_{end_index}"
    if json_param:
        output_filepath = output_filepath + "_json"
    if detailed_instructions:
        output_filepath = output_filepath + "_detailed"
    output_filepath = output_filepath + ".csv"
    # Save the DataFrame to a CSV file
    df.to_csv(output_filepath, index=False)
    print(f"DataFrame saved to {output_filepath}")
    
def load_csv(predictive_strategy, model, start_index, end_index, json_param, detailed_instructions):
    """
    Load a DataFrame from a CSV file saved using the save_csv function.

    Args:
        predictive_strategy (str): The strategy used for prediction (e.g., "ZeroShot").
        start_index (int): The starting index of the dataset.
        end_index (int): The ending index of the dataset.
        json_param (bool): Whether the data was saved with JSON output format.
        detailed_instructions (bool): Whether detailed instructions were included in the saved data.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    input_filepath = f"./data/triage_dataset_{predictive_strategy}_{model}_{start_index}_{end_index}"
    if json_param:
        input_filepath += "_json"
    if detailed_instructions:
        input_filepath += "_detailed"
    input_filepath += ".csv"
    
    try:
        df = pd.read_csv(input_filepath)
        print(f"DataFrame loaded from {input_filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {input_filepath}")
        return None
