from utils.config import OPENAI_API_KEY, TOGETHER_API_KEY, DATABRICKS_TOKEN, ANTHROPIC_KEY, GEMINI_KEY
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, 
                             classification_report, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import StratifiedShuffleSplit
from openai import OpenAI
import pandas as pd
import anthropic
from together import Together
import os
import time 
import re
import json 
from google import genai
from google.genai import types
import base64
from pydantic import BaseModel

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

client = OpenAI()
client_tog = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

client_safe = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
)

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

gemini_client = genai.Client(api_key=GEMINI_KEY)

###################
## Data Processing
###################

class Acuity(BaseModel):
    acuity: int

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
    "Triage-MIMIC-Private-Official": 
        {'filepath': "./data/mimic-iv-private/triage_stratified_2500.csv",
         'training_set_filepath':'./data/mimic-iv-private/triage_stratified_training.csv',
        'format': 'csv',
        'target': 'acuity',
        'hippa': True,
        },
    "Triage-Private-Stratified": 
        {'filepath': "./data/mimic-iv-private/triage_stratified_2500.csv",
         'training_set_filepath':'./data/mimic-iv-private/triage_stratified_training.csv',
        'format': 'csv',
        'target': 'acuity',
        'hippa': True,
        },
    "Triage-Private-Sex-Race": 
        {'filepath': "./data/mimic-iv-private/full_triage_2500.csv",
         'training_set_filepath':'./data/mimic-iv-private/triage_stratified_training.csv',
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

def load_predictions(filename, format='txt', save_path="./results/"):
    if format == 'csv':
        predictions_file = os.path.join(save_path, f"{filename}.csv")
        predictions = pd.read_csv(predictions_file)
    else: 
        predictions_file = os.path.join(save_path, f"{filename}.txt")
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

###################
## Evaluation
###################

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

    # Error metrics (MAE and MSE)
    mae = mean_absolute_error(correct_answers, predicted_answers)
    mse = mean_squared_error(correct_answers, predicted_answers)
    # Metrics by class
    if by_class:
        report = classification_report(correct_answers, predicted_answers, output_dict=True)

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
            "mae": mae,
            "mse": mse
        }
    }
    if ordinal:
        qwk = cohen_kappa_score(correct_answers, predicted_answers, weights="quadratic")
        results['overall']["quadratic_kappa"] = qwk
    if by_class:
        results["by_class"] = report

    return results

###################
## API Calling
###################

def query_gpt_safe(prompt, model="openai-gpt-4o-high-quota-chat", return_json=False,temperature=0.0, max_tokens=1000, debug=False):
    time.sleep(0.5)
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

def query_claude(message: str, model: str, temperature: float, max_tokens: int):
    try:
        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return None

def query_gemini(message, model, temperature=0, max_tokens=1000):
    response = gemini_client.models.generate_content(
        model=model,
        contents=message,
        config=types.GenerateContentConfig(
            temperature=temperature,
            seed=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=Acuity,
        ),
    )

    return response.text
 
def query_llm(prompt, max_tokens=1000, temperature=0, top_p = 0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, logprobs=False):
    if debug:
        print(prompt)
        print(f"Model: {model}")
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model:
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
            elif 'claude' in model:
                response = query_claude(prompt, model, temperature, max_tokens)
                if return_json:
                    return re.sub(r'(?<!\\)\n', '', response)
                return response
            elif 'gemini' in model:
                return query_gemini(prompt, model, temperature, max_tokens)
            else:
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
            if debug:
                print(response.choices[0].message.content.strip())
            if logprobs:
                return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response.choices[0].message.content.strip()
        except Exception as e:
            if 'gpt' in model:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            if curr_try_num >= max_try_num:
                return (-1)
            time.sleep(10)

###################
## Prompting
###################

TEMPLATES = {
  "triage-mimic": {
    "spaces": "temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint\n{temperature}°F   {heartrate} bpm   {resprate} breaths/min   {o2sat}%   {sbp} mmHg   {dbp} mmHg   {pain}   {chiefcomplaint}",
    "commas": "temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint\n{temperature}°F, {heartrate} bpm, {resprate} breaths/min, {o2sat}%, {sbp} mmHg, {dbp} mmHg, {pain}, {chiefcomplaint}",
    "newline": "temperature: {temperature}°F\nheartrate: {heartrate} bpm\nresprate: {resprate} breaths/min\no2sat: {o2sat}%\nsbp: {sbp} mmHg\ndbp: {dbp} mmHg\npain: {pain}\nchiefcomplaint: {chiefcomplaint}\n",
  },
  "triage-ktas": {
    "spaces": "temperature   heartrate   resprate   sbp   dbp   pain   chiefcomplaint   diagnosis in ED\n{temperature}°F   {heartrate} bpm   {resprate} breaths/min   {sbp} mmHg   {dbp} mmHg   {pain}   {chiefcomplaint}   {suspicion}",
    "commas": "temperature, heartrate, resprate, sbp, dbp, pain, chiefcomplaint, diagnosis in ED\n{temperature}°F, {heartrate} bpm, {resprate} breaths/min, {sbp} mmHg, {dbp} mmHg, {pain}, {chiefcomplaint}, {suspicion}",
    "newline": "temperature: {temperature}°F\nheartrate: {heartrate} bpm\nresprate: {resprate} breaths/min\nsbp: {sbp} mmHg\ndbp: {dbp} mmHg\npain: {pain}\nchiefcomplaint: {chiefcomplaint}\ndiagnosis in ED: {suspicion}",
  }
}
def convert_arrival(arrival_transport, dataset = 'triage-mimic'):
    if dataset=='triage-mimic':
        mapping = {
            "WALK IN": " via walk-in",
            "AMBULANCE": " via ambulance",
            "UNKNOWN": "",
            "OTHER": "",
            "HELICOPTER": "via helicopter",
            "EMS": " via EMS transport",
            "PRIVATE VEHICLE": " via private vehicle",
        }
    elif dataset == 'triage-ktas':
        mapping = {
        "WALKING": " arriving on foot",
        "119 AMBULANCE": " arriving by public ambulance",
        "PRIVATE VEHICLE": " arriving by private vehicle",
        "PRIVATE AMBULANCE": " arriving by private ambulance",
        'PUBLIC TRANSPORTATION': "arriving by public transportation",
        'WHEELCHAIR': 'came on a wheelchair'
    }
    return mapping.get(arrival_transport.upper(), "")

def format_row(row, dataset='triage-mimic', serialization='natural'):
    # Create a natural language description of the patient.
    if dataset == 'triage-mimic':
        if serialization in TEMPLATES.get('triage-mimic', {}):
            template = TEMPLATES['triage-mimic'][serialization]
            return template.format(
                temperature=row.get('temperature', 'N/A'),
                heartrate=row.get('heartrate', 'N/A'),
                resprate=row.get('resprate', 'N/A'),
                o2sat=row.get('o2sat', 'N/A'),
                sbp=row.get('sbp', 'N/A'),
                dbp=row.get('dbp', 'N/A'),
                pain=row.get('pain', 'N/A'),
                chiefcomplaint=row.get('chiefcomplaint', 'N/A')
            )
        elif serialization=='natural':
            # Handle vitals with fallback values
            vitals = {
                "temperature": f" a temperature of {row['temperature']}°F," if pd.notna(row.get("temperature")) else "",
                "heartrate": f" a heartrate of {row['heartrate']} bpm" if pd.notna(row.get("heartrate")) else "",
                "resprate": f", a respiratory rate of {row['resprate']} breaths per minute" if pd.notna(row.get("resprate")) else "",
                "o2sat": f", oxygen saturation at {row['o2sat']}%" if pd.notna(row.get("o2sat")) else "",
                "sbp": f", systolic blood pressure of {row['sbp']} mmHg" if pd.notna(row.get("sbp")) else "",
                "dbp": f", diastolic blood pressure of {row['dbp']} mmHg" if pd.notna(row.get("dbp")) else "",
                "pain": f", pain level reported as '{row['pain']}'" if pd.notna(row.get("pain")) else "",
                "chiefcomplaint": f", and a chief complaint described as '{row['chiefcomplaint']}'" if pd.notna(row.get("chiefcomplaint")) else "",
            }
            missing_vitals = [key for key, value in vitals.items() if value == ""]
            
            # Handle missing values
            race = row.get("race").lower()+', ' if (pd.notna(row.get("race")) and (row.get("race") != 'UNKNOWN')) else ""
            age = str(int(row.get("anchor_age"))) + '-year-old ' if pd.notna(row.get("anchor_age")) else ""
            gender = row.get("gender", "person")
            gender_str = "man" if gender == "M" else "woman" if gender == "F" else "person"
            pronoun = " He has" if gender == "M" else " She has" if gender == "F" else " They have"

            if len(missing_vitals) == 8:
                pronoun = ""
            if len(missing_vitals) > 0:
                missing = ", ".join(missing_vitals).replace("_", " ")  # Replace underscores for better readability
                missing = f" Data on {missing} is missing."
            else:
                missing = ""

                
            template = "A {race}{age}{gender_str} arrives at the emergency department{arrival}.{pronoun}{vitals}.{missing}"
            return template.format(
                race=race,
                age = age, 
                gender_str = gender_str, 
                arrival=convert_arrival(row.get('arrival_transport'), dataset=dataset),
                pronoun=pronoun, 
                vitals=''.join(vitals.values()),
                missing=missing
            )
        
    elif dataset == 'triage-ktas':
        if serialization in TEMPLATES.get('triage-mimic', {}):
            template = TEMPLATES['triage-mimic'][serialization]
            return template.format(
                temperature=row.get('BT', 'N/A'),
                heartrate=row.get('HR', 'N/A'),
                resprate=row.get('RR', 'N/A'),
                sbp=row.get('SBP', 'N/A'),
                dbp=row.get('DBP', 'N/A'),
                pain=row.get('NRS_pain', 'N/A'),
                chiefcomplaint=row.get('Chief_complain', 'N/A'),
                suspicion=row.get('Diagnosis in ED', 'N/A')
            )
        # --- Triage-ktas version ---
        age = f"{int(row['Age'])}-year-old " if pd.notna(row.get("Age")) else ""
        
        # Map the Sex column: assuming 1 = Female, 2 = Male.
        sex = row.get("Sex")
        if pd.isna(sex):
            gender_str = "person"
            pronoun = "They have"
        else:
            if sex == 'Female':
                gender_str = "woman"
                pronoun = "She has"
            elif sex == 'Male':
                gender_str = "man"
                pronoun = "He has"
            else:
                gender_str = "person"
                pronoun = "They have"
        
        arrival_text = convert_arrival(row.get("Arrival mode"),dataset=dataset)
        
        chief_text = f" with a chief complaint of '{row['Chief_complain']}'" if pd.notna(row.get("Chief_complain")) else ""
        
        # Include information about injury if applicable.
        injury = row.get("Injury")
        injury_text = ""
        if pd.notna(injury):
            if injury == 'Yes':
                injury_text = " sustained an injury"
        
        # Prepare the vital signs.
        vitals = {}
        vitals["temperature"] = f" temperature of {row['BT']}°F" if pd.notna(row.get("BT")) else ""
        vitals["heartrate"] = f", heart rate of {row['HR']} bpm" if pd.notna(row.get("HR")) else ""
        vitals["resprate"] = f", respiratory rate of {row['RR']} breaths per minute" if pd.notna(row.get("RR")) else ""
        vitals["sbp"] = f", systolic blood pressure of {row['SBP']} mmHg" if pd.notna(row.get("SBP")) else ""
        vitals["dbp"] = f", diastolic blood pressure of {row['DBP']} mmHg" if pd.notna(row.get("DBP")) else ""
        
        # Handle pain:
        # 'Pain' is a flag (0 or 1) indicating whether the patient feels pain.
        # 'NRS_pain' provides the actual pain level (and may be NA).
        # Since 'Pain' is never null, we can safely convert it to an integer.
        pain_flag = int(row["Pain"])
        if pain_flag == 1:
            if pd.notna(row.get("NRS_pain")):
                vitals["pain"] = f", and reports pain with a level of {row['NRS_pain']}."
            else:
                vitals["pain"] = ", and reports pain but no level was provided"
        else:
            vitals["pain"] = ""
        
        # Mental status, if available.
        if pd.notna(row.get("Mental")):
            vitals["mental"] = f" He is mentally {row['Mental']}"
        else:
            vitals["mental"] = ""
        
        missing_vitals = [key for key, value in vitals.items() if value == ""]
        
        description = (
            f"A {age}{gender_str}{injury_text} arrives at the emergency department"
            f"{arrival_text}{chief_text}. "
            f"{pronoun}{''.join(vitals.values())}."
            f" The patient is suspected to have {row["Diagnosis in ED"]}."
        )
        if missing_vitals:
            missing_str = ", ".join(missing_vitals).replace("_", " ")
            description += f" Data on {missing_str} is missing."
        
        return description

alpaca_prompt = """### Instruction: {}

### Input: {}

### Response: {}"""

blackbox_llm_prompt = """### Instructions
{}

### Patient's Clinical Presentation
{}"""

blackbox_llm_prompt_with_examples = """### Instructions
{}

### Examples 
{}

### Patient's Clinical Presentation
{}"""

INSTRUCTIONS = {
    "triage-mimic": {
        "vanilla": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "auto_CoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "CoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

Step 1: Assess Immediate Risk

Is the patient dying or at immediate risk of death? Consider: Apnea, pulselessness, severe respiratory distress, unresponsiveness, SpO2 < 90%, acute mental status changes.

Step 2: Determine Need for Immediate Life-Saving Intervention

Does the patient require an immediate intervention to secure an airway, support breathing, or maintain circulation? If yes, classify as ESI Level 1. If no, proceed to the next step.

Step 3: Identify Critical Interventions

Does the patient require any of the following life-saving interventions?
- Airway/Breathing: BVM ventilation, intubation, surgical airway, emergent CPAP/BiPAP.
- Electrical Therapy: Defibrillation, emergent cardioversion, external pacing.
- Procedures: Chest needle decompression, pericardiocentesis, open thoracotomy, intraosseous access.
- Hemodynamics: Major IV fluid resuscitation, blood administration, control of major bleeding.
- Medications: Naloxone, D50, dopamine, atropine, adenosine.

If yes, classify as ESI Level 1. If no, proceed to the next step.

Step 4: Evaluate Key Clinical Signs

Is the patient presenting with any of these critical conditions?
- Cardiac Arrest
- Respiratory Arrest
- Severe respiratory distress (agonal/gasping respirations)
- SpO2 < 90%
- Severe Bradycardia/Tachycardia with hypoperfusion
- Hypotension with signs of hypoperfusion
- Critically injured trauma patient (unresponsive)
- Overdose with a respiratory rate < 6
- Anaphylactic reaction
- Infant that is flaccid
- Unresponsive with a strong odor of alcohol
- Hypoglycemia with a change in mental status

If yes, classify as ESI Level 1. If no, proceed to the next step.

Step 5: Differentiate Between ESI 1 and ESI 2

Does the patient require immediate physician presence for survival? ESI Level 1: Requires physician at bedside immediately. ESI Level 2: Can wait briefly; nurse can initiate care (e.g., IV access, ECG, oxygen).

Step 6. If it's not clear by now, based on all the provided information, please determine the appropriate Emergency Severity Index (ESI) acuity level on a scale from 1 to 5.
"""
        },
        "KATE": "Below is the patient's clinical presentation. Based on the provided information, please determine the appropriate Emergency Severity Index (ESI) acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed).",
        "KATE_CoT": """Below is the patient's clinical presentation. Based on the provided information, please determine the appropriate Emergency Severity Index (ESI) acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
        
        Step 1: Learn from the following examples—the reasoning style and output format in these examples should guide your response."""
    },
    "triage-ktas": {
        "vanilla": "Based on the clincial presentation, determine the Korean Triage and Acuity Scale (KTAS) for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        "auto_CoT": "",
        "CoT": "",
        },
    "triage-handbook": {
        "vanilla": "Based on the clincial presentation, determine the Emergency Severity Index (ESI) acuity for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        "auto_CoT": "",
        "CoT": "",
    }
    
}
# Function to convert a CSV row into instruction, input, and output strings.
def format_instruction_prompt_for_blackbox(row, strategy = 'vanilla', dataset='triage-mimic', serialization='natural', return_json=False, examples = None):
    patient_description = format_row(row, dataset=dataset, serialization=serialization)
    instructions = INSTRUCTIONS['dataset']['strategy']
    if return_json:
        if 'CoT' in strategy:
            instructions['system'] = instructions['system'] + """Output your response strictly in JSON format with the following keys:
"Reasoning": <the step by step rationale>,
"Acuity": <the assigned acuity level as an integer between 1 and 5>,
"""
        else:
            instructions['system'] = instructions['system'] + """Output your response strictly in JSON format with the following key:
    "Acuity": <the assigned acuity level as an integer between 1 and 5>"""
    if strategy == 'KATE':
        return blackbox_llm_prompt_with_examples.format(instructions, patient_description)
    else:
        return blackbox_llm_prompt.format(instructions, patient_description)

def format_instruction_prompt_for_finetuning(row, EOS_TOKEN, dataset='triage-mimic', split='train'):
    patient_description = format_row(row, dataset=dataset)
    
    # Define the instruction for the model.
    if dataset=='triage-mimic' or dataset=='triage-handbook':
        instruction = "Based on the clinical presentation, determine the Emergency Severity Index (ESI) acuity for the following patient."
        output_text = f"The ESI acuity for this patient is {row['acuity']}."
    elif dataset=='triage-ktas':
        instruction = "Based on their clinical presentation, determine the KTAS acuity for the following patient."
        output_text = f"The KTAS acuity for this patient is {row['KTAS_expert']}."
    else: 
        raise Exception("Invalid dataset name")
    
    if split == 'train':
        text = alpaca_prompt.format(instruction, patient_description, output_text) + EOS_TOKEN
    elif split == 'test':
        text = alpaca_prompt.format(instruction, patient_description)
    return {"text": text}

###################
## Legacy Functions
###################

def instruction_prompt(dataset, strategy, return_json=False):
    cot_instruction = "your reasoning in the key 'reasoning' and "
    json_instruction = f"Answer in valid JSON format, providing {cot_instruction if 'CoT' in strategy else ''}acuity as a single numeric value in the key 'acuity'."
    if 'Triage' in dataset:
        if 'CoT' in strategy:
            return f"Estimate the patient's acuity from 1-5 based on the following guidelines: Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority. Let's think step by step. {json_instruction if return_json else ''}"
        return f"Estimate the patient's acuity from 1-5 based on the following guidelines: Acuity is assessed using the Emergency Severity Index (ESI) Five Level triage system. This priority is assigned by a registered nurse. Level 1 is the highest priority, while level 5 is the lowest priority. {json_instruction if return_json else ''}"
