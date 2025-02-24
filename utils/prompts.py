import pandas as pd

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

ALPACA_PROMPT = """### Instruction: {}

### Input: {}

### Response: {}"""

BLACKBOX_LLM_PROMPT = """### Instructions
{}

### Patient's Clinical Presentation
{}"""

BLACKBOX_LLM_PROMPT_WITH_EXAMPLES = """### Instructions
{}

### Examples 
{}

### Patient's Clinical Presentation
{}"""

INSTRUCTIONS = {
    "triage-mimic": {
        "vanillav0": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level"
        },
        "vanilla": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "autocot": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "cot": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk **

Is the patient dying or at immediate risk of death? Consider: Apnea, pulselessness, severe respiratory distress, unresponsiveness, SpO2 < 90%, acute mental status changes.

** Step 2: Determine Need for Immediate Life-Saving Intervention **

Does the patient require an immediate intervention to secure an airway, support breathing, or maintain circulation? If yes, classify as ESI Level 1. If no, proceed to the next step.

** Step 3: Identify Critical Interventions **

Does the patient require any of the following life-saving interventions?
- Airway/Breathing: BVM ventilation, intubation, surgical airway, emergent CPAP/BiPAP.
- Electrical Therapy: Defibrillation, emergent cardioversion, external pacing.
- Procedures: Chest needle decompression, pericardiocentesis, open thoracotomy, intraosseous access.
- Hemodynamics: Major IV fluid resuscitation, blood administration, control of major bleeding.
- Medications: Naloxone, D50, dopamine, atropine, adenosine.

If yes, classify as ESI Level 1. If no, proceed to the next step.

** Step 4: Evaluate Key Clinical Signs **

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

** Step 5: Differentiate Between ESI 1 and ESI 2 **

- Does the patient require immediate physician presence for survival? 
- ESI Level 1: Requires physician at bedside immediately. 
- ESI Level 2: Can wait briefly; nurse can initiate care (e.g., IV access, ECG, oxygen).

Three broad questions are used to determine whether the patient meets level-2 criteria. 
1. Is this a high-risk situation?
    - Vital signs are often not helpful in the identification of high-risk patients. The patient typically presents to the ED with a chief complaint, signs and symptoms, or history suggestive of a problem or condition that is serious and, unless dealt with promptly, can deteriorate rapidly. Often patient age, past medical history, and current medications influence the perceived severity of the chief complaint. 
    - For example, a frail elderly patient with severe abdominal pain is at a much higher risk of morbidity and mortality than a 20-year-old. The elderly patient with abdominal pain should be classified as ESI level 2, while the 20-year-old with stable vital signs will usually be classified as ESI level 3.
2. Is the patient confused, lethargic or disoriented?
3. Is the patient in severe pain or distress?
- All patients who have a pain rating of 7/10 or greater should be considered for meeting ESI level-2 criteria.
- Considered is a very important word. It is up to the discretion of the triage nurse to determine whether the clinical condition and pain rating in combination warrant a rating of ESI level 2. For example, a patient who had a heavy metal object fall on his toe may rate the pain a 10/10. Indeed, the patient may have a fracture and is experiencing severe pain. The patient probably has done nothing to try to relieve the pain prior to arrival in the ED. The correct triage level for this patient would be ESI level 4. Only one resource will be needed (an x-ray).
- Examples of patients for whom the triage nurse could use severe pain criteria to justify an ESI level-2 rating include: A patient with 10/10 flank pain who is writhing at triage.

** Step 6: Differentiate between ESI 3, 4 and 5
Resource prediction is an integral part of the ESI for patients identified as ESI level 3, 4, or 5.

** Step 7: If it's not clear by now, analyze the examples provided and identify key similarities and differences between the patient and the examples.
     """
        },
        "kate": """Below are some example cases with their ESI acuity levels, followed by a new patient case. Please analyze the examples and use them to determine the appropriate Emergency Severity Index (ESI) acuity level for the new patient on a scale from 1 to 5.""",
        "katecot": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step. 
        
Step 1: Assess Immediate Risk

- Is the patient dying or at immediate risk of death? 
- Consider: Apnea, pulselessness, severe respiratory distress, unresponsiveness, SpO2 < 90%, acute mental status changes.

Step 2: Determine Need for Immediate Life-Saving Intervention

- Does the patient require an immediate intervention to secure an airway, support breathing, or maintain circulation? 
- If yes, classify as ESI Level 1. 
- If no, proceed to the next step.

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

- Does the patient require immediate physician presence for survival? 
- ESI Level 1: Requires physician at bedside immediately. 
- ESI Level 2: Can wait briefly; nurse can initiate care (e.g., IV access, ECG, oxygen).

Step 6: Determine if the patient is ESI 2

Three broad questions are used to determine whether the patient meets level-2 criteria. 
1. Is this a high-risk situation? 
    - Vital signs are often not helpful in the identification of high-risk patients. The patient typically presents to the ED with a chief complaint, signs and symptoms, or history suggestive of a problem or condition that is serious and, unless dealt with promptly, can deteriorate rapidly. Often patient age, past medical history, and current medications influence the perceived severity of the chief complaint. 
    - For example, a frail elderly patient with severe abdominal pain is at a much higher risk of morbidity and mortality than a 20-year-old. The elderly patient with abdominal pain should be classified as ESI level 2, while the 20-year-old with stable vital signs will usually be classified as ESI level 3.
2. Is the patient confused, lethargic or disoriented? 
3. Is the patient in severe pain or distress?
- All patients who have a pain rating of 7/10 or greater should be considered for meeting ESI level-2 criteria.
- Considered is a very important word. It is up to the discretion of the triage nurse to determine whether the clinical condition and pain rating in combination warrant a rating of ESI level 2. For example, a patient who had a heavy metal object fall on his toe may rate the pain a 10/10. Indeed, the patient may have a fracture and is experiencing severe pain. The patient probably has done nothing to try to relieve the pain prior to arrival in the ED. The correct triage level for this patient would be ESI level 4. Only one resource will be needed (an x-ray).
- Examples of patients for whom the triage nurse could use severe pain criteria to justify an ESI level-2 rating include: A patient with 10/10 flank pain who is writhing at triage.

Step 6: If it's not clear by now, analyze the examples provided and identify key similarities and differences between the patient and the examples.
        
Step 7: Based on all the provided information, please determine the appropriate Emergency Severity Index (ESI) acuity level on a scale from 1 to 5."""
    },
    "triage-ktas": {
        "vanilla": "Based on the clincial presentation, determine the Korean Triage and Acuity Scale (KTAS) for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        "auto_CoT": "",
        "KATE": "Based on the clincial presentation, determine the Korean Triage and Acuity Scale (KTAS) for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        "KATE_CoT": "Based on the clincial presentation, determine the Korean Triage and Acuity Scale (KTAS) for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        },
    "triage-handbook": {
        "vanilla": "Based on the clincial presentation, determine the Emergency Severity Index (ESI) acuity for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        "auto_CoT": "",
        "CoT": "",
        "KATE": "Based on the clincial presentation, determine the Emergency Severity Index (ESI) acuity for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
        "KATE_CoT": "Based on the clincial presentation, determine the Emergency Severity Index (ESI) acuity for the following patient from 1-5, where 1 is the highest priority and 5 is the lowest priority.",
    }
}
    
def convert_arrival(arrival_transport, dataset = 'triage-mimic'):
    if 'triage' in dataset.lower():
        mapping = {
            "WALK IN": " via walk-in",
            "AMBULANCE": " via ambulance",
            "UNKNOWN": "",
            "OTHER": "",
            "HELICOPTER": "via helicopter",
            "EMS": " via EMS transport",
            "PRIVATE VEHICLE": " via private vehicle",
        }
    elif 'ktas' in dataset.lower():
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
    if 'triage' in dataset.lower():
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
                "resprate": f", a respiratory rate of {row['resprate']} breaths/min" if pd.notna(row.get("resprate")) else "",
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
        
    elif 'ktas' in dataset.lower():
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
        vitals["resprate"] = f", respiratory rate of {row['RR']} breaths/min" if pd.notna(row.get("RR")) else ""
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
            f" The patient is suspected to have {row['Diagnosis in ED']}."
        )
        if missing_vitals:
            missing_str = ", ".join(missing_vitals).replace("_", " ")
            description += f" Data on {missing_str} is missing."
        
        return description


# Function to convert a CSV row into instruction, input, and output strings.
def format_instruction_prompt_for_blackbox(row, strategy='vanilla', dataset='triage-mimic', serialization='natural', return_json=False, examples=None):
    """
    Format instruction prompt based on strategy.
    
    Args:
        row: The patient data row
        strategy: The prompting strategy to use ('Vanilla', 'CoT', 'FewShot', etc.)
        dataset: The dataset being used (default: 'triage-mimic')
        serialization: How to format the patient data ('natural', 'spaces', etc.)
        return_json: Whether to request JSON-formatted output
        examples: Previously labeled examples for few-shot learning (required for some strategies)
    
    Returns:
        dict: Formatted prompt with 'system' and 'user' messages
    
    Raises:
        ValueError: If strategy requires examples but none provided, or if strategy unsupported
    """
    # Validate inputs
    strategy = strategy.lower()
    dataset = dataset.lower()
    
    if dataset not in INSTRUCTIONS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if strategy not in INSTRUCTIONS[dataset]:
        raise ValueError(f"Strategy {strategy} not supported for dataset {dataset}")
        
    # Get base prompt template & serialize patient data
    prompt = INSTRUCTIONS[dataset][strategy].copy()  # Create copy to avoid modifying original
    patient_description = format_row(row, dataset=dataset, serialization=serialization)
    
    # Add JSON formatting instructions if needed
    if return_json:
        json_instruction = (
            """ Output your response strictly in JSON format with the following keys:
"Reasoning": <the step by step rationale>,
"Acuity": <the assigned acuity level as an integer between 1 and 5>,
"""
            if 'cot' in strategy else
            """ Output your response strictly in JSON format with the following key:
"Acuity": <the assigned acuity level as an integer between 1 and 5>"""
        )
        prompt['system'] += json_instruction
    
    # Handle different strategy types
    EXAMPLE_REQUIRED_STRATEGIES = {'kate', 'katecot', 'fewshot', 'fewshotcot'}
    VANILLA_STRATEGIES = {'vanilla', 'cot', 'autocot', 'vanillav0'}
    
    if strategy in EXAMPLE_REQUIRED_STRATEGIES:
        if not examples:
            raise ValueError(f"Strategy {strategy} requires examples but none were provided")
        prompt['user'] = BLACKBOX_LLM_PROMPT_WITH_EXAMPLES.format(
            prompt['user'],
            examples,
            patient_description
        )
    elif strategy in VANILLA_STRATEGIES:
        prompt['user'] = BLACKBOX_LLM_PROMPT.format(
            prompt['user'],
            patient_description
        )
    elif strategy == 'selfconsistency':
        prompt['user'] = BLACKBOX_LLM_PROMPT.format(
            prompt['user'] + "\nPlease be decisive in your assessment.",
            patient_description
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
        
    return prompt

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
        text = ALPACA_PROMPT.format(instruction, patient_description, output_text) + EOS_TOKEN
    elif split == 'test':
        text = ALPACA_PROMPT.format(instruction, patient_description)
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
