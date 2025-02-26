import pandas as pd

TEMPLATES = {
  "triage-mimic": {
    "spaces": "temperature   heartrate   resprate   o2sat   sbp   dbp   pain   chiefcomplaint\n{temperature}°F   {heartrate} bpm   {resprate} breaths/min   {o2sat}%   {sbp} mmHg   {dbp} mmHg   {pain}   {chiefcomplaint}",
    "commas": "temperature, heartrate, resprate, o2sat, sbp, dbp, pain, chiefcomplaint\n{temperature}°F, {heartrate} bpm, {resprate} breaths/min, {o2sat}%, {sbp} mmHg, {dbp} mmHg, {pain}, {chiefcomplaint}",
    "newline": "temperature: {temperature}°F\nheartrate: {heartrate} bpm\nresprate: {resprate} breaths/min\no2sat: {o2sat}%\nsbp: {sbp} mmHg\ndbp: {dbp} mmHg\npain: {pain}\nchiefcomplaint: {chiefcomplaint}\n",
  },
  "triage-ktas": {
    "spaces": "temperature   heartrate   resprate   sbp   dbp   pain   chiefcomplaint   diagnosis in ED   injury   arrival   mental\n{temperature}°F   {heartrate} bpm   {resprate} breaths/min   {sbp} mmHg   {dbp} mmHg   {pain}   {chiefcomplaint}   {suspicion}   {injury}   {arrival}   {mental}",
    "commas": "temperature, heartrate, resprate, sbp, dbp, pain, chiefcomplaint, diagnosis in ED, injury, arrival, mental\n{temperature}°F, {heartrate} bpm, {resprate} breaths/min, {sbp} mmHg, {dbp} mmHg, {pain}, {chiefcomplaint}, {suspicion}, {injury}, {arrival}, {mental}",
    "newline": "temperature: {temperature}°F\nheartrate: {heartrate} bpm\nresprate: {resprate} breaths/min\nsbp: {sbp} mmHg\ndbp: {dbp} mmHg\npain: {pain}\nchiefcomplaint: {chiefcomplaint}\ndiagnosis in ED: {suspicion}\ninjury: {injury}\narrival mode: {arrival}\nmental: {mental}",
  }
}

ALPACA_PROMPT = """### Instruction: {instruction}

### Input: {input}

### Response: {response}"""

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
        "Vanilla": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "SelfConsistency": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk for ESI Level 1 **

- Is the patient dying or at immediate risk of death?
- Does the patient require an immediate life-saving intervention?

** Step 2: Evaluate Key Clinical Signs for ESI Level 2 **

Assess High-Risk Situations: Identify patients with potentially life-threatening conditions who do not require immediate life-saving interventions but should not wait. Use the rule of thumb: “Would I use my last open bed for this patient?” Key indicators include:
- Severe chest pain, suspected acute coronary syndrome (ACS)
- Respiratory distress (e.g., severe asthma, pleural effusion)
- Neurological deficits (e.g., stroke symptoms, meningitis suspicion)
- Uncontrolled bleeding or significant trauma

Evaluate Mental Status Changes: Classify patients with new onset confusion, lethargy, or disorientation as ESI Level 2. This includes:
- Suspected stroke (sudden motor/speech deficits)
- Altered mental status from metabolic, toxic, or neurological causes
- Pediatric patients with signs of sepsis or dehydration

Prioritize Severe Pain or Distress: Patients reporting severe pain (≥7/10) or severe psychological distress may qualify for Level 2 if their condition necessitates immediate intervention. Examples:
- Sickle cell crisis or acute oncologic pain
- Severe abdominal pain in elderly patients
- Trauma with high pain burden requiring urgent evaluation

** Step 3: Differentiate between ESI 3, 4 and 5 **
- Resource prediction is an integral part of the ESI for patients identified as ESI level 3 (many resources), 4 (one resource), or 5 (no resources).
- Resources include: Labs (blood, urine), ECG, X-rays, CT-MRI-ultrasound, angiography, IV fluids (hydration) , IV, IM or nebulized medications, Specialty consultation, Simple procedure (lac repair, Foley cath), Complex procedure (conscious sedation) 
- Check Danger Zone Vitals: If the patient has very high heart rate, breathing rate, or oxygen saturation below 92%, consider upgrading to ESI Level 2. Use these thresholds:
    - HR: >180 (infants), >160 (toddlers), >140 (children), >100 (older children/adults)
    - RR: >50 (infants), >40 (toddlers), >30 (children), >20 (older children/adults)
    - SaO₂: <92%

** Step 4: If it's not clear by now, determine the appropriate ESI acuity level to the best of your ability. **"""
        },
        "AutoCoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "DemonstrationCoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Here's a demonstration of how to do this:

Case: A 34-year-old obese female presents to triage complaining of generalized abdominal pain (pain scale rating: 6/10) for 2 days. She has vomited several times and states her last bowel movement was 3 days ago. She has a history of back surgery, takes no medications, and is allergic to peanuts. Vital signs: T 97.8° F, HR 104, RR 16, BP 132/80, SpO2 99 percent.
Rationale: This patient will need a minimum of two or more resources: lab, IV fluids, perhaps IV medication for nausea, and a CT scan. The triage nurse would review the patient's vital signs and consider the heart rate. The heart rate falls just outside the accepted parameter for the age of the patient but could be due to pain or exertion. In this case, the decision should be to assign the patient to ESI level 3."""
        },
        "CoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk for ESI Level 1 **

- Is the patient dying or at immediate risk of death?
- Does the patient require an immediate life-saving intervention?

** Step 2: Evaluate Key Clinical Signs for ESI Level 2 **

Assess High-Risk Situations: Identify patients with potentially life-threatening conditions who do not require immediate life-saving interventions but should not wait. Use the rule of thumb: “Would I use my last open bed for this patient?” Key indicators include:
- Severe chest pain, suspected acute coronary syndrome (ACS)
- Respiratory distress (e.g., severe asthma, pleural effusion)
- Neurological deficits (e.g., stroke symptoms, meningitis suspicion)
- Uncontrolled bleeding or significant trauma

Evaluate Mental Status Changes: Classify patients with new onset confusion, lethargy, or disorientation as ESI Level 2. This includes:
- Suspected stroke (sudden motor/speech deficits)
- Altered mental status from metabolic, toxic, or neurological causes
- Pediatric patients with signs of sepsis or dehydration

Prioritize Severe Pain or Distress: Patients reporting severe pain (≥7/10) or severe psychological distress may qualify for Level 2 if their condition necessitates immediate intervention. Examples:
- Sickle cell crisis or acute oncologic pain
- Severe abdominal pain in elderly patients
- Trauma with high pain burden requiring urgent evaluation

** Step 3: Differentiate between ESI 3, 4 and 5 **
- Resource prediction is an integral part of the ESI for patients identified as ESI level 3 (many resources), 4 (one resource), or 5 (no resources).
- Resources include: Labs (blood, urine), ECG, X-rays, CT-MRI-ultrasound, angiography, IV fluids (hydration) , IV, IM or nebulized medications, Specialty consultation, Simple procedure (lac repair, Foley cath), Complex procedure (conscious sedation) 
- Check Danger Zone Vitals: If the patient has very high heart rate, breathing rate, or oxygen saturation below 92%, consider upgrading to ESI Level 2. Use these thresholds:
    - HR: >180 (infants), >160 (toddlers), >140 (children), >100 (older children/adults)
    - RR: >50 (infants), >40 (toddlers), >30 (children), >20 (older children/adults)
    - SaO₂: <92%

** Step 4: If it's not clear by now, determine the appropriate ESI acuity level to the best of your ability. **
     """
        },
        "FewShot": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "FewShotCoT": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk & Resource Needs **
** Step 2: Analyze and Compare with Other Patient Cases **
** Step 3: Determine the ESI acuity level for this patient based on the examples and your assessment of the patient's clinical presentation **"""
        },
        "KATE": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "KATEAutoCoT": {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "KATECoT": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk & Resource Needs **
** Step 2: Analyze and Compare with Similar Patient Cases **
** Step 3: Determine the ESI acuity level for this patient based on the examples and your assessment of the patient's clinical presentation **"""
        }
    },
    "triage-ktas": {
        "Vanilla": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate KTAS acuity level on a scale from 1 to 5. KTAS 1 indicates the highest priority and KTAS 5 indicates the lowest priority."
        },
        "CoT": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority and KTAS 5 indicates the lowest priority. Let's understand the KTAS levels and solve this step by step.

** Step 1: Understand KTAS Levels **
KTAS 1: Immediate aggressive treatment needed for life-threatening conditions. Requires immediate medical examination.

KTAS 2: Potential threats to life, limb or body function requiring quick intervention. Physician's/nurse's reevaluation needed within 15 minutes.

KTAS 3: Conditions that could lead to serious problems potentially requiring emergency intervention. Causes significant discomfort or impacts daily functions. Physician evaluation needed within 30 minutes.

KTAS 4: Condition associated with age-related factors or possibility of pain/complications. Treatment and re-verification within 1-2 hours. Physician evaluation needed within 60 minutes.

KTAS 5: Conditions from chronic problems with possible exacerbation. May be delayed or scheduled. Physician evaluation needed within 120 minutes.

** Step 2: Evaluate the Patient's Condition **
** Step 3: Determine Final KTAS Level Based on Assessment **"""
        },
        "AutoCoT": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate KTAS acuity level on a scale from 1 to 5. KTAS 1 indicates the highest priority and KTAS 5 indicates the lowest priority."
        },
        "FewShot": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Use the examples and your knowledge of KTAS protocol to determine the appropriate KTAS acuity level for the new patient."""
        },
        "FewShotCoT": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk**
** Step 2: Analyze and Compare with Other Patient Cases **
** Step 3: Determine the KTAS acuity level for this patient based on the examples, KTAS protocol, and your assessment of the patient's clinical presentation **"""
        },
        "KATE": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Use the examples and your knowledge of KTAS protocol to determine the appropriate KTAS acuity level for the new patient."""
        },
        "KATECoT": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk**
** Step 2: Analyze and Compare with Similar Patient Cases **
** Step 3: Determine the KTAS acuity level for this patient based on the examples, KTAS protocol, and your assessment of the patient's clinical presentation **"""
        },
    },
    "triage-handbook": {
        "vanillav0": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level"
        },
        "Vanilla": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "SelfConsistency": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk for ESI Level 1 **

- Is the patient dying or at immediate risk of death?
- Does the patient require an immediate life-saving intervention?

** Step 2: Evaluate Key Clinical Signs for ESI Level 2 **

Assess High-Risk Situations: Identify patients with potentially life-threatening conditions who do not require immediate life-saving interventions but should not wait. Use the rule of thumb: “Would I use my last open bed for this patient?” Key indicators include:
- Severe chest pain, suspected acute coronary syndrome (ACS)
- Respiratory distress (e.g., severe asthma, pleural effusion)
- Neurological deficits (e.g., stroke symptoms, meningitis suspicion)
- Uncontrolled bleeding or significant trauma

Evaluate Mental Status Changes: Classify patients with new onset confusion, lethargy, or disorientation as ESI Level 2. This includes:
- Suspected stroke (sudden motor/speech deficits)
- Altered mental status from metabolic, toxic, or neurological causes
- Pediatric patients with signs of sepsis or dehydration

Prioritize Severe Pain or Distress: Patients reporting severe pain (≥7/10) or severe psychological distress may qualify for Level 2 if their condition necessitates immediate intervention. Examples:
- Sickle cell crisis or acute oncologic pain
- Severe abdominal pain in elderly patients
- Trauma with high pain burden requiring urgent evaluation

** Step 3: Differentiate between ESI 3, 4 and 5 **
- Resource prediction is an integral part of the ESI for patients identified as ESI level 3 (many resources), 4 (one resource), or 5 (no resources).
- Resources include: Labs (blood, urine), ECG, X-rays, CT-MRI-ultrasound, angiography, IV fluids (hydration) , IV, IM or nebulized medications, Specialty consultation, Simple procedure (lac repair, Foley cath), Complex procedure (conscious sedation) 
- Check Danger Zone Vitals: If the patient has very high heart rate, breathing rate, or oxygen saturation below 92%, consider upgrading to ESI Level 2. Use these thresholds:
    - HR: >180 (infants), >160 (toddlers), >140 (children), >100 (older children/adults)
    - RR: >50 (infants), >40 (toddlers), >30 (children), >20 (older children/adults)
    - SaO₂: <92%

** Step 4: If it's not clear by now, determine the appropriate ESI acuity level to the best of your ability. **"""
        },
        "AutoCoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "DemonstrationCoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Here's a demonstration of how to do this:

Case: A 34-year-old obese female presents to triage complaining of generalized abdominal pain (pain scale rating: 6/10) for 2 days. She has vomited several times and states her last bowel movement was 3 days ago. She has a history of back surgery, takes no medications, and is allergic to peanuts. Vital signs: T 97.8° F, HR 104, RR 16, BP 132/80, SpO2 99 percent.
Rationale: This patient will need a minimum of two or more resources: lab, IV fluids, perhaps IV medication for nausea, and a CT scan. The triage nurse would review the patient's vital signs and consider the heart rate. The heart rate falls just outside the accepted parameter for the age of the patient but could be due to pain or exertion. In this case, the decision should be to assign the patient to ESI level 3."""
        },
        "CoT": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk for ESI Level 1 **

- Is the patient dying or at immediate risk of death?
- Does the patient require an immediate life-saving intervention?

** Step 2: Evaluate Key Clinical Signs for ESI Level 2 **

Assess High-Risk Situations: Identify patients with potentially life-threatening conditions who do not require immediate life-saving interventions but should not wait. Use the rule of thumb: “Would I use my last open bed for this patient?” Key indicators include:
- Severe chest pain, suspected acute coronary syndrome (ACS)
- Respiratory distress (e.g., severe asthma, pleural effusion)
- Neurological deficits (e.g., stroke symptoms, meningitis suspicion)
- Uncontrolled bleeding or significant trauma

Evaluate Mental Status Changes: Classify patients with new onset confusion, lethargy, or disorientation as ESI Level 2. This includes:
- Suspected stroke (sudden motor/speech deficits)
- Altered mental status from metabolic, toxic, or neurological causes
- Pediatric patients with signs of sepsis or dehydration

Prioritize Severe Pain or Distress: Patients reporting severe pain (≥7/10) or severe psychological distress may qualify for Level 2 if their condition necessitates immediate intervention. Examples:
- Sickle cell crisis or acute oncologic pain
- Severe abdominal pain in elderly patients
- Trauma with high pain burden requiring urgent evaluation

** Step 3: Differentiate between ESI 3, 4 and 5 **
- Resource prediction is an integral part of the ESI for patients identified as ESI level 3 (many resources), 4 (one resource), or 5 (no resources).
- Resources include: Labs (blood, urine), ECG, X-rays, CT-MRI-ultrasound, angiography, IV fluids (hydration) , IV, IM or nebulized medications, Specialty consultation, Simple procedure (lac repair, Foley cath), Complex procedure (conscious sedation) 
- Check Danger Zone Vitals: If the patient has very high heart rate, breathing rate, or oxygen saturation below 92%, consider upgrading to ESI Level 2. Use these thresholds:
    - HR: >180 (infants), >160 (toddlers), >140 (children), >100 (older children/adults)
    - RR: >50 (infants), >40 (toddlers), >30 (children), >20 (older children/adults)
    - SaO₂: <92%

** Step 4: If it's not clear by now, determine the appropriate ESI acuity level to the best of your ability. **
     """
        },
        "FewShot": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "FewShotCoT": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk & Resource Needs **
** Step 2: Analyze and Compare with Other Patient Cases **
** Step 3: Determine the ESI acuity level for this patient based on the examples and your assessment of the patient's clinical presentation **"""
        },
        "KATE": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "KATEAutoCoT": {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "KATECoT": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk & Resource Needs **
** Step 2: Analyze and Compare with Similar Patient Cases **
** Step 3: Determine the ESI acuity level for this patient based on the examples and your assessment of the patient's clinical presentation **"""
        }
    },
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
    if 'triage-mimic' in dataset.lower():
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
                suspicion=row.get('Diagnosis in ED', 'N/A'),
                injury=row.get('Injury', 'N/A'),
                arrival=row.get('Arrival mode', 'N/A'),
                mental=row.get('Mental', 'N/A')
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
                injury_text = " who sustained an injury"
        
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
    elif dataset.lower() == 'triage-handbook':
        return row.get('Clinical Vignettes', 'N/A')


# Function to convert a CSV row into instruction, input, and output strings.
def format_instruction_prompt_for_blackbox(row, strategy='Vanilla', dataset='triage-mimic', serialization='natural', return_json=False, examples=None):
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
            if 'CoT' in strategy else
            """ Output your response strictly in JSON format with the following key:
"Acuity": <the assigned acuity level as an integer between 1 and 5>"""
        )
        prompt['system'] += json_instruction
    
    # Handle different strategy types
    if 'KATE' in strategy or 'FewShot' in strategy:
        if not examples:
            raise ValueError(f"Strategy {strategy} requires examples but none were provided")
        prompt['user'] = BLACKBOX_LLM_PROMPT_WITH_EXAMPLES.format(
            prompt['user'],
            examples,
            patient_description
        )
    else:
        prompt['user'] = BLACKBOX_LLM_PROMPT.format(
            prompt['user'],
            patient_description
        )

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
        text = ALPACA_PROMPT.format(instruction=instruction, input=patient_description, response=output_text) + EOS_TOKEN
    elif split == 'test':
        text = ALPACA_PROMPT.format(instruction=instruction, input=patient_description, response='')
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
