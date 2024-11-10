from config import OPENAI_API_KEY, TOGETHER_API_KEY
from openai import OpenAI
from together import Together
import os
import time 
import json 

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
client = OpenAI()
client_tog = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

_DATASETS = {
    "MedQA": "./data/medqa/questions/US/4_options/phrases_no_exclude_test.jsonl",
    "OtherDataset": "data/path/to/other_dataset.jsonl"
    }

def load_dataset(filepath, max_questions=200):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if i < max_questions]
    return data

def load_predictions(timestamp, save_path="./results/"):
    predictions_file = os.path.join(save_path, f"{timestamp}_predictions.txt")
    with open(predictions_file, 'r') as f:
        predictions = [json.loads(line.strip()) for line in f]
    return predictions
        
def query_gpt(prompt, max_tokens=100, temperature=0, max_try_num=10, model="gpt-3.5-turbo", debug=False, json=False):
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
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1,
                    stop=["<|eot_id|>"]
                )
            else:
                if json:
                    response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"}
                    )  
                else:
                    response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                    )   
            
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