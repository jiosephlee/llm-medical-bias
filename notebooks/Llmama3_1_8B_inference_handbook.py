from unsloth import FastLanguageModel
import torch
import re 
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
from datasets import Dataset

import sys
import os
sys.path.append(os.path.abspath(".."))
from importlib import reload
import utils.utils as utils
import utils.prompts as prompts

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

import time
print("Sleeping...")

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "./llama_3_1_8B_ESI_Handbook_10_+_ESI_Case_Examples_25, #./llama_3_1_8B_ESI_Handbook_10",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

reload(utils)
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs       = examples["input"]
#     outputs      = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }
# pass

# from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched = True,)

EOS_TOKEN = tokenizer.eos_token  # Ensure EOS_TOKEN is defined

# Read your CSV data.
train_df = pd.read_csv("../data/ESI-Handbook/train.csv")

# Convert the pandas DataFrame into a Hugging Face Dataset.
dataset = Dataset.from_pandas(train_df)

# Map our serialization function over each example.
dataset = dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-handbook'))


test_df = pd.read_csv("../data/ESI-Handbook/train.csv")
test_dataset = Dataset.from_pandas(test_df)
test_dataset = dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-handbook',split='test'))


from sklearn.metrics import cohen_kappa_score, mean_squared_error

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

def extract_response(text):
    # This pattern looks for "Response:" and then non-greedily skips any characters until it finds a sequence of digits.
    match = re.search(r"Response:.*?(\d+)", text, re.DOTALL)
    return int(match.group(1)) if match else None

# Initialize tracking variables
correct = 0
wrong = 0
y_true = []
y_pred = []
undertriage = 0
overtriage = 0
outside_by_2 = 0  # New variable to track predictions off by 2 or more

def generate_response(input_text):
    inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=75, use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return extract_response(decoded_output)

# Iterate through test dataset
for i, sample in tqdm(enumerate(test_dataset)):
    input_text = sample['Clinical Vignettes']
    true_acuity = sample['acuity']
    predicted_acuity = generate_response(input_text)
    
    if predicted_acuity is not None:
        # Uncomment for debugging if needed:
        # print(f"Sample {i}: True Acuity: {true_acuity}, Predicted: {predicted_acuity}")
        y_true.append(true_acuity)
        y_pred.append(predicted_acuity)
        
        if predicted_acuity == true_acuity:
            correct += 1
        else:
            wrong += 1
            
            # Track undertriage and overtriage (directional errors)
            if predicted_acuity < true_acuity:
                undertriage += 1
            elif predicted_acuity > true_acuity:
                overtriage += 1
            
            # Track predictions that differ from the ground truth by 2 or more
            if abs(predicted_acuity - true_acuity) >= 2:
                outside_by_2 += 1
    else:
        print(f"Sample {i}: No valid response extracted.")
        wrong += 1


metrics = utils.evaluate_predictions(y_pred, y_true, ordinal=True, by_class=True)
print("Overall Metrics:", metrics)
output_filepath = '../results/Triage-Handbook/Triage-Handbook-Llama_3.1'
utils.save_metrics(metrics,output_filepath)
print("Evaluation complete. Metrics and plots saved.")