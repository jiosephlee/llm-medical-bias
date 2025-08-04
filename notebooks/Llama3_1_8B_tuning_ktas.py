from unsloth import FastLanguageModel
import torch
import re 
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
import numpy as np
import pandas as pd
from datasets import Dataset

import sys
import os
sys.path.append(os.path.abspath(".."))
from importlib import reload
import utils.utils as utils

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

import time
print("Sleeping...")

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "qwen3_8B_full_bit", #"unsloth/Meta-Llama-3.1-8B-Instruct", #"./llama_3_1_8B_ESI_Handbook_10",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
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

# ---- 1. Define helper functions ----

# Function to convert a CSV row into instruction, input, and output strings.
def format_example(row):
    # Create a natural language description of the patient.
    patient_description = utils.format_row(row, dataset='triage-mimic')
    
    # Define the instruction for the model.
    instruction = "Based on their clinical presentation, determine the Emergency Severity Index (ESI) acuity for the following patient."
    
    # The input is the patient description.
    input_text = patient_description
    
    # The expected output is a formatted acuity statement.
    output_text = f"The ESI acuity for this patient is {row['acuity']}."
    
    # Return a dict that contains our three keys (and optionally the label for evaluation).
    return {"instruction": instruction, "input": input_text, "output": output_text, "label": row["acuity"]}

# ---- 2. Load the CSV and convert to a Dataset ----

# Read your CSV data.
train_df = pd.read_csv("../data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/small_train_dataset.csv")

# Convert the pandas DataFrame into a Hugging Face Dataset.
dataset = Dataset.from_pandas(train_df)

# Map our serialization function over each example.
dataset = dataset.map(lambda x: format_example(x))

# ---- 3. Format the examples into a single prompt string ----

# Define the Alpaca-style prompt template.
alpaca_prompt = """### Instruction: {}

### Input: {}

### Response: {}"""

lima_prompt = """{}
"""

# Function to wrap our instruction, input, and output into a single text string.
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Append EOS_TOKEN to avoid infinite generation.
        text = alpaca_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def formatting_prompts_func_test(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Append EOS_TOKEN to avoid infinite generation.
        text = alpaca_prompt.format(instruction, input_text, "")
        texts.append(text)
    return {"text": texts}


# Map the formatting function over the dataset (batched for efficiency).
dataset = dataset.map(formatting_prompts_func, batched=True)

# ---- At this point, your dataset is in the format expected by your code. ----
# Each example in the dataset now has a "text" field containing your full prompt.
print(dataset[1])

test_df = pd.read_csv("../data/mimic-iv-private/anchor_year_group_datasets/2017_-_2019/test_dataset.csv")
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(lambda x: format_example(x))
test_dataset = test_dataset.map(formatting_prompts_func_test, batched=True)

# response_template = "### Response: "
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "qwen3_8B_full_bit+_MIMIC_10",
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()

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
    input_text = sample['text']
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

# Calculate rates based on the samples where a valid prediction was obtained
total_samples = len(y_true)
undertriage_rate = undertriage / total_samples * 100
overtriage_rate = overtriage / total_samples * 100
outside_by_2_rate = outside_by_2 / total_samples * 100

# Calculate additional evaluation metrics
accuracy = correct / (correct + wrong) * 100
qwk_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
mse = mean_squared_error(y_true, y_pred)

# Print out the results
print(f"Model Accuracy: {accuracy:.2f}%")
print(f"Quadratic Weighted Kappa (QWK): {qwk_score:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Undertriage Rate: {undertriage_rate:.2f}%")
print(f"Overtriage Rate: {overtriage_rate:.2f}%")
# Compute F1 score (binary classification by default)
f1 = f1_score(y_true, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Outside-by-2 Rate: {outside_by_2_rate:.2f}%")

# model.save_pretrained("llama_3_1_8B_+_2014_-_2016_Small_10_Diff_Params")  # Local saving
