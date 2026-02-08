# !pip install unsloth
#!pip install xformers

# If above doesn't work, try:
# !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
# !pip install --no-deps cut_cross_entropy unsloth_zoo
# !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
# !pip install --no-deps unsloth

import sys
import os
sys.path.append(os.path.abspath(".."))
import utils.utils as utils
import utils.prompts as prompts
from importlib import reload

reload(utils)
reload(prompts)

from unsloth import FastLanguageModel
import torch
import re 
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
from datasets import Dataset
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datetime import datetime

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


def main():
    parser = argparse.ArgumentParser(description="Finetuning script for medical LLMs.")
    parser.add_argument('--cpt', action='store_true', help='Enable Continuous Pre-Training.')
    parser.add_argument('--para', type=int, default=1, choices=[1, 5, 10], help='Level of paraphrasing for CPT handbook. 0 for original, 5 for 5x, 10 for 10x.')
    parser.add_argument('--dataset', type=str, required=True, choices=['handbook', 'ktas', 'mimic'], help='Dataset to use for finetuning.')
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen2.5-7B", help='Name of the model to use for finetuning.')
    parser.add_argument('--device_batch_size', type=int, default=2, help='Batch size per device during training.')
    parser.add_argument('--peft', action='store_true', help='Enable PEFT for finetuning.')
    args = parser.parse_args()

    assert 8 % args.device_batch_size == 0 and args.device_batch_size <= 8, "Batch size must be a divisor of 8 and not larger than 8."

    if args.peft:
        cpt_learning_rate = 4e-5
        finetuning_learning_rate = 2e-4
    else:
        cpt_learning_rate = 15e-6
        finetuning_learning_rate = 2e-5

    if args.dataset == 'handbook':
        num_train_epochs = 20
    elif args.dataset == 'ktas':
        num_train_epochs = 20
    else: # mimic
        num_train_epochs = 10

    cpt_num_train_epochs = 0
    if args.cpt:
        if args.para > 0:
            cpt_num_train_epochs = int(30 / args.para)
        else:
            cpt_num_train_epochs = 30

    # Create a base name for runs and files
    filename_parts = [
        args.model_name.split("/")[-1], 
        args.dataset,
        f"epochs{num_train_epochs}",
        f"lr{finetuning_learning_rate}"
    ]
    if args.cpt:
        filename_parts.append(f"cpt{cpt_num_train_epochs}")
        if args.para > 1:
            filename_parts.append(f"para{args.para}")
    if args.peft:
        filename_parts.append("peft")
    
    filename = "_".join(filename_parts)
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    filename_with_timestamp = f"{filename}_{timestamp}"

    max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 1024,   # Context length - can be longer, but uses more memory
        #load_in_4bit = load_in_4bit,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = not args.peft, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )

    if args.peft:
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

    EOS_TOKEN = tokenizer.eos_token


    if args.cpt:
        #  --- DATA for CPT -----

        # Load the cleaned text file
        if args.para == 1:
            handbook_path = "../data/ESI-Handbook/cleaned_handbook.txt"
        else:
            handbook_path = f"../data/ESI-Handbook/paragraphed_{args.para}x_handbook.txt"

        with open(handbook_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Further clean the text
        def remove_newline_between_chars(text):
            return re.sub(r"(\S)\n(\S)", r"\1 \2", text)

        # Example usage
        cleaned_text = remove_newline_between_chars(text)

        # Tokenize and chunk text
        def chunk_text(text, max_length):
            """Splits text into chunks while maintaining sentence boundaries."""
            import re
            sentences = re.split(r"(?<=[.!?]) +", text)  # Split by sentence
            chunks, current_chunk = [], ""
            
            for sentence in sentences:
                if len(tokenizer.encode(current_chunk + sentence)) < max_length:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

        # Create chunks
        chunks = chunk_text(cleaned_text, max_seq_length)

        # Add EOS token at the end of each chunk
        
        formatted_chunks = [{"text": chunk} for chunk in chunks]

        # Convert to Hugging Face dataset
        handbook = Dataset.from_list(formatted_chunks)

        # Function for formatting
        def formatting_prompts_func(examples):
            return {"text": [example for example in examples["text"]]}

        # Apply formatting function
        handbook = handbook.map(formatting_prompts_func, batched=True)

        ## --- CPT ---- 

        if args.para > 0:
            cpt_num_train_epochs = int(30 / args.para)
        else:
            cpt_num_train_epochs = 30

        trainer = UnslothTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = handbook,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 4,
            args = UnslothTrainingArguments(
                run_name = f"CPT_{filename_with_timestamp}",
                per_device_train_batch_size = args.device_batch_size,
                gradient_accumulation_steps = int(16/int(args.device_batch_size)),

                warmup_steps=5,
                num_train_epochs = cpt_num_train_epochs,

                learning_rate = cpt_learning_rate,
                # embedding_learning_rate = 5e-6,

                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 3407,
                report_to = "wandb", # Use this for WandB etc
            ),
        )

        trainer_stats = trainer.train()
        # print(trainer_stats)

    ###  ----Finetuning ---- 
    true_acuity_col = 'acuity'
    if args.dataset == 'handbook':
        # Read your CSV data.
        train_df = pd.read_csv("../data/ESI-Handbook/train.csv")
        test_df = pd.read_csv("../data/ESI-Handbook/test.csv")

        # Convert the pandas DataFrame into a Hugging Face Dataset.
        dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Map our serialization function over each example.
        dataset = dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-handbook'))
        test_dataset = test_dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-handbook',split='test'))
        true_acuity_col = 'acuity'
    
    elif args.dataset == 'ktas':
        train_df = pd.read_csv("../data/kaggle/train.csv")
        test_df = pd.read_csv("../data/kaggle/test.csv")
        dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        dataset = dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-ktas'))
        test_dataset = test_dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-ktas',split='test'))
        true_acuity_col = 'KTAS_expert'
    
    elif args.dataset == 'mimic':
        train_df = pd.read_csv("../data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/small_train_dataset.csv")
        test_df = pd.read_csv("../data/mimic-iv-private/anchor_year_group_datasets/2017_-_2019/test_dataset.csv")
        dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        dataset = dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-mimic'))
        test_dataset = test_dataset.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, EOS_TOKEN, dataset='triage-mimic',split='test'))
        true_acuity_col = 'acuity'

    trainer = SFTTrainer(
        run_name=f"Finetuning-{args.dataset}",
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 4,
        packing = True, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            run_name=f"Finetune_{filename_with_timestamp}",
            per_device_train_batch_size = args.device_batch_size,
            gradient_accumulation_steps = int(8/args.device_batch_size),
            num_train_epochs = num_train_epochs, # Set this for 1 full training run.
            learning_rate = finetuning_learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            warmup_ratio=0.1,
            lr_scheduler_type = "cosine",
            seed = 3407,
            report_to = "wandb", # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()
    # print(trainer_stats)

    # Inference for ESI

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
        #print(decoded_output)
        return extract_response(decoded_output)

    # Iterate through test dataset
    for i, sample in tqdm(enumerate(test_dataset)):
        input_text = sample['text']
        true_acuity = sample[true_acuity_col]
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
    
    # Calculate overtriage and undertriage rates
    total_predictions = len(y_pred)
    if total_predictions > 0:
        overtriage_rate = overtriage / total_predictions
        undertriage_rate = undertriage / total_predictions
        outside_by_2_rate = outside_by_2 / total_predictions
    else:
        overtriage_rate = 0.0
        undertriage_rate = 0.0
        outside_by_2_rate = 0.0
    
    # Add triage metrics to results
    metrics["overall"]["overtriage_count"] = overtriage
    metrics["overall"]["undertriage_count"] = undertriage
    metrics["overall"]["overtriage_rate"] = overtriage_rate
    metrics["overall"]["undertriage_rate"] = undertriage_rate
    metrics["overall"]["outside_by_2_count"] = outside_by_2
    metrics["overall"]["outside_by_2_rate"] = outside_by_2_rate
    
    print("Overall Metrics:", metrics)
    print(f"\nTriage Analysis:")
    print(f"  Overtriage:  {overtriage} ({overtriage_rate:.2%})")
    print(f"  Undertriage: {undertriage} ({undertriage_rate:.2%})")
    print(f"  Off by ≥2:   {outside_by_2} ({outside_by_2_rate:.2%})")
    
    output_filepath = f"../results/Triage-{args.dataset.upper() if args.dataset != 'handbook' else 'Handbook'}/{filename_with_timestamp}"
    utils.save_metrics(metrics,output_filepath)
    print("Evaluation complete. Metrics and plots saved.")

if __name__ == '__main__':
    main()