#!/bin/bash

# Base command
BASE_CMD="python llm-predict-triage.py --dataset Triage-MIMIC --model openai-gpt-4o-high-quota-chat --json"

# Function to run experiment with delay
run_with_delay() {
    echo "Starting experiment: $1"
    eval $1
    echo "Completed. Sleeping for for a bit..."
    sleep 10  # 5 minute delay between runs
}

# Run KATE with different numbers of shots
for shots in 3 5 10 20 30 40 50; do
    CMD="$BASE_CMD --strategy KATE --serialization natural --k_shots $shots --k_shots_ablation"
    run_with_delay "$CMD"
done

# Run FewShot with different numbers of shots
for shots in 3 5 10 20 30 40 50; do
    CMD="$BASE_CMD --strategy FewShot --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

# Run FewShot with different numbers of shots
for shots in 40; do
    CMD="$BASE_CMD --strategy FewShotCoT --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

# Run FewShot with different numbers of shots
for shots in 40; do
    CMD="$BASE_CMD --strategy KATECoT --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

echo "All experiments completed!"
