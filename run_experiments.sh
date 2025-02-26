#!/bin/bash

# Base command
BASE_CMD="python llm-predict-triage.py --dataset Triage-Handbook --model gpt-4o --json"

# Function to run experiment with delay
run_with_delay() {
    echo "Starting experiment: $1"
    eval $1
    echo "Completed. Sleeping for for a bit..."
    sleep 10  # 5 minute delay between runs
}

# Run Vanilla strategy
CMD="$BASE_CMD --strategy Vanilla --serialization natural"
run_with_delay "$CMD"

# Run CoT strategy
CMD="$BASE_CMD --strategy CoT --serialization natural"
run_with_delay "$CMD"

# Run KATE with different numbers of shots
for shots in 20 40; do
    CMD="$BASE_CMD --strategy KATE --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

# Run FewShot with different numbers of shots
for shots in 20 40; do
    CMD="$BASE_CMD --strategy FewShot --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

# Run FewShotCoT with different numbers of shots
for shots in 20 40; do
    CMD="$BASE_CMD --strategy FewShotCoT --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

# Run KATECoT with different numbers of shots
for shots in 20 40; do
    CMD="$BASE_CMD --strategy KATECoT --serialization natural --k_shots $shots"
    run_with_delay "$CMD"
done

echo "All experiments completed!"
