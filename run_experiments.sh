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

# Run Vanilla with different serializations
for serialization in commas newline spaces; do
    CMD="$BASE_CMD --strategy Vanilla --serialization $serialization"
    run_with_delay "$CMD"
done

echo "All experiments completed!"
