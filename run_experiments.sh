#!/bin/bash

# Base command
BASE_CMD="python llm-predict-triage.py --dataset Triage-MIMIC --model openai-gpt-4o-high-quota-chat --json"

# Function to run experiment with delay
run_with_delay() {
    echo "Starting experiment: $1"
    eval $1
    echo "Completed. Sleeping for for a bit..."
    sleep 25  # 5 minute delay between runs
}

# Run CoT with natural serialization
CMD="$BASE_CMD --strategy AutoCoT --serialization natural"
run_with_delay "$CMD"

# Run Vanilla with different serializations
for serial in "natural" "spaces" "commas" "newline"; do
    CMD="$BASE_CMD --strategy Vanilla --serialization $serial"
    run_with_delay "$CMD"
done

echo "All experiments completed!"