#!/bin/bash

# Define an array of model configurations
# Each entry contains: MODEL_PATH MODEL_NAME LORA_NAME MODE
declare -a MODELS=(
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct Base evaluate"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct v0 train"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct v0_few_shot train"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct v0_few_shot_combined train"
)

DATASET="gsm8k"
DATE=$(date +"%Y-%m-%d_%H_%M")

# Build a command chain that runs each model sequentially
CMD=""
for MODEL_CONFIG in "${MODELS[@]}"; do
    read -r MODEL_PATH MODEL_NAME LORA_NAME MODE <<< "$MODEL_CONFIG"
    
    if [ -n "$CMD" ]; then
        CMD+=" && "
    fi
    
    CMD+="python train_grpo_unsloth.py --model \"${MODEL_PATH}\" --mode ${MODE} --dataset ${DATASET} --lora_name ${LORA_NAME} > logs/${MODEL_NAME}_${LORA_NAME}_${MODE}_${DATASET}_${DATE}.out 2>&1"
done

# Run the command chain in the background
nohup bash -c "$CMD" &
