#!/bin/bash

# Define an array of model configurations
# Each entry contains: MODEL_PATH MODEL_NAME LORA_NAME MODE [ADDITIONAL_FLAGS]
declare -a MODELS=(
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct v1_3_few_shot_chat evaluate --eval_zero_shot --eval_few_shot --eval_k_shot 4"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct v1_2_few_shot_chat train --eval_zero_shot --eval_few_shot --eval_k_shot 4"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct v1_1_few_shot_chat train --eval_zero_shot --eval_few_shot --eval_k_shot 4"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct Base_v1 evaluate --eval_few_shot --eval_k_shot 4"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct Base_v2 evaluate --eval_few_shot --eval_k_shot 4"
    "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct Base_v3 evaluate --eval_few_shot --eval_k_shot 4"
    )

DATASET="gsm8k"
DATE=$(date +"%Y-%m-%d_%H_%M")

# Build a command chain that runs each model sequentially
CMD=""
for MODEL_CONFIG in "${MODELS[@]}"; do
    # Split the configuration into model path, name, lora name, mode, and any additional flags
    read -r MODEL_PATH MODEL_NAME LORA_NAME MODE ADDITIONAL_FLAGS <<< "$MODEL_CONFIG"
    
    if [ -n "$CMD" ]; then
        CMD+=" && "
    fi
    
    CMD+="python train_grpo_unsloth.py --model \"${MODEL_PATH}\" --mode ${MODE} --dataset ${DATASET} --lora_name ${LORA_NAME} ${ADDITIONAL_FLAGS} > logs/${MODEL_NAME}_${LORA_NAME}_${MODE}_${DATASET}_${DATE}.out 2>&1"
done

# Run the command chain in the background
nohup bash -c "$CMD" &
