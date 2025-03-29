# MODEL="meta-llama/meta-Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
DATASET="gsm8k"
MODE="train"
LORA_NAME="v1"
DATE=$(date +"%Y-%m-%d_%H_%M")
nohup python train_grpo.py --model ${MODEL} --mode ${MODE} --dataset ${DATASET} --lora_name ${LORA_NAME} > logs/${MODEL_NAME}_${LORA_NAME}_${MODE}_${DATASET}_${DATE}.out 2>&1 &
