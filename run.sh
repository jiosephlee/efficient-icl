MODEL="meta-llama/meta-Llama-3.1-8B-Instruct"
MODEL_NAME="meta-Llama-3.1-8B-Instruct"
DATASET="gsm8k"
MODE="evaluate"
LORA_NAME="v1"
DATE=$(date +"%Y%m%d_%H%M")
nohup python python train_grpo.py --model ${MODEL} --mode ${MODE} --dataset ${DATASET} --lora_name ${LORA_NAME} > ${MODEL_NAME}_${LORA_NAME}_${MODE}_${DATASET}_${DATE}.out 2>&1 &
