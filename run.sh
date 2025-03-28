MODEL="llama_3.1_8b_instruct"
VERSION="v0"
nohup python notebooks+scripts/grpo.py --mode evaluate  --dataset gsm8k > v0_0326.out 2>&1 &
