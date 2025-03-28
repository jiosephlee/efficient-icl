from unsloth import FastLanguageModel
from vllm import SamplingParams
import torch
import re
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from tqdm import tqdm
import json
import sys
import os
import argparse
import datetime
import logging
import utils

# Parse command line arguments
parser = argparse.ArgumentParser(description="GRPO training and evaluation script")
parser.add_argument("--model", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", 
                    help="Model to load for training or evaluation")
parser.add_argument("--mode", type=str, choices=["train", "evaluate", "continue"], default="train",
                    help="Mode to run the script in: train (also evaluates), evaluate only, or continue training")
parser.add_argument("--lora_name", type=str, help="Name of the LoRA adapter to save or load")
parser.add_argument("--dataset", type=str, default="gsm8k",
                    help="Dataset to train or evaluate on (default: gsm8k)")
parser.add_argument("--checkpoint_for_continued_training", type=str, help="Path to checkpoint for continuing training")

args = parser.parse_args()

# Create experiment name to centralize logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{args.dataset}_{args.model.split('/')[-1]}_{args.lora_name}_{args.mode}_{timestamp}"

# Setup logging to the output directory
log_file = f"./logs/{experiment_name}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Starting experiment: {experiment_name}")
logger.info(f"Arguments: {args}")

# Check if LoRA already exists when training
if args.mode == "train" and os.path.exists(args.lora_name):
    raise ValueError(f"Warning: LoRA adapter '{args.lora_name}' already exists. It will be overwritten.")

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
logger.info(f"Using max_seq_length={max_seq_length}, lora_rank={lora_rank}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)
logger.info(f"Model loaded: {args.model}")

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    # lora_dropout = 0,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
logger.info("PEFT model configured")

if args.mode == "train" or args.mode == "continue":
    logger.info(f"Loading {args.dataset} training dataset")
    dataset = utils.get_dataset(args.dataset, "train")
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    output_dir = f"{args.model.split('/')[-1]}/{args.lora_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    max_prompt_length = 512

    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 6, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        output_dir = f"./checkpoints/{output_dir}",
    )
    logger.info(f"Training configuration: {training_args}")

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            utils.xmlcount_reward_func,
            utils.soft_format_reward_func,
            utils.strict_format_reward_func,
            utils.int_reward_func,
            utils.correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    logger.info("Starting training")
    
    if args.mode == "continue":
        logger.info("Continuing training from checkpoint")
        if args.checkpoint_path:
            logger.info(f"Using specified checkpoint: {args.checkpoint_path}")
            trainer.train(resume_from_checkpoint=args.checkpoint_path)
        else:
            logger.info("Automatically finding latest checkpoint")
            trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    logger.info("Training completed")
    
    # Add timestamp to lora_name if continuing training
    lora_save_name = args.lora_name
    if args.mode == "continue":
        lora_base_name = args.lora_name
        lora_save_name = f"models/{args.model.split('/')[-1]}/{lora_base_name}_{timestamp}"

    logger.info(f"Saving LoRA adapter to {lora_save_name}")
    model.save_lora(lora_save_name)

    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : utils.SYSTEM_PROMPT},
        {"role" : "user", "content" : "Calculate pi."},
    ], tokenize = False, add_generation_prompt = True)

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    logger.info("Output without LoRA:")
    logger.info(output)

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora(lora_save_name),
    )[0].outputs[0].text

    logger.info("Output with LoRA:")
    logger.info(output)

# Evaluation mode (runs for both train and evaluate modes)
logger.info(f"Loading {args.dataset} test dataset for evaluation")
test_dataset = utils.get_dataset(args.dataset, "test")
logger.info(f"Test dataset loaded with {len(test_dataset)} examples")

# Run evaluation with the model
logger.info("Starting evaluation")
lora_path = args.lora_name if args.mode == "evaluate" else (lora_save_name if 'lora_save_name' in locals() else args.lora_name)
results = utils.evaluate_model(
    model, 
    test_dataset, 
    tokenizer, 
    lora_path=lora_path
)

# Create directory for results if it doesn't exist
results_dir = f"models/{args.model.split('/')[-1]}/{lora_base_name}_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Save detailed results to file in the model-specific directory
results_filename = f"./{results_dir}/results.json"
with open(results_filename, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"Detailed results saved to {results_filename}")

utils.analyze_errors(results)

logger.info(f"Accuracy: {results['accuracy']:.2f}%")
utils.analyze_errors(results)
logger.info(f"Experiment completed. Log saved to {log_file}")