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
import rewards
from unsloth_config import get_config, TrainingConfig

# Parse command line arguments
parser = argparse.ArgumentParser(description="GRPO training and evaluation script")
parser.add_argument("--model", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", 
                    help="Model to load for training or evaluation")
parser.add_argument("--mode", type=str, choices=["train", "train_no_evaluate", "evaluate", "continue"], default="train",
                    help="Mode to run the script in: train (also evaluates), train_no_evaluate (does not evaluate), evaluate only, or continue training")
parser.add_argument("--lora_name", type=str, help="Name of the LoRA adapter to save or load")
parser.add_argument("--dataset", type=str, default="gsm8k",
                    help="Dataset to evaluate on (default: gsm8k)")
parser.add_argument("--checkpoint_for_continued_training", type=str, help="Path to checkpoint for continuing training")
parser.add_argument("--eval_zero_shot", action="store_true", help="Whether to evaluate in zero-shot setting")
parser.add_argument("--eval_few_shot", action="store_true", help="Whether to evaluate in few-shot setting")
parser.add_argument("--eval_few_shot_train", action="store_true", help="Whether to evaluate in train few-shot setting")
parser.add_argument("--eval_k_shot", type=int, default=4, help="Number of examples to use for few-shot evaluation")

args = parser.parse_args()

# Track experiment
import wandb
wandb.login(key='d385cbc08ef0c734e84aff78ce2bb293b07f34e0')
import os
os.environ["WANDB_PROJECT"]="GRPO_Few-Shot-Learning"


# Get the training configuration based on lora_name
CONFIG = get_config(args.lora_name)
CONFIG.model_name = args.model

# Create experiment name to centralize logging
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
experiment_name = f"{args.model.split('/')[-1]}_{args.lora_name}_{args.mode}_{args.dataset}_{timestamp}"
# Tracking model-specific weights results, and metrics
model_dir = f"{args.model.split('/')[-1]}/{args.lora_name}/"

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
logger.info(f"Using config: {CONFIG.lora_name}")

# Check if LoRA already exists when training
if args.mode == "train" and os.path.exists(args.lora_name):
    raise ValueError(f"Warning: LoRA adapter '{args.lora_name}' already exists. It will be overwritten.")

max_seq_length = CONFIG.max_seq_length
lora_rank = CONFIG.lora_rank
logger.info(f"Using max_seq_length={max_seq_length}, lora_rank={lora_rank}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model,
    max_seq_length = max_seq_length,
    load_in_4bit = CONFIG.load_in_4bit,
    fast_inference = CONFIG.fast_inference,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = CONFIG.gpu_memory_utilization,
)
logger.info(f"Model loaded: {args.model}")

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = CONFIG.get_target_modules(model),
    lora_alpha = CONFIG.lora_alpha,
    lora_dropout = CONFIG.lora_dropout,
    use_gradient_checkpointing = CONFIG.use_gradient_checkpointing,
    random_state = CONFIG.random_state,
)
logger.info("PEFT model configured")

if args.mode == "train" or args.mode == "continue" or args.mode == 'train_no_evaluate':
    logger.info(f"Loading {CONFIG.train_dataset} training dataset")
    dataset = utils.get_dataset(CONFIG.train_dataset, 
                                "train", 
                                CONFIG.prompt_version,
                                few_shot=CONFIG.few_shot, 
                                k_shot=CONFIG.k_shot, 
                                few_shot_template=CONFIG.few_shot_template)
    logger.info(f"Dataset loaded with {len(dataset)} examples")

    max_prompt_length = CONFIG.max_prompt_length

    training_args = GRPOConfig(
        learning_rate = CONFIG.learning_rate,
        adam_beta1 = CONFIG.adam_beta1,
        adam_beta2 = CONFIG.adam_beta2,
        weight_decay = CONFIG.weight_decay,
        warmup_ratio = CONFIG.warmup_ratio,
        lr_scheduler_type = CONFIG.lr_scheduler_type,
        optim = CONFIG.optim,
        logging_steps = 1,
        per_device_train_batch_size = CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps = CONFIG.gradient_accumulation_steps, 
        num_generations = CONFIG.num_generations, 
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = CONFIG.num_train_epochs,
        max_steps = CONFIG.max_steps,
        save_steps = CONFIG.save_steps,
        max_grad_norm = CONFIG.max_grad_norm,
        # scale_rewards = CONFIG.scale_rewards,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = CONFIG.get_checkpoint_dir(),
    )
    logger.info(f"Training configuration: {training_args}")

    # Create reward functions list from config
    reward_funcs = []
    for func_name in CONFIG.reward_functions:
        if hasattr(rewards, func_name):
            reward_funcs.append(getattr(rewards, func_name))
        else:
            logger.warning(f"Reward function {func_name} not found in rewards module")

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )
    logger.info("Starting training")
    
    if args.mode == "continue":
        logger.info("Continuing training from checkpoint")
        if args.checkpoint_for_continued_training:
            logger.info(f"Using specified checkpoint: {args.checkpoint_for_continued_training}")
            trainer.train(resume_from_checkpoint=args.checkpoint_for_continued_training)
        else:
            logger.info("Automatically finding latest checkpoint")
            trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    logger.info("Training completed")
    
    # Save Lora Adapter
    lora_file_path = CONFIG.get_model_dir()
    logger.info(f"Saving LoRA adapter to {lora_file_path}")
    model.save_lora(lora_file_path)

    # Model sample output
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : utils.SYSTEM_PROMPT},
        {"role" : "user", "content" : "Calculate pi."},
    ], tokenize = False, add_generation_prompt = True)

    sampling_params = SamplingParams(
        temperature = CONFIG.sampling_temperature,
        top_p = CONFIG.sampling_top_p,
        max_tokens = CONFIG.max_tokens,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    logger.info("Output without LoRA:")
    logger.info(output)
    
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora(lora_file_path),
    )[0].outputs[0].text

    logger.info("Output with LoRA:")
    logger.info(output)

if args.mode != "train_no_evaluate":
    # Evaluation mode (runs for both train and evaluate modes)
    logger.info(f"Loading {args.dataset} test dataset for evaluation")
    
    # Few-Shot Train evaluation
    if args.eval_few_shot_train:
        logger.info("Running few-shot evaluation on training data.")
        test_dataset = utils.get_dataset(args.dataset, 
                                         "train", 
                                         CONFIG.prompt_version,
                                         few_shot=True, 
                                         k_shot=4,
                                         few_shot_template=CONFIG.few_shot_template)
        logger.info(f"Train dataset loaded with {len(test_dataset)} examples")

        # Run evaluation with the model
        logger.info("Starting training evaluation")
        lora_path = CONFIG.get_model_dir() if args.mode == "evaluate" else (lora_file_path if 'lora_file_path' in locals() else args.lora_name)
        logger.info(f"Using the lora adapter: {'Base' if args.lora_name == 'Base' else lora_path}")
        results = utils.evaluate_model(
            model, 
            test_dataset, 
            tokenizer, 
            lora_path=None if args.lora_name == 'Base' else lora_path,
        )

        # Create directory for results if it doesn't exist
        results_dir = CONFIG.get_model_dir()
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results to file in the model-specific directory
        results_filename = f"./{results_dir}/{args.dataset}_train_4-shot_{timestamp}_results.json"
        if 'Base' in results_dir:
            results_filename = f"./{results_dir}/{args.dataset}_train_4-shot-{CONFIG.few_shot_template}_{CONFIG.prompt_version}_{timestamp}_results.json"
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Training Few-Shot detailed results saved to {results_filename}")

        utils.analyze_errors(results)
        logger.info(f"Training Few-shot Accuracy: {results['accuracy']:.2f}%")
        
    # Zero-shot evaluation
    if args.eval_zero_shot:
        logger.info("Running zero-shot evaluation")
        test_dataset = utils.get_dataset(args.dataset, 
                                         "test", 
                                         CONFIG.prompt_version,
                                         few_shot=False, 
                                         few_shot_template=CONFIG.few_shot_template)
        logger.info(f"Test dataset loaded with {len(test_dataset)} examples")

        # Run evaluation with the model
        logger.info("Starting zero-shot evaluation")
        lora_path = CONFIG.get_model_dir() if args.mode == "evaluate" else (lora_file_path if 'lora_file_path' in locals() else args.lora_name)
        logger.info(f"Using the lora adapter: {'Base' if args.lora_name == 'Base' else lora_path}")
        results = utils.evaluate_model(
            model, 
            test_dataset, 
            tokenizer, 
            lora_path=None if args.lora_name == 'Base' else lora_path,
        )

        # Create directory for results if it doesn't exist
        results_dir = CONFIG.get_model_dir()
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results to file in the model-specific directory
        results_filename = f"./{results_dir}/{args.dataset}_zeroshot_{timestamp}_results.json"
        if 'Base' in results_dir:
            results_filename = f"./{results_dir}/{args.dataset}_zeroshot_{CONFIG.prompt_version}_{timestamp}_results.json"
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Zero-shot detailed results saved to {results_filename}")

        utils.analyze_errors(results)
        logger.info(f"Zero-shot Accuracy: {results['accuracy']:.2f}%")
    
    # Few-shot evaluation
    if args.eval_few_shot:
        logger.info("Running few-shot evaluation")
        test_dataset_few_shot = utils.get_dataset(args.dataset, 
                                                  "test", 
                                                  CONFIG.prompt_version,
                                                  few_shot=True, 
                                                  k_shot=args.eval_k_shot, 
                                                  few_shot_template=CONFIG.few_shot_template)
        logger.info(f"Few-shot test dataset loaded with {len(test_dataset_few_shot)} examples")

        # Run few-shot evaluation with the model
        logger.info("Starting few-shot evaluation")
        lora_path = CONFIG.get_model_dir() if args.mode == "evaluate" else (lora_file_path if 'lora_file_path' in locals() else args.lora_name)
        few_shot_results = utils.evaluate_model(
            model, 
            test_dataset_few_shot, 
            tokenizer, 
            lora_path=None if args.lora_name == 'Base' else lora_path
        )

        # Save few-shot detailed results
        few_shot_results_filename = f"./{results_dir}/{args.dataset}_{args.eval_k_shot}-shot_{timestamp}_results.json"
        if 'Base' in results_dir:
            few_shot_results_filename = f"./{results_dir}/{args.dataset}_{args.eval_k_shot}-shot_{CONFIG.prompt_version}_{timestamp}_results.json"
        with open(few_shot_results_filename, "w") as f:
            json.dump(few_shot_results, f, indent=2)
        logger.info(f"Few-shot detailed results saved to {few_shot_results_filename}")

        utils.analyze_errors(few_shot_results)
        logger.info(f"Few-shot Accuracy: {few_shot_results['accuracy']:.2f}%")
    
    logger.info(f"Experiment completed. Log saved to {log_file}")