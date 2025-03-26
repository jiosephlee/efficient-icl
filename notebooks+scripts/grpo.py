from unsloth import FastLanguageModel
from vllm import SamplingParams
import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from tqdm import tqdm
import json
import sys
sys.path.append("../")
import utils

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

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

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': utils.SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': utils.extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

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
    output_dir = "unsloth_recipe",
)

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
trainer.train()

text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

model.save_lora("grpo_saved_lora")

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
print(output)

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)

test_dataset = get_gsm8k_questions(split = "test")

# Run evaluation with your fine-tuned model
results = utils.evaluate_model_on_gsm8k(model, test_dataset, tokenizer, lora_path="grpo_saved_lora")

# Save detailed results to file

with open("gsm8k_evaluation_results.json", "w") as f:
    json.dump(results['detailed_results'], f, indent=2)

utils.analyze_errors(results)


###
# --- Previous code to load model, tokenizer, dataset ---

# --- UPDATE max_steps in training_args ---
training_args = GRPOConfig(
    # ... (keep previous settings) ...
    max_steps = 1000, # Or however many total steps you want
    # save_steps = 100, # Maybe save more frequently now
    # ... (other args) ...
    output_dir = "outputs", # Keep the same output directory
)

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

# --- Resume training ---
# Option 1: Automatically find the latest checkpoint in output_dir
trainer.train(resume_from_checkpoint = True)

# Option 2: Explicitly specify the checkpoint path
# checkpoint_path = "outputs/checkpoint-250" # Path to the specific checkpoint
# trainer.train(resume_from_checkpoint = checkpoint_path)

# --- Save final LoRA adapters after continued training ---
model.save_lora("grpo_saved_lora_continued") # Save with a new name or overwrite

# --- Continue with evaluation etc. ---