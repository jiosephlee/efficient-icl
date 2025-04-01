import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import torch
import bitsandbytes as bnb

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = ''
    lora_name: str = "Base"  # Default to Base model
    lora_rank: int = 32
    max_seq_length: int = 2048
    target_modules: Optional[List[str]] = None
    auto_find_modules: bool = False  # New field to control automatic module finding
    lora_alpha: int = 32  # Same as lora_rank by default
    lora_dropout = 0 # Optimized value for unsloth fast training
    use_gradient_checkpointing: str = "unsloth"  # Enable long context finetuning
    random_state: int = 3407
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.6
    
    # Dataset configuration
    train_dataset: str = "gsm8k"
    few_shot: bool = False
    k_shot: int = 4
    few_shot_template: str = "chat"  # Options: 'chat', 'combined'
    
    # Prompt configuration
    prompt_version: str = "v0"  # Options: 'v0', 'v1'
    
    # Training parameters
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2 # Increase to 4 for smoother training
    num_generations: int = 8 # Decrease if out of memory
    max_prompt_length: int = 1096
    max_steps: int = 250  # Set to 1000 for better results
    num_train_epochs: Optional[int] = None # Set to 1 for a full training run
    save_steps: int = 250
    # scale_rewards: bool = True, # Dr. GRPO recommands False
    max_grad_norm: float = 0.1
    
    # Reward functions to use (in order of application)
    reward_functions: List[str] = field(default_factory=lambda: [
        "xmlcount_reward_func",
        "soft_format_reward_func", 
        "strict_format_reward_func",
        "int_reward_func",
        "correctness_reward_func"
    ])

    # Inference parameters
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    max_tokens: int = 1024
    
    # Paths and directories
    output_base_dir: str = "./models"
    checkpoint_base_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    
    def get_model_dir(self) -> str:
        """Get the model-specific directory path."""
        return os.path.join(self.output_base_dir, f"{self.model_name.split('/')[-1]}/{self.lora_name}/")
    
    def get_checkpoint_dir(self) -> str:
        """Get the checkpoint directory path."""
        return os.path.join(self.checkpoint_base_dir, f"{self.model_name.split('/')[-1]}/{self.lora_name}/")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create a config from a dictionary."""
        return cls(**config_dict)
    
    @staticmethod
    def find_all_linear_names(model, load_in_4bit: bool = True) -> List[str]:
        """Find all linear module names in the model."""
        cls = bnb.nn.Linear4bit if load_in_4bit else torch.nn.Linear
        lora_module_names: Set[str] = set()
        
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
        # Remove lm_head if present
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
            
        return list(lora_module_names)
    
    def get_target_modules(self, model) -> List[str]:
        """Get target modules, either from config or by finding them automatically."""
        if self.auto_find_modules:
            return self.find_all_linear_names(model, self.load_in_4bit)
        return self.target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]


# Default configurations for different experiments
DEFAULT_CONFIGS = {
    "Base": TrainingConfig(
        lora_name="Base",
        prompt_version="v0",
        few_shot=False
    ),
    "Base_v1": TrainingConfig(
        lora_name="Base",
        prompt_version="v1",
        few_shot=False,
    ),
    "Base_v2": TrainingConfig(
        lora_name="Base",
        prompt_version="v2",
        few_shot=False,
    ),
    "Base_v3": TrainingConfig(
        lora_name="Base",
        prompt_version="v3",
        few_shot=False,
    ),
    "v0": TrainingConfig(
        lora_name="v0",
        prompt_version="v0",
        few_shot=False,
        max_steps=250
    ),
    "v0_few_shot_chat": TrainingConfig(
        lora_name="v0_few_shot_chat",
        prompt_version="v0",
        few_shot=True,
        k_shot=4,
        few_shot_template="chat",
        max_steps=250
    ),
    "v0_few_shot_combined": TrainingConfig(
        lora_name="v0_few_shot_combined",
        prompt_version="v0",
        few_shot=True,
        k_shot=4,
        few_shot_template="combined",
        max_steps=250
    ),
    "v1_1_few_shot_chat": TrainingConfig(
        lora_name="v1_1_few_shot_chat",
        train_dataset="gsm8k_hard",
        prompt_version="v1",
        few_shot=True,
        max_steps=300,
        num_generations=12,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4
    ),
    "v1_2_few_shot_chat": TrainingConfig(
        lora_name="v1_2_few_shot_chat",
        train_dataset="gsm8k_hard",
        prompt_version="v2",
        few_shot=True,
        max_steps=300,
        num_generations=12,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4
    ),
    "v1_3_few_shot_chat": TrainingConfig(
        lora_name="v1_3_few_shot_chat",
        train_dataset="gsm8k_hard",
        prompt_version="v3",
        few_shot=True,
        max_steps=300,
        num_generations=12,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        reward_functions=field(default_factory=lambda: [
            "int_reward_func",
            "correctness_reward_func"
        ])
    ),
}


def get_config(config_name: str = "Base") -> TrainingConfig:
    """Get a configuration by name, or return the default if not found."""
    return DEFAULT_CONFIGS.get(config_name, DEFAULT_CONFIGS["Base"])
