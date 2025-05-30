"""
Configuration file for Vietnamese Non-accented GPT Model
Inspired by v7 architecture with improvements
"""

import torch
import os
from dataclasses import dataclass
from typing import Dict, Any


# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model constants from v7 (improved)
MAX_SEQUENCE_LEN = 32
VOCAB_SIZE = 50000  # Will be updated based on actual vocabulary

# Model configurations inspired by v7
MODEL_SIZES = {
    'tiny': {
        'block_size': MAX_SEQUENCE_LEN,
        'vocab_size': VOCAB_SIZE,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.1,
        'bias': True
    },
    'small': {
        'block_size': MAX_SEQUENCE_LEN,
        'vocab_size': VOCAB_SIZE,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 192,
        'dropout': 0.1,
        'bias': True
    },
    'base': {
        'block_size': MAX_SEQUENCE_LEN,
        'vocab_size': VOCAB_SIZE,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 256,
        'dropout': 0.1,
        'bias': True
    },
    'large': {
        'block_size': MAX_SEQUENCE_LEN,
        'vocab_size': VOCAB_SIZE,
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 384,
        'dropout': 0.1,
        'bias': True
    }
}

# Training configurations
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 3e-4,
    'max_epochs': 100,
    'warmup_steps': 1000,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'eval_interval': 500,
    'save_interval': 1000,
    'early_stopping_patience': 10
}

# Data paths
DATA_PATHS = {
    'vocab_file': 'ml/data/vocab.json',
    'word_to_non_accented': 'ml/data/word_to_non_accented.json',
    'non_accented_to_words': 'ml/data/non_accented_to_words.json',
    'word_freq': 'ml/data/word_freq.json',
    'training_data': 'ml/data/training_pairs.csv',
    'checkpoints_dir': 'checkpoints/'
}

# Inference configurations
INFERENCE_CONFIG = {
    'max_suggestions': 8,
    'temperature': 0.8,
    'top_k': 50,
    'cache_size': 1000,
    'context_window': 5  # Number of previous words to consider
}


@dataclass
class GPTConfig:
    """Enhanced GPT Configuration inspired by v7"""
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self):
        # Validate configuration
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size > 0, "block_size must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"


def get_model_config(size: str = 'base', vocab_size: int = None) -> GPTConfig:
    """Get model configuration for specified size"""
    if size not in MODEL_SIZES:
        raise ValueError(
            f"Model size {size} not supported. Available: {list(MODEL_SIZES.keys())}")

    config_dict = MODEL_SIZES[size].copy()

    # Update vocab size if provided
    if vocab_size is not None:
        config_dict['vocab_size'] = vocab_size

    return GPTConfig(**config_dict)


def get_checkpoint_path(model_size: str = 'base', epoch: int = None) -> str:
    """Get checkpoint path for model"""
    base_name = f"vietnamese_gpt_{model_size}"
    if epoch is not None:
        return os.path.join(DATA_PATHS['checkpoints_dir'], f"{base_name}_epoch_{epoch}.pth")
    else:
        return os.path.join(DATA_PATHS['checkpoints_dir'], f"{base_name}_best.pth")


def update_vocab_size(new_vocab_size: int):
    """Update vocabulary size in all model configurations"""
    global VOCAB_SIZE
    VOCAB_SIZE = new_vocab_size

    for size_config in MODEL_SIZES.values():
        size_config['vocab_size'] = new_vocab_size


def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    info = {
        'device': DEVICE,
        'device_name': DEVICE,
        'cuda_available': torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        info.update({
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9
        })

    return info
