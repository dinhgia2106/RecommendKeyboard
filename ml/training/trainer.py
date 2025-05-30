"""
Trainer for Vietnamese Non-accented GPT Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from ..models.gpt_model import VietnameseNonAccentedGPT, GPTConfig, save_model, load_model
from ..tokenizer import VietnameseNonAccentedTokenizer


class VietnameseNonAccentedTrainer:
    """Trainer for Vietnamese Non-accented GPT"""

    def __init__(
        self,
        model: VietnameseNonAccentedGPT,
        tokenizer: VietnameseNonAccentedTokenizer,
        config: GPTConfig,
        device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Set device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)

    def setup_optimizer(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95)
    ):
        """Setup AdamW optimizer"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        self.optimizer = optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas)
        print(
            f"Setup optimizer with {len(decay_params)} decay params and {len(no_decay_params)} no-decay params")

    def setup_scheduler(
        self,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        min_lr_ratio: float = 0.1
    ):
        """Setup cosine learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                progress = min(progress, 1.0)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        print(
            f"Setup scheduler with {warmup_steps} warmup steps and {max_steps} max steps")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)

        # Forward pass
        logits, loss = self.model(input_ids, target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            logits, loss = self.model(input_ids, target_ids)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            epoch_losses.append(loss)

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'avg_loss': f"{np.mean(epoch_losses):.4f}",
                'lr': f"{current_lr:.6f}"
            })

            # Log metrics every 100 steps
            if self.global_step % 100 == 0:
                self.train_losses.append(loss)
                self.learning_rates.append(current_lr)

        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_losses)
        train_perplexity = np.exp(avg_train_loss)

        metrics = {
            'train_loss': avg_train_loss,
            'train_perplexity': train_perplexity
        }

        # Validation
        if val_loader is not None:
            val_metrics = self.validate(val_loader)
            metrics.update(val_metrics)
            self.val_losses.append(val_metrics['val_loss'])

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        num_epochs: int = 10,
        save_every: int = 2,
        eval_every: int = 1
    ):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            if val_loader and epoch % eval_every == 0:
                metrics = self.train_epoch(train_loader, val_loader)
            else:
                metrics = self.train_epoch(train_loader)

            # Print metrics
            print(f"\nEpoch {epoch} Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(
                    f"checkpoints/vietnamese_non_accented_gpt_epoch_{epoch}.pth", metrics)

            # Save best model
            if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = metrics['val_loss']
                self.save_checkpoint(
                    "checkpoints/vietnamese_non_accented_gpt_best.pth", metrics)
                print(f"  New best validation loss: {self.best_val_loss:.4f}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")

        # Plot training curves
        self.plot_training_curves()

    def save_checkpoint(self, path: str, metrics: Dict[str, float] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }

        if hasattr(self, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

        print(f"Checkpoint loaded from {path}")
        print(
            f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def plot_training_curves(self):
        """Plot training curves"""
        if not self.train_losses:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Training loss
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

        # Validation loss
        if self.val_losses:
            axes[0, 1].plot(self.val_losses)
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)

        # Learning rate
        if self.learning_rates:
            axes[1, 0].plot(self.learning_rates)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True)

        # Combined losses
        if self.val_losses:
            train_x = np.linspace(0, len(self.val_losses),
                                  len(self.train_losses))
            val_x = range(len(self.val_losses))

            axes[1, 1].plot(train_x, self.train_losses,
                            label='Train', alpha=0.7)
            axes[1, 1].plot(val_x, self.val_losses,
                            label='Validation', marker='o')
            axes[1, 1].set_title('Training vs Validation Loss')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('checkpoints/training_curves.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        print("Training curves saved to checkpoints/training_curves.png")

    @torch.no_grad()
    def test_predictions(self, test_cases: List[Tuple[str, str]], num_predictions: int = 5):
        """Test model predictions on sample cases"""
        self.model.eval()

        print("\n" + "="*50)
        print("TESTING MODEL PREDICTIONS")
        print("="*50)

        for non_accented, expected_word in test_cases:
            print(
                f"\nNon-accented: '{non_accented}' -> Expected: '{expected_word}'")

            # Get candidates from tokenizer
            candidates = self.tokenizer.non_accented_to_candidates(
                non_accented, num_predictions)

            if candidates:
                print("Tokenizer candidates:")
                for i, (word, conf) in enumerate(candidates):
                    marker = "✓" if word == expected_word else " "
                    print(f"  {marker} {i+1}. {word} ({conf:.3f})")
            else:
                print("  No candidates found in tokenizer")

            # Test with model prediction (if we have context)
            context = ["tôi", "muốn"]  # Simple context
            try:
                context_ids = self.tokenizer.encode_sequence(
                    context, max_length=10)
                context_tensor = torch.tensor(
                    [context_ids], device=self.device)

                top_indices, probs = self.model.predict_next_words(
                    context_tensor,
                    num_predictions=num_predictions,
                    temperature=0.8
                )

                print("Model predictions (with context):")
                for i, (idx, prob) in enumerate(zip(top_indices[0], probs[0])):
                    word = self.tokenizer.decode_token(idx.item())
                    marker = "✓" if word == expected_word else " "
                    print(f"  {marker} {i+1}. {word} ({prob:.3f})")

            except Exception as e:
                print(f"  Model prediction error: {e}")

        print("="*50)


def create_trainer(
    vocab_size: int,
    data_dir: str = "ml/data",
    model_config: Dict = None,
    device: str = "auto"
) -> VietnameseNonAccentedTrainer:
    """Create a new trainer instance"""

    # Default model config
    if model_config is None:
        model_config = {
            'vocab_size': vocab_size,
            'block_size': 32,
            'n_layer': 8,
            'n_head': 8,
            'n_embd': 256,
            'dropout': 0.1
        }

    # Create config
    config = GPTConfig(**model_config)

    # Create model
    from ..models.gpt_model import create_model
    model = create_model(config)

    # Create tokenizer
    tokenizer = VietnameseNonAccentedTokenizer(data_dir)

    # Create trainer
    trainer = VietnameseNonAccentedTrainer(model, tokenizer, config, device)

    return trainer
