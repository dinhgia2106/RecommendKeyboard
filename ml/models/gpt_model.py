"""
Vietnamese Non-accented GPT Model
Lightweight GPT architecture for Vietnamese word prediction
Based on v7 architecture but optimized for non-accented input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GPTConfig:
    """GPT model configuration"""

    def __init__(
        self,
        vocab_size: int = 50000,
        block_size: int = 32,
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.1,
        bias: bool = True
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VietnameseNonAccentedGPT(nn.Module):
    """Vietnamese Non-accented GPT Model"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing between token embedding and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params():,}")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # Shape (t)

        # Token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # Position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            # Note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Crop the block size if necessary"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,
                                                  :, :block_size, :block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens given a conditioning sequence"""
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def predict_next_words(self, context_ids: torch.Tensor, num_predictions: int = 10, temperature: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k next words given context"""
        self.eval()

        # Forward pass
        logits, _ = self(context_ids)

        # Get logits for the last position
        last_logits = logits[:, -1, :] / temperature

        # Get top-k predictions
        top_logits, top_indices = torch.topk(
            last_logits, num_predictions, dim=-1)

        # Convert to probabilities
        probs = F.softmax(top_logits, dim=-1)

        return top_indices, probs


def create_model(config: GPTConfig) -> VietnameseNonAccentedGPT:
    """Create a new model instance"""
    return VietnameseNonAccentedGPT(config)


def load_model(checkpoint_path: str, config: Optional[GPTConfig] = None) -> VietnameseNonAccentedGPT:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if config is None:
        config = checkpoint.get('config', None)
        if config is None:
            # Default config if not saved
            config = GPTConfig()

    model = create_model(config)

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Handle potential key mismatches
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: {e}")
        # Try to load with strict=False
        model.load_state_dict(state_dict, strict=False)

    return model


def save_model(model: VietnameseNonAccentedGPT, save_path: str, config: GPTConfig, optimizer=None, epoch=None, loss=None):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'num_parameters': model.get_num_params(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if loss is not None:
        checkpoint['loss'] = loss

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
