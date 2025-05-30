"""
Dataset classes for Vietnamese Non-accented GPT training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import random
import os
import json


class VietnameseNonAccentedDataset(Dataset):
    """Dataset for training Vietnamese Non-accented GPT"""

    def __init__(
        self,
        data_dir: str = "ml/data",
        max_length: int = 32,
        context_length: int = 5,
        tokenizer=None
    ):
        self.data_dir = data_dir
        self.max_length = max_length
        self.context_length = context_length
        self.tokenizer = tokenizer

        # Load data
        self.load_training_data()

    def load_training_data(self):
        """Load training data from preprocessed files"""
        # Load training pairs
        pairs_path = os.path.join(self.data_dir, "training_pairs.csv")
        if os.path.exists(pairs_path):
            self.training_df = pd.read_csv(pairs_path)
            print(f"Loaded {len(self.training_df)} training pairs")
        else:
            raise FileNotFoundError(
                f"Training pairs not found at {pairs_path}")

        # Load word frequencies for sampling
        freq_path = os.path.join(self.data_dir, "word_freq.json")
        if os.path.exists(freq_path):
            with open(freq_path, 'r', encoding='utf-8') as f:
                self.word_freq = json.load(f)
        else:
            self.word_freq = {}

        # Create sequences for training
        self.create_training_sequences()

    def create_training_sequences(self):
        """Create training sequences with context"""
        self.sequences = []

        # Group by non_accented to create context-aware sequences
        non_accented_groups = self.training_df.groupby('non_accented')

        for non_accented, group in non_accented_groups:
            words = group['word'].tolist()

            # Create sequences with context
            for i, target_word in enumerate(words):
                # Create context from previous words
                context_words = []

                # Add some random context words
                if len(words) > 1:
                    other_words = [w for w in words if w != target_word]
                    context_size = min(self.context_length, len(other_words))
                    context_words = random.sample(other_words, context_size)

                # Add high-frequency words as context sometimes
                if random.random() < 0.3:  # 30% chance
                    high_freq_words = [
                        w for w, f in self.word_freq.items() if f > 1000]
                    if high_freq_words:
                        additional_context = random.sample(
                            high_freq_words,
                            min(2, len(high_freq_words),
                                self.context_length - len(context_words))
                        )
                        context_words.extend(additional_context)

                # Create sequence: [<sos>] + context + [target_word]
                sequence = ["<sos>"] + context_words + [target_word]

                # Pad or truncate to max_length
                if len(sequence) > self.max_length:
                    sequence = sequence[:self.max_length]

                self.sequences.append({
                    'sequence': sequence,
                    'target_word': target_word,
                    'non_accented': non_accented,
                    'context': context_words
                })

        print(f"Created {len(self.sequences)} training sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        sequence = item['sequence']

        if self.tokenizer:
            # Encode sequence
            encoded = self.tokenizer.encode_sequence(sequence, self.max_length)

            # Create input and target
            input_ids = encoded[:-1]  # All but last token
            target_ids = encoded[1:]  # All but first token

            # Pad to ensure consistent length
            if len(input_ids) < self.max_length - 1:
                pad_id = self.tokenizer.vocab.get("<pad>", 0)
                input_ids.extend(
                    [pad_id] * (self.max_length - 1 - len(input_ids)))
                target_ids.extend(
                    [pad_id] * (self.max_length - 1 - len(target_ids)))

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'non_accented': item['non_accented'],
                'target_word': item['target_word']
            }
        else:
            return item


class VietnameseContextDataset(Dataset):
    """Dataset for context-aware training"""

    def __init__(
        self,
        corpus_path: str = "data/corpus-full.txt",
        tokenizer=None,
        max_length: int = 32,
        num_samples: int = 100000
    ):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples

        self.sequences = []
        self.load_corpus_sequences()

    def load_corpus_sequences(self):
        """Load sequences from corpus for context training"""
        print(f"Loading corpus sequences from {self.corpus_path}")

        try:
            with open(self.corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= self.num_samples:
                        break

                    if i % 10000 == 0:
                        print(f"Processed {i} lines")

                    # Clean and tokenize line
                    line = line.strip().lower()
                    if len(line) < 20:  # Skip short lines
                        continue

                    # Simple word tokenization
                    words = line.split()
                    words = [w for w in words if len(w) >= 2 and w.isalpha()]

                    if len(words) < 3:  # Need at least 3 words
                        continue

                    # Create sliding window sequences
                    for j in range(len(words) - 2):
                        sequence = words[j:j +
                                         min(self.max_length, len(words) - j)]
                        if len(sequence) >= 3:
                            self.sequences.append(sequence)

        except Exception as e:
            print(f"Error loading corpus: {e}")

        print(f"Loaded {len(self.sequences)} corpus sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        if self.tokenizer:
            # Add special tokens
            sequence_with_tokens = ["<sos>"] + sequence + ["<eos>"]

            # Encode
            encoded = self.tokenizer.encode_sequence(
                sequence_with_tokens, self.max_length)

            # Create input and target
            input_ids = encoded[:-1]
            target_ids = encoded[1:]

            # Pad if necessary
            if len(input_ids) < self.max_length - 1:
                pad_id = self.tokenizer.vocab.get("<pad>", 0)
                input_ids.extend(
                    [pad_id] * (self.max_length - 1 - len(input_ids)))
                target_ids.extend(
                    [pad_id] * (self.max_length - 1 - len(target_ids)))

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'non_accented': None,  # Context dataset doesn't have this field
                'target_word': None   # Context dataset doesn't have this field
            }
        else:
            return {'sequence': sequence}


def custom_collate_fn(batch):
    """Custom collate function to handle None values"""
    # Separate fields
    input_ids = []
    target_ids = []
    non_accenteds = []
    target_words = []

    for item in batch:
        input_ids.append(item['input_ids'])
        target_ids.append(item['target_ids'])

        # Handle None values by converting to empty string
        non_accenteds.append(item['non_accented']
                             if item['non_accented'] is not None else "")
        target_words.append(item['target_word']
                            if item['target_word'] is not None else "")

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
        'non_accented': non_accenteds,
        'target_word': target_words
    }


def create_data_loaders(
    data_dir: str = "ml/data",
    corpus_path: str = "data/corpus-full.txt",
    tokenizer=None,
    batch_size: int = 32,
    max_length: int = 32,
    train_split: float = 0.8,
    num_corpus_samples: int = 50000
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""

    # Create pinyin dataset
    pinyin_dataset = VietnameseNonAccentedDataset(
        data_dir=data_dir,
        max_length=max_length,
        tokenizer=tokenizer
    )

    # Create context dataset
    context_dataset = VietnameseContextDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_corpus_samples
    )

    # Combine datasets
    combined_data = []

    # Add pinyin data (higher weight)
    for i in range(len(pinyin_dataset)):
        combined_data.append(('pinyin', i))

    # Add context data
    for i in range(len(context_dataset)):
        combined_data.append(('context', i))

    # Shuffle
    random.shuffle(combined_data)

    # Split train/val
    split_idx = int(len(combined_data) * train_split)
    train_indices = combined_data[:split_idx]
    val_indices = combined_data[split_idx:]

    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # Create combined dataset
    class CombinedDataset(Dataset):
        def __init__(self, indices, pinyin_dataset, context_dataset):
            self.indices = indices
            self.pinyin_dataset = pinyin_dataset
            self.context_dataset = context_dataset

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            dataset_type, dataset_idx = self.indices[idx]
            if dataset_type == 'pinyin':
                return self.pinyin_dataset[dataset_idx]
            else:
                return self.context_dataset[dataset_idx]

    train_dataset = CombinedDataset(
        train_indices, pinyin_dataset, context_dataset)
    val_dataset = CombinedDataset(val_indices, pinyin_dataset, context_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader
