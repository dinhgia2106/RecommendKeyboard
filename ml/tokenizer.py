"""
Vietnamese Non-accented Tokenizer
Handles conversion between non-accented input and Vietnamese words
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pickle


class VietnameseNonAccentedTokenizer:
    def __init__(self, data_dir: str = "ml/data"):
        self.data_dir = data_dir
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_to_non_accented = {}
        self.non_accented_to_words = defaultdict(list)
        self.word_freq = {}

        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"

        # Padding token index constant (from v7)
        self.PADDING_TOKEN_INDEX = 0

        self.special_tokens = [self.PAD_TOKEN,
                               self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

        # Load data if available
        self.load_data()

    def load_data(self):
        """Load preprocessed data"""
        try:
            # Load vocabulary
            vocab_path = os.path.join(self.data_dir, "vocab.json")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.vocab = json.load(f)
                self.reverse_vocab = {
                    idx: word for word, idx in self.vocab.items()}
                print(f"Loaded vocabulary: {len(self.vocab)} words")

            # Load word-non_accented mappings
            word_non_accented_path = os.path.join(
                self.data_dir, "word_to_non_accented.json")
            if os.path.exists(word_non_accented_path):
                with open(word_non_accented_path, 'r', encoding='utf-8') as f:
                    self.word_to_non_accented = json.load(f)
                print(
                    f"Loaded word-non_accented mappings: {len(self.word_to_non_accented)} mappings")

            # Load non_accented-words mappings
            non_accented_words_path = os.path.join(
                self.data_dir, "non_accented_to_words.json")
            if os.path.exists(non_accented_words_path):
                with open(non_accented_words_path, 'r', encoding='utf-8') as f:
                    non_accented_data = json.load(f)
                    self.non_accented_to_words = defaultdict(
                        list, non_accented_data)
                print(
                    f"Loaded non_accented-words mappings: {len(self.non_accented_to_words)} non_accenteds")

            # Load word frequencies
            freq_path = os.path.join(self.data_dir, "word_freq.json")
            if os.path.exists(freq_path):
                with open(freq_path, 'r', encoding='utf-8') as f:
                    self.word_freq = json.load(f)
                print(f"Loaded word frequencies: {len(self.word_freq)} words")

        except Exception as e:
            print(f"Error loading tokenizer data: {e}")

    def encode_word(self, word: str) -> int:
        """Encode single word to token ID"""
        return self.vocab.get(word, self.vocab.get(self.UNK_TOKEN, 1))

    def decode_token(self, token_id: int) -> str:
        """Decode token ID to word"""
        return self.reverse_vocab.get(token_id, self.UNK_TOKEN)

    def encode_sequence(self, words: List[str], max_length: Optional[int] = None) -> List[int]:
        """Encode sequence of words to token IDs"""
        encoded = [self.encode_word(word) for word in words]

        if max_length:
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                # Pad with PAD_TOKEN
                pad_id = self.vocab.get(self.PAD_TOKEN, 0)
                encoded.extend([pad_id] * (max_length - len(encoded)))

        return encoded

    def decode_sequence(self, token_ids: List[int], remove_special: bool = True) -> List[str]:
        """Decode sequence of token IDs to words"""
        words = [self.decode_token(token_id) for token_id in token_ids]

        if remove_special:
            words = [word for word in words if word not in self.special_tokens]

        return words

    def non_accented_to_candidates(self, non_accented: str, max_candidates: int = 10) -> List[Tuple[str, float]]:
        """Get word candidates for non_accented input, sorted by frequency"""
        candidates = []

        if non_accented in self.non_accented_to_words:
            words = self.non_accented_to_words[non_accented]

            # Sort by frequency
            word_freq_pairs = []
            for word in words:
                freq = self.word_freq.get(word, 0)
                word_freq_pairs.append((word, freq))

            # Sort by frequency (descending)
            word_freq_pairs.sort(key=lambda x: x[1], reverse=True)

            # Convert to candidates with confidence scores
            total_freq = sum(
                freq for _, freq in word_freq_pairs[:max_candidates])

            for word, freq in word_freq_pairs[:max_candidates]:
                confidence = freq / total_freq if total_freq > 0 else 0.1
                candidates.append((word, confidence))

        return candidates

    def word_to_non_accented_map(self, word: str) -> Optional[str]:
        """Get non_accented representation of a word"""
        return self.word_to_non_accented.get(word)

    def create_context_sequence(self, context_words: List[str], target_non_accented: str, max_length: int = 32) -> Tuple[List[int], List[str]]:
        """Create input sequence for model training/inference"""
        # Start with SOS token
        sequence = [self.SOS_TOKEN]

        # Add context words
        # Leave space for target
        sequence.extend(context_words[-max_length+2:])

        # Encode sequence
        encoded = self.encode_sequence(sequence, max_length)

        # Get target candidates
        candidates = self.non_accented_to_candidates(target_non_accented)
        target_words = [word for word, _ in candidates]

        return encoded, target_words

    def batch_encode(self, batch_data: List[List[str]], max_length: int = 32) -> List[List[int]]:
        """Batch encode sequences"""
        return [self.encode_sequence(seq, max_length) for seq in batch_data]

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def get_statistics(self) -> Dict:
        """Get tokenizer statistics"""
        return {
            'vocab_size': len(self.vocab),
            'word_non_accented_mappings': len(self.word_to_non_accented),
            'non_accented_word_mappings': len(self.non_accented_to_words),
            'word_frequencies': len(self.word_freq),
            'special_tokens': len(self.special_tokens)
        }

    def save_tokenizer(self, save_path: str):
        """Save tokenizer state"""
        tokenizer_data = {
            'vocab': self.vocab,
            'word_to_non_accented': self.word_to_non_accented,
            'non_accented_to_words': dict(self.non_accented_to_words),
            'word_freq': self.word_freq,
            'special_tokens': self.special_tokens
        }

        with open(save_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)

        print(f"Tokenizer saved to {save_path}")

    def load_tokenizer(self, load_path: str):
        """Load tokenizer state"""
        with open(load_path, 'rb') as f:
            tokenizer_data = pickle.load(f)

        self.vocab = tokenizer_data['vocab']
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.word_to_non_accented = tokenizer_data['word_to_non_accented']
        self.non_accented_to_words = defaultdict(
            list, tokenizer_data['non_accented_to_words'])
        self.word_freq = tokenizer_data['word_freq']
        self.special_tokens = tokenizer_data['special_tokens']

        print(f"Tokenizer loaded from {load_path}")


# Global tokenizer instance
_tokenizer = None


def get_tokenizer(data_dir: str = "ml/data") -> VietnameseNonAccentedTokenizer:
    """Get global tokenizer instance"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = VietnameseNonAccentedTokenizer(data_dir)
    return _tokenizer
