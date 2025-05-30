"""
Enhanced Vietnamese Non-accented Data Preprocessor
Integrates Viet74K.txt dictionary with corpus for better coverage
"""

import os
import json
import csv
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import unicodedata
import random
import pandas as pd


class VietnameseNonAccentedPreprocessor:
    def __init__(self, corpus_path: str = "data/corpus-full.txt", viet74k_path: str = "data/Viet74K.txt"):
        self.corpus_path = corpus_path
        self.viet74k_path = viet74k_path

        # Data storage
        self.vocab = set()
        self.word_freq = defaultdict(int)
        self.word_to_non_accented = {}
        self.non_accented_to_words = defaultdict(list)
        self.training_pairs = []

        # Statistics
        self.stats = {
            'total_lines_processed': 0,
            'total_words_found': 0,
            'vocab_size': 0,
            'viet74k_words_integrated': 0,
            'corpus_words_processed': 0,
            'training_pairs_created': 0
        }

        print(f"🚀 Enhanced Vietnamese Non-accented Preprocessor")
        print(f"📖 Corpus: {corpus_path}")
        print(f"📚 Viet74K Dictionary: {viet74k_path}")

    def remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents from text"""
        # Vietnamese accent mapping
        accent_map = {
            'àáạảãâầấậẩẫăằắặẳẵ': 'a',
            'èéẹẻẽêềếệểễ': 'e',
            'ìíịỉĩ': 'i',
            'òóọỏõôồốộổỗơờớợởỡ': 'o',
            'ùúụủũưừứựửữ': 'u',
            'ỳýỵỷỹ': 'y',
            'đ': 'd',
            'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ': 'A',
            'ÈÉẸẺẼÊỀẾỆỂỄ': 'E',
            'ÌÍỊỈĨ': 'I',
            'ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ': 'O',
            'ÙÚỤỦŨƯỪỨỰỬỮ': 'U',
            'ỲÝỴỶỸ': 'Y',
            'Đ': 'D'
        }

        result = text
        for accented_chars, non_accented in accent_map.items():
            for char in accented_chars:
                result = result.replace(char, non_accented)

        return result

    def clean_text(self, text: str) -> str:
        """Clean and standardize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep Vietnamese chars
        text = re.sub(
            r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', ' ', text)

        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text.strip())

        return text.lower()

    def load_viet74k_dictionary(self):
        """Load and process Viet74K.txt dictionary"""
        print("📚 Loading Viet74K dictionary...")

        viet74k_words = set()

        try:
            with open(self.viet74k_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0:
                        print(f"  Processed {line_num} dictionary entries...")

                    word = line.strip()
                    # Keep single words and short phrases
                    if word and len(word.split()) <= 3:
                        cleaned_word = self.clean_text(word)
                        if cleaned_word and len(cleaned_word) >= 2:
                            viet74k_words.add(cleaned_word)

                            # Create non-accented mapping WITH spaces (for multi-word)
                            non_accented_spaced = self.remove_accents(
                                cleaned_word)
                            self.word_to_non_accented[cleaned_word] = non_accented_spaced
                            self.non_accented_to_words[non_accented_spaced].append(
                                cleaned_word)

                            # Debug multi-word phrases
                            if ' ' in cleaned_word:
                                print(
                                    f"🔄 Multi-word: '{cleaned_word}' → '{non_accented_spaced}'")

                            # Add to vocabulary
                            self.vocab.add(cleaned_word)
                            # Give dictionary words higher base frequency
                            self.word_freq[cleaned_word] += 5

        except FileNotFoundError:
            print(f"⚠️ Viet74K file not found at {self.viet74k_path}")
            return set()
        except Exception as e:
            print(f"❌ Error loading Viet74K: {e}")
            return set()

        self.stats['viet74k_words_integrated'] = len(viet74k_words)
        print(f"✅ Loaded {len(viet74k_words)} words from Viet74K dictionary")

        return viet74k_words

    def process_corpus(self, sample_size: int = 100000):
        """Process corpus file with improved sampling"""
        print(f"📖 Processing corpus with {sample_size} samples...")

        try:
            total_lines = 0
            processed_words = set()

            with open(self.corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if line_num >= sample_size:
                        break

                    if line_num % 5000 == 0:
                        print(
                            f"  Processed {line_num} lines, found {len(processed_words)} unique words")

                    cleaned_line = self.clean_text(line)
                    if len(cleaned_line) < 10:  # Skip very short lines
                        continue

                    words = cleaned_line.split()
                    for word in words:
                        if len(word) >= 2 and word.isalpha():
                            processed_words.add(word)

                            # Create non-accented mapping (single words từ corpus)
                            non_accented = self.remove_accents(word)
                            self.word_to_non_accented[word] = non_accented
                            self.non_accented_to_words[non_accented].append(
                                word)

                            # Add to vocabulary and frequency
                            self.vocab.add(word)
                            self.word_freq[word] += 1

                    total_lines += 1

            self.stats['total_lines_processed'] = total_lines
            self.stats['corpus_words_processed'] = len(processed_words)
            print(
                f"✅ Processed {total_lines} lines, extracted {len(processed_words)} unique words")

        except FileNotFoundError:
            print(f"⚠️ Corpus file not found at {self.corpus_path}")
        except Exception as e:
            print(f"❌ Error processing corpus: {e}")

    def build_vocabulary(self, min_freq: int = 3):
        """Build final vocabulary with frequency filtering"""
        print(f"🔨 Building vocabulary (min_freq={min_freq})...")

        # First load Viet74K dictionary
        viet74k_words = self.load_viet74k_dictionary()

        # Then process corpus
        self.process_corpus()

        # Filter by frequency but keep all Viet74K words
        filtered_vocab = set()
        filtered_word_freq = {}

        for word, freq in self.word_freq.items():
            if freq >= min_freq or word in viet74k_words:
                filtered_vocab.add(word)
                filtered_word_freq[word] = freq

        self.vocab = filtered_vocab
        self.word_freq = filtered_word_freq

        # Clean up mappings
        filtered_word_to_non_accented = {}
        filtered_non_accented_to_words = defaultdict(list)

        for word in self.vocab:
            if word in self.word_to_non_accented:
                non_accented = self.word_to_non_accented[word]
                filtered_word_to_non_accented[word] = non_accented
                if word not in filtered_non_accented_to_words[non_accented]:
                    filtered_non_accented_to_words[non_accented].append(word)

        self.word_to_non_accented = filtered_word_to_non_accented
        self.non_accented_to_words = filtered_non_accented_to_words

        self.stats['vocab_size'] = len(self.vocab)
        self.stats['total_words_found'] = len(self.word_freq)

        print(f"✅ Built vocabulary: {len(self.vocab)} words")
        print(
            f"📊 Coverage: Viet74K={self.stats['viet74k_words_integrated']}, Corpus={self.stats['corpus_words_processed']}")

    def create_training_pairs(self):
        """Create enhanced training pairs for model training"""
        print("🎯 Creating training pairs...")

        training_pairs = []

        # Create pairs from non-accented mappings
        for non_accented, words in self.non_accented_to_words.items():
            if len(words) > 0:
                # Sort words by frequency
                sorted_words = sorted(
                    words, key=lambda w: self.word_freq.get(w, 0), reverse=True)

                for word in sorted_words:
                    freq = self.word_freq.get(word, 1)

                    # Create multiple training examples for high-frequency words
                    num_examples = min(5, max(1, freq // 10))

                    for _ in range(num_examples):
                        training_pairs.append({
                            'non_accented': non_accented,
                            'word': word,
                            'frequency': freq,
                            'word_length': len(word),
                            'num_syllables': len(word.split())
                        })

        # Add some challenging examples (words with multiple possible accented forms)
        multi_accent_pairs = []
        for non_accented, words in self.non_accented_to_words.items():
            if len(words) > 1:  # Multiple possible accented forms
                for word in words:
                    multi_accent_pairs.append({
                        'non_accented': non_accented,
                        'word': word,
                        'frequency': self.word_freq.get(word, 1),
                        'word_length': len(word),
                        'num_syllables': len(word.split()),
                        'is_challenging': True
                    })

        # Sample challenging examples
        if len(multi_accent_pairs) > 10000:
            multi_accent_pairs = random.sample(multi_accent_pairs, 10000)

        training_pairs.extend(multi_accent_pairs)

        # Shuffle training pairs
        random.shuffle(training_pairs)

        self.training_pairs = training_pairs
        self.stats['training_pairs_created'] = len(training_pairs)

        print(f"✅ Created {len(training_pairs)} training pairs")
        print(f"📈 Including {len(multi_accent_pairs)} challenging examples")

    def create_keyboard_mappings(self):
        """Create keyboard-specific mappings (no spaces in keys)"""
        print("⌨️ Creating keyboard-specific mappings...")

        keyboard_mappings = defaultdict(list)

        # Process all existing mappings
        for non_accented_key, words in self.non_accented_to_words.items():
            # Keep existing spaced mappings
            keyboard_mappings[non_accented_key].extend(words)

            # Create no-space version for keyboard input
            if ' ' in non_accented_key:
                nospace_key = non_accented_key.replace(' ', '')
                keyboard_mappings[nospace_key].extend(words)
                # Show first 3
                print(f"🔄 Keyboard mapping: '{nospace_key}' → {words[:3]}...")

        # Update the main mapping
        self.non_accented_to_words = keyboard_mappings

        print(
            f"✅ Created keyboard mappings with {len(keyboard_mappings)} keys")

        # Show some examples
        examples = list(keyboard_mappings.items())[:5]
        print("📝 Examples:")
        for key, values in examples:
            print(f"   '{key}' → {values[:2]}")

    def save_processed_data(self, output_dir: str):
        """Save all processed data"""
        print(f"💾 Saving processed data to {output_dir}...")

        os.makedirs(output_dir, exist_ok=True)

        # Save vocabulary
        vocab_dict = {word: idx for idx, word in enumerate(sorted(self.vocab))}
        with open(os.path.join(output_dir, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # Save word frequencies
        with open(os.path.join(output_dir, "word_freq.json"), 'w', encoding='utf-8') as f:
            json.dump(self.word_freq, f, ensure_ascii=False, indent=2)

        # Save mappings
        with open(os.path.join(output_dir, "word_to_non_accented.json"), 'w', encoding='utf-8') as f:
            json.dump(self.word_to_non_accented, f,
                      ensure_ascii=False, indent=2)

        # Create keyboard mappings before saving
        self.create_keyboard_mappings()

        with open(os.path.join(output_dir, "non_accented_to_words.json"), 'w', encoding='utf-8') as f:
            json.dump(dict(self.non_accented_to_words),
                      f, ensure_ascii=False, indent=2)

        # Save training pairs
        if not self.training_pairs:
            self.create_training_pairs()

        df = pd.DataFrame(self.training_pairs)
        df.to_csv(os.path.join(output_dir, "training_pairs.csv"),
                  index=False, encoding='utf-8')

        # Save statistics
        with open(os.path.join(output_dir, "preprocessing_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        print(f"✅ Data saved successfully!")
        print(f"📊 Final Statistics:")
        for key, value in self.stats.items():
            print(f"   {key}: {value:,}")

        return self.stats

    def analyze_coverage(self) -> Dict:
        """Analyze vocabulary coverage and quality"""
        print("🔍 Analyzing vocabulary coverage...")

        analysis = {
            'total_vocab_size': len(self.vocab),
            'viet74k_coverage': self.stats['viet74k_words_integrated'],
            'corpus_coverage': self.stats['corpus_words_processed'],
            'avg_word_frequency': sum(self.word_freq.values()) / len(self.word_freq) if self.word_freq else 0,
            'high_freq_words': sum(1 for freq in self.word_freq.values() if freq > 100),
            'multi_accent_mappings': sum(1 for words in self.non_accented_to_words.values() if len(words) > 1),
            'single_syllable_words': sum(1 for word in self.vocab if len(word.split()) == 1),
            'multi_syllable_words': sum(1 for word in self.vocab if len(word.split()) > 1)
        }

        print("📈 Coverage Analysis:")
        for key, value in analysis.items():
            print(f"   {key}: {value:,}")

        return analysis


def main():
    """Main preprocessing function"""
    print("Starting Vietnamese Non-accented Preprocessing...")

    preprocessor = VietnameseNonAccentedPreprocessor("data/corpus-full.txt")

    # Process corpus (limited sample for development)
    preprocessor.process_corpus(sample_size=100000)  # Start with 100K lines

    # Build vocabulary
    preprocessor.build_vocabulary(min_freq=5)

    # Save processed data
    stats = preprocessor.save_processed_data("ml/data")

    print("Preprocessing completed!")
    print(f"Statistics: {stats}")

    return preprocessor


if __name__ == "__main__":
    main()
