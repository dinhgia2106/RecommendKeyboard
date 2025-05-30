"""
Data Preprocessor for Vietnamese Non-accented Keyboard
Converts Vietnamese corpus to training data for neural model
"""

import re
import os
import pickle
import unicodedata
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict
import pandas as pd
from unidecode import unidecode
import json


class VietnameseNonAccentedPreprocessor:
    def __init__(self, corpus_path: str = "data/corpus-full.txt"):
        self.corpus_path = corpus_path
        self.vocab = {}
        self.word_to_non_accented = {}
        self.non_accented_to_words = defaultdict(list)
        self.word_freq = Counter()

        # Vietnamese specific patterns
        self.vietnamese_chars = set(
            "aăâeêioyuưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵđ")

    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text"""
        # Fix encoding issues
        try:
            # Try to decode if it's encoded
            if 'Ã' in text or 'â€' in text:
                text = text.encode('latin1').decode('utf-8')
        except:
            pass

        # Normalize unicode
        text = unicodedata.normalize('NFC', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep Vietnamese
        text = re.sub(
            r'[^\w\s\.\,\!\?\-\'\"àáảãạâấầẩẫậăắằẳẵặèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', text, flags=re.IGNORECASE)

        return text.lower()

    def vietnamese_to_non_accented(self, word: str) -> str:
        """Convert Vietnamese word to non-accented representation"""
        # Remove diacritics to get base form
        non_accented = unidecode(word)

        # Normalize some special cases
        non_accented = non_accented.replace('dd', 'd')  # đ -> d
        non_accented = non_accented.replace('ph', 'f')  # ph -> f (optional)
        non_accented = non_accented.replace('th', 't')  # th -> t (optional)
        non_accented = non_accented.replace('ch', 'c')  # ch -> c (optional)
        non_accented = non_accented.replace('nh', 'n')  # nh -> n (optional)
        non_accented = non_accented.replace('gh', 'g')  # gh -> g (optional)
        non_accented = non_accented.replace('kh', 'k')  # kh -> k (optional)

        return non_accented.lower()

    def extract_words_from_sentence(self, sentence: str) -> List[str]:
        """Extract valid Vietnamese words from sentence"""
        words = []

        # Split by whitespace and punctuation
        tokens = re.findall(
            r'\b[a-záàảãạâấầẩẫậăắằẳẵặèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+\b', sentence, re.IGNORECASE)

        for token in tokens:
            token = token.strip().lower()
            if len(token) >= 2 and any(c in self.vietnamese_chars for c in token):
                words.append(token)

        return words

    def process_corpus(self, max_lines: int = None, sample_size: int = 1000000):
        """Process the corpus file and extract word-non_accented mappings"""
        print(f"Processing corpus: {self.corpus_path}")

        processed_lines = 0
        total_words = 0

        try:
            with open(self.corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if max_lines and processed_lines >= max_lines:
                        break

                    if processed_lines >= sample_size:
                        break

                    # Clean the line
                    clean_line = self.clean_text(line)
                    if len(clean_line) < 10:  # Skip too short lines
                        continue

                    # Extract words
                    words = self.extract_words_from_sentence(clean_line)

                    for word in words:
                        if len(word) >= 2:
                            # Convert to non_accented
                            non_accented = self.vietnamese_to_non_accented(
                                word)

                            # Store mappings
                            self.word_to_non_accented[word] = non_accented
                            self.non_accented_to_words[non_accented].append(
                                word)
                            self.word_freq[word] += 1
                            total_words += 1

                    processed_lines += 1

                    if processed_lines % 10000 == 0:
                        print(
                            f"Processed {processed_lines} lines, {total_words} words")

        except Exception as e:
            print(f"Error processing corpus: {e}")

        print(
            f"Finished processing. Total: {processed_lines} lines, {total_words} words")
        print(f"Unique words: {len(self.word_to_non_accented)}")
        print(f"Unique non_accenteds: {len(self.non_accented_to_words)}")

    def build_vocabulary(self, min_freq: int = 5):
        """Build vocabulary from processed words"""
        # Filter by frequency
        filtered_words = {word: freq for word,
                          freq in self.word_freq.items() if freq >= min_freq}

        # Create vocabulary
        vocab_list = ['<pad>', '<unk>', '<sos>',
                      '<eos>'] + list(filtered_words.keys())
        self.vocab = {word: idx for idx, word in enumerate(vocab_list)}

        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab

    def create_training_pairs(self, context_length: int = 5) -> List[Tuple[str, str]]:
        """Create training pairs (non_accented -> word) with context"""
        training_pairs = []

        # Group words by frequency for better sampling
        high_freq_words = {word: freq for word,
                           freq in self.word_freq.items() if freq >= 50}
        medium_freq_words = {word: freq for word,
                             freq in self.word_freq.items() if 10 <= freq < 50}
        low_freq_words = {word: freq for word,
                          freq in self.word_freq.items() if 5 <= freq < 10}

        print(f"High freq words: {len(high_freq_words)}")
        print(f"Medium freq words: {len(medium_freq_words)}")
        print(f"Low freq words: {len(low_freq_words)}")

        # Create pairs for each frequency group
        for word_dict, name in [(high_freq_words, "high"), (medium_freq_words, "medium"), (low_freq_words, "low")]:
            for word in word_dict:
                non_accented = self.word_to_non_accented.get(word)
                if non_accented:
                    # Simple pair
                    training_pairs.append((non_accented, word))

                    # Add some context-based variations
                    similar_words = self.non_accented_to_words.get(
                        non_accented, [])
                    if len(similar_words) > 1:
                        # This non_accented maps to multiple words
                        # Limit to top 3
                        for similar_word in similar_words[:3]:
                            if similar_word != word and similar_word in self.word_freq:
                                training_pairs.append(
                                    (non_accented, similar_word))

        print(f"Created {len(training_pairs)} training pairs")
        return training_pairs

    def save_processed_data(self, output_dir: str = "ml/data"):
        """Save processed data for training"""
        os.makedirs(output_dir, exist_ok=True)

        # Save word-non_accented mappings
        with open(f"{output_dir}/word_to_non_accented.json", 'w', encoding='utf-8') as f:
            json.dump(self.word_to_non_accented, f,
                      ensure_ascii=False, indent=2)

        with open(f"{output_dir}/non_accented_to_words.json", 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in self.non_accented_to_words.items()},
                      f, ensure_ascii=False, indent=2)

        # Save vocabulary
        with open(f"{output_dir}/vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save word frequencies
        with open(f"{output_dir}/word_freq.json", 'w', encoding='utf-8') as f:
            json.dump(dict(self.word_freq.most_common(50000)),
                      f, ensure_ascii=False, indent=2)

        # Create training pairs and save
        training_pairs = self.create_training_pairs()

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(training_pairs, columns=['non_accented', 'word'])
        df.to_csv(f"{output_dir}/training_pairs.csv",
                  index=False, encoding='utf-8')

        print(f"Saved processed data to {output_dir}")
        print(f"Training pairs: {len(training_pairs)}")

        return {
            'vocab_size': len(self.vocab),
            'training_pairs': len(training_pairs),
            'unique_words': len(self.word_to_non_accented),
            'unique_non_accenteds': len(self.non_accented_to_words)
        }


def main():
    """Main preprocessing function"""
    print("Starting Vietnamese Non-accented Preprocessing...")

    preprocessor = VietnameseNonAccentedPreprocessor("data/corpus-full.txt")

    # Process corpus (limited sample for development)
    preprocessor.process_corpus(sample_size=100000)  # Start with 100K lines

    # Build vocabulary
    preprocessor.build_vocabulary(min_freq=5)

    # Save processed data
    stats = preprocessor.save_processed_data()

    print("Preprocessing completed!")
    print(f"Statistics: {stats}")

    return preprocessor


if __name__ == "__main__":
    main()
