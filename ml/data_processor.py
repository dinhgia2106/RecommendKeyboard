#!/usr/bin/env python3
"""
Vietnamese Data Processor
Xử lý Viet74K dictionary và corpus để xây dựng hệ thống gợi ý nâng cao
"""

import re
import json
import unicodedata
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict


class VietnameseDataProcessor:
    """Xử lý và phân loại dữ liệu tiếng Việt từ Viet74K và corpus"""

    def __init__(self, viet74k_path: str = "data/Viet74K.txt", corpus_path: str = "data/corpus-full.txt"):
        self.viet74k_path = viet74k_path
        self.corpus_path = corpus_path

        # Phân loại từ vựng
        self.syllables = {}           # Âm tiết đơn
        self.simple_words = {}        # Từ đơn giản 2-3 âm tiết
        self.compound_words = {}      # Từ ghép phức tạp
        self.technical_terms = {}     # Thuật ngữ chuyên môn
        self.place_names = {}         # Địa danh
        self.person_names = {}        # Tên người
        self.foreign_words = {}       # Từ ngoại lai

        # Statistics và patterns
        self.word_frequency = Counter()
        self.ngram_patterns = defaultdict(Counter)
        self.context_patterns = defaultdict(list)

        print("🚀 Vietnamese Data Processor initialized")

    def remove_diacritics(self, text: str) -> str:
        """Loại bỏ dấu tiếng Việt"""
        # Vietnamese diacritics mapping
        vietnamese_map = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd',
            # Uppercase
            'À': 'A', 'Á': 'A', 'Ạ': 'A', 'Ả': 'A', 'Ã': 'A',
            'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ậ': 'A', 'Ẩ': 'A', 'Ẫ': 'A',
            'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ặ': 'A', 'Ẳ': 'A', 'Ẵ': 'A',
            'È': 'E', 'É': 'E', 'Ẹ': 'E', 'Ẻ': 'E', 'Ẽ': 'E',
            'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ệ': 'E', 'Ể': 'E', 'Ễ': 'E',
            'Ì': 'I', 'Í': 'I', 'Ị': 'I', 'Ỉ': 'I', 'Ĩ': 'I',
            'Ò': 'O', 'Ó': 'O', 'Ọ': 'O', 'Ỏ': 'O', 'Õ': 'O',
            'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ộ': 'O', 'Ổ': 'O', 'Ỗ': 'O',
            'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ợ': 'O', 'Ở': 'O', 'Ỡ': 'O',
            'Ù': 'U', 'Ú': 'U', 'Ụ': 'U', 'Ủ': 'U', 'Ũ': 'U',
            'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ự': 'U', 'Ử': 'U', 'Ữ': 'U',
            'Ỳ': 'Y', 'Ý': 'Y', 'Ỵ': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y',
            'Đ': 'D'
        }

        result = ""
        for char in text:
            result += vietnamese_map.get(char, char)
        return result.lower()

    def classify_word(self, word: str) -> str:
        """Phân loại từ vựng theo category"""
        word = word.strip()
        if not word:
            return "unknown"

        # Check technical terms (có dấu gạch nối, từ ngoại lai)
        if '-' in word or any(c in word for c in 'xyzwq'):
            return "technical"

        # Check proper names (viết hoa)
        if word[0].isupper():
            if any(place in word.lower() for place in ['hà nội', 'hcm', 'tp.', 'quận', 'phường', 'xã']):
                return "place"
            return "name"

        # Check by syllable count
        syllables = word.split()
        if len(syllables) == 1:
            return "syllable"
        elif len(syllables) == 2:
            return "simple_word"
        else:
            return "compound_word"

    def process_viet74k(self) -> Dict:
        """Xử lý file Viet74K.txt và phân loại từ vựng"""
        print("📚 Processing Viet74K dictionary...")

        categories = {
            "syllable": [],
            "simple_word": [],
            "compound_word": [],
            "technical": [],
            "place": [],
            "name": [],
            "unknown": []
        }

        try:
            with open(self.viet74k_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    word = line.strip()
                    if not word or len(word) < 2:
                        continue

                    # Phân loại từ
                    category = self.classify_word(word)
                    categories[category].append(word)

                    # Tạo mapping không dấu -> có dấu
                    no_diacritic = self.remove_diacritics(word)
                    if category == "syllable":
                        self.syllables[no_diacritic] = word
                    elif category == "simple_word":
                        self.simple_words[no_diacritic] = word
                    elif category == "compound_word":
                        self.compound_words[no_diacritic] = word
                    elif category == "technical":
                        self.technical_terms[no_diacritic] = word
                    elif category == "place":
                        self.place_names[no_diacritic] = word
                    elif category == "name":
                        self.person_names[no_diacritic] = word

                    if line_num % 10000 == 0:
                        print(f"  Processed {line_num:,} words...")

        except FileNotFoundError:
            print(f"❌ File not found: {self.viet74k_path}")
            return {}

        # Statistics
        total_words = sum(len(cat) for cat in categories.values())
        print(f"\n📊 Viet74K Processing Results:")
        print(f"  Total words: {total_words:,}")
        for category, words in categories.items():
            if words:
                print(f"  {category.replace('_', ' ').title()}: {len(words):,}")

        return categories

    def analyze_corpus_sample(self, max_lines: int = 100000) -> Dict:
        """Phân tích mẫu corpus để extract patterns"""
        print(f"🔍 Analyzing corpus sample ({max_lines:,} lines)...")

        sentence_patterns = []
        word_sequences = []

        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > max_lines:
                        break

                    sentence = line.strip()
                    if len(sentence) < 10:  # Skip short lines
                        continue

                    # Clean sentence
                    sentence = re.sub(r'[^\w\s]', ' ', sentence)
                    words = sentence.split()

                    if len(words) >= 3:
                        sentence_patterns.append(' '.join(words))

                        # Extract word sequences
                        for i in range(len(words) - 1):
                            bigram = f"{words[i]} {words[i+1]}"
                            self.ngram_patterns['bigram'][bigram] += 1

                        for i in range(len(words) - 2):
                            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                            self.ngram_patterns['trigram'][trigram] += 1

                    if line_num % 10000 == 0:
                        print(f"  Processed {line_num:,} lines...")

        except FileNotFoundError:
            print(f"❌ Corpus file not found: {self.corpus_path}")
            return {}

        # Get top patterns
        top_bigrams = self.ngram_patterns['bigram'].most_common(1000)
        top_trigrams = self.ngram_patterns['trigram'].most_common(500)

        print(f"\n📊 Corpus Analysis Results:")
        print(f"  Bigrams found: {len(self.ngram_patterns['bigram']):,}")
        print(f"  Trigrams found: {len(self.ngram_patterns['trigram']):,}")
        print(f"  Sample patterns:")

        for pattern, count in top_bigrams[:5]:
            print(f"    {pattern} ({count:,})")

        return {
            'bigrams': top_bigrams,
            'trigrams': top_trigrams,
            'total_sentences': len(sentence_patterns)
        }

    def build_enhanced_dictionaries(self) -> Dict:
        """Xây dựng từ điển nâng cao từ dữ liệu đã xử lý"""
        print("🔧 Building enhanced dictionaries...")

        # Combine all dictionaries
        enhanced_dict = {
            'syllables': self.syllables,
            'simple_words': self.simple_words,
            'compound_words': self.compound_words,
            'technical_terms': self.technical_terms,
            'place_names': self.place_names,
            'person_names': self.person_names,
            'ngram_patterns': dict(self.ngram_patterns),
        }

        # Add common sentences from ngrams
        common_sentences = {}
        for trigram, count in self.ngram_patterns['trigram'].most_common(100):
            no_diacritic = self.remove_diacritics(trigram.replace(' ', ''))
            if len(no_diacritic) >= 8:  # Only longer sequences
                common_sentences[no_diacritic] = trigram

        enhanced_dict['common_sentences'] = common_sentences

        # Statistics
        total_entries = sum(len(d)
                            for d in enhanced_dict.values() if isinstance(d, dict))
        print(f"📊 Enhanced Dictionary Stats:")
        print(f"  Total entries: {total_entries:,}")
        print(f"  Syllables: {len(self.syllables):,}")
        print(f"  Simple words: {len(self.simple_words):,}")
        print(f"  Compound words: {len(self.compound_words):,}")
        print(f"  Common sentences: {len(common_sentences):,}")

        return enhanced_dict

    def save_processed_data(self, output_path: str = "data/processed_vietnamese_data.json"):
        """Lưu dữ liệu đã xử lý"""
        print(f"💾 Saving processed data to {output_path}...")

        # Process all data
        viet74k_data = self.process_viet74k()
        corpus_data = self.analyze_corpus_sample()
        enhanced_dict = self.build_enhanced_dictionaries()

        # Combine all data
        final_data = {
            'metadata': {
                'total_viet74k_words': sum(len(cat) for cat in viet74k_data.values()),
                'total_syllables': len(self.syllables),
                'total_compounds': len(self.compound_words),
                'bigram_patterns': len(self.ngram_patterns['bigram']),
                'trigram_patterns': len(self.ngram_patterns['trigram'])
            },
            'categories': viet74k_data,
            'dictionaries': enhanced_dict,
            'corpus_patterns': corpus_data
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Data saved successfully!")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

        return final_data

    def quick_test(self):
        """Test nhanh để kiểm tra functionality"""
        print("🧪 Quick test...")

        # Test diacritic removal
        test_words = ["học bài", "sinh viên", "Hà Nội", "máy tính"]
        for word in test_words:
            no_diacritic = self.remove_diacritics(word)
            print(f"  {word} → {no_diacritic}")

        # Test classification
        test_classify = ["tôi", "học bài",
                         "máy tính điện tử", "Nguyễn Văn A", "TP.HCM"]
        for word in test_classify:
            category = self.classify_word(word)
            print(f"  {word} → {category}")


def main():
    """Main function for testing"""
    processor = VietnameseDataProcessor()

    # Quick test
    processor.quick_test()

    # Process and save data
    processed_data = processor.save_processed_data()

    print("\n🎉 Vietnamese Data Processing completed!")
    return processed_data


if __name__ == "__main__":
    main()
