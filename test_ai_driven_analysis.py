#!/usr/bin/env python3
"""
AI-Driven Analysis for Vietnamese Keyboard Issues
Thay vì thêm thủ công, sử dụng AI để phân tích và học từ data
"""

import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor


class AIPatternAnalyzer:
    """AI-driven pattern analyzer thay vì thêm thủ công"""

    def __init__(self, data_path: str = "data/processed_vietnamese_data.json"):
        self.data_path = data_path
        self.processor = HybridVietnameseProcessor()
        self.raw_data = self.load_raw_data()

    def load_raw_data(self) -> Dict:
        """Load raw data để phân tích patterns"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {}

    def analyze_missing_pattern(self, test_input: str, expected_output: str):
        """AI analysis tại sao pattern bị thiếu"""
        print(f"\n🔍 AI Analysis for: '{test_input}' → '{expected_output}'")

        # 1. Phân tích có trong data không
        self.check_in_existing_data(test_input, expected_output)

        # 2. Tìm similar patterns trong data
        similar_patterns = self.find_similar_patterns(
            test_input, expected_output)

        # 3. Phân tích corpus patterns
        corpus_analysis = self.analyze_corpus_patterns(
            test_input, expected_output)

        # 4. Generate AI recommendations
        recommendations = self.generate_ai_recommendations(
            test_input, expected_output, similar_patterns)

        return {
            'input': test_input,
            'expected': expected_output,
            'similar_patterns': similar_patterns,
            'corpus_analysis': corpus_analysis,
            'ai_recommendations': recommendations
        }

    def check_in_existing_data(self, input_text: str, expected: str):
        """Kiểm tra có sẵn trong data không"""
        dictionaries = self.raw_data.get('dictionaries', {})

        found_exact = False
        found_similar = []

        for dict_name, dict_data in dictionaries.items():
            if input_text in dict_data:
                found_exact = True
                print(
                    f"  ✅ Found EXACT match in {dict_name}: {dict_data[input_text]}")

            # Tìm similar keys
            for key, value in dict_data.items():
                if (key != input_text and
                    (input_text in key or key in input_text) and
                        len(key) >= 4):
                    found_similar.append({
                        'key': key,
                        'value': value,
                        'dict': dict_name,
                        'similarity': self.calculate_similarity(input_text, key)
                    })

        if not found_exact:
            print(f"  ❌ NOT found in any dictionary")

        if found_similar:
            print(f"  🔍 Found {len(found_similar)} similar patterns:")
            for item in sorted(found_similar, key=lambda x: x['similarity'], reverse=True)[:5]:
                print(
                    f"     {item['key']} → {item['value']} ({item['similarity']:.2f}) [{item['dict']}]")

    def find_similar_patterns(self, input_text: str, expected: str) -> List[Dict]:
        """Tìm patterns tương tự trong data để học"""
        dictionaries = self.raw_data.get('dictionaries', {})
        similar_patterns = []

        # Pattern 1: Same structure (toi + verb + object)
        toi_patterns = []
        for dict_name, dict_data in dictionaries.items():
            for key, value in dict_data.items():
                if (isinstance(value, str) and
                    key.startswith('toi') and
                    len(key) > 6 and
                        len(value.split()) == 3):  # tôi + verb + object
                    toi_patterns.append({
                        'key': key,
                        'value': value,
                        'structure': 'toi_verb_object',
                        'dict': dict_name
                    })

        # Pattern 2: Similar length and structure
        target_length = len(input_text)
        target_words = len(expected.split())

        for dict_name, dict_data in dictionaries.items():
            for key, value in dict_data.items():
                if (isinstance(value, str) and
                    abs(len(key) - target_length) <= 2 and
                        len(value.split()) == target_words):
                    similarity = self.calculate_similarity(input_text, key)
                    if similarity > 0.4:  # Threshold
                        similar_patterns.append({
                            'key': key,
                            'value': value,
                            'similarity': similarity,
                            'structure': f"{len(value.split())}_words",
                            'dict': dict_name
                        })

        return similar_patterns

    def analyze_corpus_patterns(self, input_text: str, expected: str) -> Dict:
        """Phân tích corpus để tìm frequency patterns"""
        corpus_patterns = self.raw_data.get('corpus_patterns', {})

        analysis = {
            'bigrams': [],
            'trigrams': [],
            'frequency_score': 0
        }

        expected_words = expected.split()

        # Check bigrams
        if len(expected_words) >= 2:
            for i in range(len(expected_words) - 1):
                bigram = f"{expected_words[i]} {expected_words[i+1]}"
                for pattern, freq in corpus_patterns.get('bigrams', []):
                    if pattern == bigram:
                        analysis['bigrams'].append({
                            'pattern': bigram,
                            'frequency': freq,
                            'rank': self.get_pattern_rank(pattern, corpus_patterns.get('bigrams', []))
                        })

        # Check trigrams
        if len(expected_words) >= 3:
            trigram = ' '.join(expected_words)
            for pattern, freq in corpus_patterns.get('trigrams', []):
                if pattern == trigram:
                    analysis['trigrams'].append({
                        'pattern': trigram,
                        'frequency': freq,
                        'rank': self.get_pattern_rank(pattern, corpus_patterns.get('trigrams', []))
                    })

        return analysis

    def generate_ai_recommendations(self, input_text: str, expected: str, similar_patterns: List[Dict]) -> Dict:
        """Generate AI recommendations thay vì thêm thủ công"""

        recommendations = {
            'approach': 'data_driven',
            'confidence': 0.0,
            'methods': []
        }

        # Method 1: Pattern-based learning
        if similar_patterns:
            pattern_confidence = len(similar_patterns) / 10  # Normalize
            recommendations['methods'].append({
                'method': 'pattern_learning',
                'confidence': min(pattern_confidence, 0.9),
                'description': f"Learn from {len(similar_patterns)} similar patterns",
                'implementation': 'Extract common segmentation rules from similar patterns'
            })

        # Method 2: Corpus frequency analysis
        words = expected.split()
        if len(words) <= 3:
            recommendations['methods'].append({
                'method': 'corpus_frequency',
                'confidence': 0.8,
                'description': "Analyze corpus frequency for pattern validation",
                'implementation': 'Use bigram/trigram frequencies to validate segmentation'
            })

        # Method 3: Segmentation rules learning
        if len(input_text) >= 6:
            recommendations['methods'].append({
                'method': 'segmentation_learning',
                'confidence': 0.75,
                'description': "Learn optimal segmentation from data",
                'implementation': 'Train segmentation model on existing patterns'
            })

        # Calculate overall confidence
        if recommendations['methods']:
            recommendations['confidence'] = max(
                m['confidence'] for m in recommendations['methods'])

        return recommendations

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two strings"""
        # Simple Jaccard similarity with character bigrams
        def get_bigrams(text):
            return set(text[i:i+2] for i in range(len(text)-1))

        bigrams1 = get_bigrams(text1)
        bigrams2 = get_bigrams(text2)

        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))

        return intersection / union

    def get_pattern_rank(self, pattern: str, pattern_list: List[Tuple]) -> int:
        """Get rank of pattern in frequency list"""
        for i, (p, freq) in enumerate(pattern_list):
            if p == pattern:
                return i + 1
        return -1

    def demonstrate_ai_approach(self):
        """Demonstrate AI approach vs manual approach"""
        print("🤖 AI-DRIVEN APPROACH vs ✋ MANUAL APPROACH")
        print("=" * 60)

        test_cases = [
            ("toidemden", "tôi đem đến"),
            ("toitangban", "tôi tặng bạn"),
            ("toilambep", "tôi làm bếp"),
            ("toidicho", "tôi đi chợ")
        ]

        for input_text, expected in test_cases:
            print(f"\n📝 Case: {input_text} → {expected}")

            # Current performance
            current_results = self.processor.process_text(
                input_text, max_suggestions=3)
            if current_results:
                best = current_results[0]
                print(
                    f"  Current: {best['vietnamese_text']} ({best['confidence']}%)")

                if best['vietnamese_text'] == expected:
                    print("  ✅ Already correct!")
                    continue
            else:
                print("  ❌ No suggestions")

            # AI Analysis
            analysis = self.analyze_missing_pattern(input_text, expected)

            print(f"  🤖 AI Recommendations:")
            for method in analysis['ai_recommendations']['methods']:
                print(
                    f"     • {method['method']}: {method['confidence']:.0%} - {method['description']}")


def main():
    """Main demonstration"""
    analyzer = AIPatternAnalyzer()
    analyzer.demonstrate_ai_approach()


if __name__ == "__main__":
    main()
