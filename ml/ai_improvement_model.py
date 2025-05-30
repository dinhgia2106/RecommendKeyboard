#!/usr/bin/env python3
"""
AI Improvement Model for Vietnamese Keyboard
Sử dụng AI để học từ error patterns và cải tiến systematic
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import re


class VietnameseAIImprovementModel:
    """AI Model để improvement systematic Vietnamese keyboard"""

    def __init__(self, evaluation_report_path: str = "systematic_evaluation_report.json"):
        self.evaluation_report = self.load_evaluation_report(
            evaluation_report_path)

        # AI Components
        self.segmentation_patterns = {}
        self.vocabulary_expansion_candidates = {}
        self.context_rules = {}

        # Learning từ errors
        self.learn_from_errors()

    def load_evaluation_report(self, path: str) -> Dict:
        """Load evaluation report để analyze"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            print(
                f"📊 Loaded evaluation report: {report['performance_metrics']['accuracy']:.2f}% accuracy")
            return report
        except Exception as e:
            print(f"❌ Error loading evaluation report: {e}")
            return {}

    def learn_from_errors(self):
        """AI Learning từ systematic errors"""
        print("🧠 AI Learning from Error Patterns...")

        top_errors = self.evaluation_report.get('top_errors', [])

        # 1. Learn segmentation patterns
        self.learn_segmentation_patterns(top_errors)

        # 2. Learn vocabulary gaps
        self.learn_vocabulary_gaps(top_errors)

        # 3. Learn context rules
        self.learn_context_rules(top_errors)

    def learn_segmentation_patterns(self, errors: List[Dict]):
        """Learn optimal segmentation patterns từ errors"""
        print("  📝 Learning Segmentation Patterns...")

        segmentation_errors = [e for e in errors if e.get('error_type') == 'incorrect'
                               and e.get('predicted') and len(e['predicted'].split()) != len(e['expected'].split())]

        for error in segmentation_errors:
            input_text = error['input']
            expected = error['expected']
            predicted = error['predicted']

            # Analyze optimal segmentation
            expected_words = expected.split()
            input_length = len(input_text)

            # Calculate optimal segment lengths
            segment_lengths = []
            current_pos = 0

            for word in expected_words:
                # Remove diacritics để match với input
                word_no_diacritics = self.remove_diacritics(word)
                word_length = len(word_no_diacritics)
                segment_lengths.append(word_length)
                current_pos += word_length

            # Store segmentation pattern
            pattern_key = f"length_{input_length}_words_{len(expected_words)}"
            if pattern_key not in self.segmentation_patterns:
                self.segmentation_patterns[pattern_key] = []

            self.segmentation_patterns[pattern_key].append({
                'input': input_text,
                'expected_segments': segment_lengths,
                'expected_output': expected,
                'frequency': 1
            })

        print(
            f"    Learned {len(self.segmentation_patterns)} segmentation patterns")

    def learn_vocabulary_gaps(self, errors: List[Dict]):
        """Identify vocabulary gaps cần fill"""
        print("  📚 Identifying Vocabulary Gaps...")

        vocabulary_errors = [e for e in errors if e.get(
            'error_type') == 'incorrect']

        gap_patterns = defaultdict(int)

        for error in vocabulary_errors:
            input_text = error['input']
            expected = error['expected']

            # Categorize vocabulary gaps
            if len(expected.split()) == 1:
                gap_patterns['single_word'] += 1
                self.vocabulary_expansion_candidates[input_text] = {
                    'expected': expected,
                    'category': 'single_word',
                    'priority': 'high'
                }
            elif len(expected.split()) == 2:
                gap_patterns['compound_word'] += 1
                self.vocabulary_expansion_candidates[input_text] = {
                    'expected': expected,
                    'category': 'compound_word',
                    'priority': 'medium'
                }
            else:
                gap_patterns['phrase'] += 1
                self.vocabulary_expansion_candidates[input_text] = {
                    'expected': expected,
                    'category': 'phrase',
                    'priority': 'low'
                }

        print(
            f"    Identified {len(self.vocabulary_expansion_candidates)} vocabulary gaps")
        for gap_type, count in gap_patterns.items():
            print(f"      {gap_type}: {count}")

    def learn_context_rules(self, errors: List[Dict]):
        """Learn context rules từ error patterns"""
        print("  🎯 Learning Context Rules...")

        context_errors = [e for e in errors if e.get(
            'error_type') == 'incorrect']

        for error in context_errors:
            input_text = error['input']
            expected = error['expected']

            # Extract context patterns
            if len(input_text) >= 6:
                # Analyze prefix/suffix patterns
                prefix = input_text[:3]
                suffix = input_text[-3:]

                prefix_rule = f"prefix_{prefix}"
                suffix_rule = f"suffix_{suffix}"

                if prefix_rule not in self.context_rules:
                    self.context_rules[prefix_rule] = []
                if suffix_rule not in self.context_rules:
                    self.context_rules[suffix_rule] = []

                self.context_rules[prefix_rule].append({
                    'input': input_text,
                    'expected': expected,
                    'confidence': 0.7
                })

                self.context_rules[suffix_rule].append({
                    'input': input_text,
                    'expected': expected,
                    'confidence': 0.6
                })

        print(f"    Learned {len(self.context_rules)} context rules")

    def remove_diacritics(self, text: str) -> str:
        """Remove Vietnamese diacritics"""
        diacritics_map = {
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
            'đ': 'd'
        }

        result = ""
        for char in text.lower():
            result += diacritics_map.get(char, char)
        return result

    def generate_ai_improvements(self) -> Dict:
        """Generate AI-driven improvements"""
        print("\n🚀 Generating AI-Driven Improvements...")

        improvements = {
            'segmentation_rules': self.generate_segmentation_rules(),
            'vocabulary_additions': self.generate_vocabulary_additions(),
            'context_enhancements': self.generate_context_enhancements(),
            'algorithm_optimizations': self.generate_algorithm_optimizations()
        }

        return improvements

    def generate_segmentation_rules(self) -> List[Dict]:
        """Generate improved segmentation rules"""
        rules = []

        # Analyze patterns để tạo rules
        pattern_frequency = Counter()
        for pattern_key, examples in self.segmentation_patterns.items():
            pattern_frequency[pattern_key] = len(examples)

        # Top patterns thành rules
        for pattern_key, frequency in pattern_frequency.most_common(50):
            if frequency >= 3:  # Chỉ lấy patterns xuất hiện nhiều lần
                examples = self.segmentation_patterns[pattern_key]

                # Calculate average segment lengths
                avg_segments = []
                for example in examples:
                    segments = example['expected_segments']
                    if len(avg_segments) == 0:
                        avg_segments = segments[:]
                    else:
                        for i in range(min(len(avg_segments), len(segments))):
                            avg_segments[i] = (
                                avg_segments[i] + segments[i]) / 2

                rules.append({
                    'pattern': pattern_key,
                    'frequency': frequency,
                    'optimal_segments': [int(x) for x in avg_segments],
                    'confidence': min(0.9, 0.5 + (frequency * 0.1))
                })

        return rules

    def generate_vocabulary_additions(self) -> List[Dict]:
        """Generate vocabulary additions based on gaps"""
        additions = []

        # Priority order
        priority_order = ['high', 'medium', 'low']

        for priority in priority_order:
            priority_additions = []

            for input_text, gap_info in self.vocabulary_expansion_candidates.items():
                if gap_info['priority'] == priority:
                    priority_additions.append({
                        'input': input_text,
                        'expected': gap_info['expected'],
                        'category': gap_info['category'],
                        'priority': priority,
                        'confidence': 0.95 if priority == 'high' else 0.85 if priority == 'medium' else 0.75
                    })

            # Limit per priority
            max_per_priority = {'high': 100, 'medium': 200, 'low': 100}
            additions.extend(
                priority_additions[:max_per_priority.get(priority, 50)])

        return additions

    def generate_context_enhancements(self) -> List[Dict]:
        """Generate context-aware enhancements"""
        enhancements = []

        # Consolidate context rules
        rule_frequency = Counter()
        for rule_key, examples in self.context_rules.items():
            rule_frequency[rule_key] = len(examples)

        # Top context rules
        for rule_key, frequency in rule_frequency.most_common(20):
            if frequency >= 5:
                examples = self.context_rules[rule_key]

                enhancements.append({
                    'rule_type': rule_key,
                    'frequency': frequency,
                    'examples': examples[:5],  # Top 5 examples
                    'confidence': min(0.8, 0.4 + (frequency * 0.05))
                })

        return enhancements

    def generate_algorithm_optimizations(self) -> List[Dict]:
        """Generate algorithm optimization suggestions"""
        optimizations = []

        error_analysis = self.evaluation_report.get('error_analysis', {})
        common_errors = error_analysis.get('common_error_types', {})

        # Algorithm optimizations based on error patterns
        if common_errors.get('wrong_segmentation', 0) > 100:
            optimizations.append({
                'type': 'segmentation_algorithm',
                'suggestion': 'Implement dynamic programming with learned segment patterns',
                'impact': 'high',
                'estimated_improvement': '2-3%'
            })

        if common_errors.get('vocabulary_gap', 0) > 50:
            optimizations.append({
                'type': 'vocabulary_coverage',
                'suggestion': 'Add domain-specific dictionaries based on error analysis',
                'impact': 'medium',
                'estimated_improvement': '1-2%'
            })

        return optimizations

    def save_ai_improvements(self, improvements: Dict, filename: str = "ai_improvements.json"):
        """Save AI-generated improvements"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(improvements, f, ensure_ascii=False, indent=2)

        print(f"💾 AI improvements saved to {filename}")

        # Print summary
        print(f"\n📊 AI IMPROVEMENT SUMMARY:")
        print(
            f"  Segmentation Rules: {len(improvements['segmentation_rules'])}")
        print(
            f"  Vocabulary Additions: {len(improvements['vocabulary_additions'])}")
        print(
            f"  Context Enhancements: {len(improvements['context_enhancements'])}")
        print(
            f"  Algorithm Optimizations: {len(improvements['algorithm_optimizations'])}")


def main():
    """Main AI improvement function"""
    # Initialize AI model
    ai_model = VietnameseAIImprovementModel()

    # Generate improvements
    improvements = ai_model.generate_ai_improvements()

    # Save improvements
    ai_model.save_ai_improvements(improvements)

    return ai_model, improvements


if __name__ == "__main__":
    main()
