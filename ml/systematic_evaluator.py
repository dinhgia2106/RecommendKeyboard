#!/usr/bin/env python3
"""
Systematic Vietnamese Keyboard Evaluator
Tách data thành train/test sets và đánh giá performance systematically
"""

import json
import random
import time
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd


class SystematicVietnameseEvaluator:
    """Evaluator systematic cho Vietnamese keyboard system"""

    def __init__(self, data_path: str = "data/processed_vietnamese_data.json"):
        self.data_path = data_path

        # Load all available data
        self.all_data = {}
        self.train_data = {}
        self.test_data = {}

        # Evaluation metrics
        self.evaluation_results = {}

        # Load và prepare data
        self.load_data()
        self.create_train_test_split()

    def load_data(self):
        """Load toàn bộ data từ processed file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Combine all dictionaries thành một dataset unified
            self.all_data = {}

            dictionaries = data.get('dictionaries', {})

            # Syllables
            syllables = dictionaries.get('syllables', {})
            for key, value in syllables.items():
                self.all_data[key] = {
                    'vietnamese': value,
                    'category': 'syllable',
                    'length': len(value.split()),
                    'source': 'viet74k'
                }

            # Simple words
            simple_words = dictionaries.get('simple_words', {})
            for key, value in simple_words.items():
                self.all_data[key] = {
                    'vietnamese': value,
                    'category': 'simple_word',
                    'length': len(value.split()),
                    'source': 'viet74k'
                }

            # Compound words
            compound_words = dictionaries.get('compound_words', {})
            for key, value in compound_words.items():
                self.all_data[key] = {
                    'vietnamese': value,
                    'category': 'compound_word',
                    'length': len(value.split()),
                    'source': 'viet74k'
                }

            # Common sentences từ corpus
            common_sentences = dictionaries.get('common_sentences', {})
            for key, value in common_sentences.items():
                self.all_data[key] = {
                    'vietnamese': value,
                    'category': 'sentence',
                    'length': len(value.split()),
                    'source': 'corpus'
                }

            print(f"📚 Loaded {len(self.all_data):,} total data points")

        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def create_train_test_split(self, test_ratio: float = 0.2, random_seed: int = 42):
        """Tách data thành train/test sets systematically"""
        random.seed(random_seed)

        # Group data by category để đảm bảo balanced split
        data_by_category = defaultdict(list)
        for key, value in self.all_data.items():
            data_by_category[value['category']].append((key, value))

        print(f"\n📊 Data Distribution by Category:")
        for category, items in data_by_category.items():
            print(f"  {category}: {len(items):,} items")

        # Split each category separately
        for category, items in data_by_category.items():
            random.shuffle(items)

            test_size = int(len(items) * test_ratio)
            test_items = items[:test_size]
            train_items = items[test_size:]

            # Add to train/test sets
            for key, value in train_items:
                self.train_data[key] = value

            for key, value in test_items:
                self.test_data[key] = value

            print(
                f"  {category}: Train={len(train_items):,}, Test={len(test_items):,}")

        print(f"\n✅ Train/Test Split Complete:")
        print(
            f"  Training Set: {len(self.train_data):,} items ({(1-test_ratio)*100:.1f}%)")
        print(
            f"  Test Set: {len(self.test_data):,} items ({test_ratio*100:.1f}%)")

    def evaluate_current_system(self, processor) -> Dict:
        """Đánh giá hệ thống hiện tại trên test set"""
        print(
            f"\n🧪 Evaluating Current System on {len(self.test_data):,} unseen test cases...")

        results = {
            'total_tests': len(self.test_data),
            'correct_predictions': 0,
            'partially_correct': 0,
            'incorrect_predictions': 0,
            'no_predictions': 0,
            'category_performance': defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0}),
            'detailed_errors': [],
            'processing_times': []
        }

        for i, (input_key, expected_data) in enumerate(self.test_data.items()):
            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{len(self.test_data):,}")

            expected_vietnamese = expected_data['vietnamese']
            category = expected_data['category']

            # Time the prediction
            start_time = time.time()
            predictions = processor.process_text(input_key, max_suggestions=3)
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)

            # Evaluate prediction
            results['category_performance'][category]['total'] += 1

            if not predictions:
                results['no_predictions'] += 1
                results['detailed_errors'].append({
                    'input': input_key,
                    'expected': expected_vietnamese,
                    'predicted': None,
                    'category': category,
                    'error_type': 'no_prediction'
                })
                continue

            # Check if any prediction matches
            best_prediction = predictions[0]['vietnamese_text']
            exact_match = any(pred['vietnamese_text'] ==
                              expected_vietnamese for pred in predictions)
            partial_match = any(expected_vietnamese in pred['vietnamese_text'] or
                                pred['vietnamese_text'] in expected_vietnamese for pred in predictions)

            if exact_match:
                results['correct_predictions'] += 1
                results['category_performance'][category]['correct'] += 1
            elif partial_match:
                results['partially_correct'] += 1
                results['category_performance'][category]['partial'] += 1
            else:
                results['incorrect_predictions'] += 1
                results['detailed_errors'].append({
                    'input': input_key,
                    'expected': expected_vietnamese,
                    'predicted': best_prediction,
                    'category': category,
                    'error_type': 'incorrect'
                })

        # Calculate metrics
        total = results['total_tests']
        results['accuracy'] = results['correct_predictions'] / total * 100
        results['partial_accuracy'] = (
            results['correct_predictions'] + results['partially_correct']) / total * 100
        results['avg_processing_time'] = sum(
            results['processing_times']) / len(results['processing_times']) * 1000  # ms

        return results

    def analyze_error_patterns(self, results: Dict) -> Dict:
        """Phân tích patterns của errors để identify improvement areas"""
        print(f"\n🔍 Analyzing Error Patterns...")

        error_analysis = {
            'common_error_types': defaultdict(int),
            'error_by_length': defaultdict(int),
            'error_by_category': defaultdict(int),
            'systematic_failures': [],
            'improvement_suggestions': []
        }

        for error in results['detailed_errors']:
            input_text = error['input']
            expected = error['expected']
            predicted = error['predicted']
            category = error['category']

            # Error by category
            error_analysis['error_by_category'][category] += 1

            # Error by input length
            length_group = f"{len(input_text)//3*3}-{len(input_text)//3*3+2} chars"
            error_analysis['error_by_length'][length_group] += 1

            # Analyze error patterns
            if predicted is None:
                error_analysis['common_error_types']['no_prediction'] += 1
            elif 'inch' in predicted or any(char in predicted for char in 'xyzwq'):
                error_analysis['common_error_types']['english_contamination'] += 1
            elif len(predicted.split()) != len(expected.split()):
                error_analysis['common_error_types']['wrong_segmentation'] += 1
            else:
                error_analysis['common_error_types']['vocabulary_gap'] += 1

        # Generate improvement suggestions
        if error_analysis['common_error_types']['english_contamination'] > 10:
            error_analysis['improvement_suggestions'].append(
                "Add English word filtering in segmentation"
            )

        if error_analysis['common_error_types']['wrong_segmentation'] > 50:
            error_analysis['improvement_suggestions'].append(
                "Improve segmentation algorithm with better context awareness"
            )

        if error_analysis['common_error_types']['vocabulary_gap'] > 100:
            error_analysis['improvement_suggestions'].append(
                "Expand vocabulary with more domain-specific terms"
            )

        return error_analysis

    def generate_improvement_dataset(self, results: Dict, top_n: int = 1000) -> List[Dict]:
        """Generate dataset for training improvement model"""
        print(f"\n🎯 Generating Improvement Dataset (top {top_n} errors)...")

        # Get top errors by frequency and impact
        error_counts = defaultdict(int)
        for error in results['detailed_errors']:
            error_pattern = f"{error['input']} -> {error['expected']}"
            error_counts[error_pattern] += 1

        # Sort by frequency
        top_errors = sorted(error_counts.items(),
                            key=lambda x: x[1], reverse=True)[:top_n]

        improvement_dataset = []
        for error_pattern, frequency in top_errors:
            input_text, expected = error_pattern.split(' -> ')
            improvement_dataset.append({
                'input': input_text,
                'expected_output': expected,
                'frequency': frequency,
                'priority': 'high' if frequency > 5 else 'medium' if frequency > 2 else 'low'
            })

        return improvement_dataset

    def save_evaluation_report(self, results: Dict, error_analysis: Dict, filename: str = "systematic_evaluation_report.json"):
        """Save comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_data_points': len(self.all_data),
                'train_size': len(self.train_data),
                'test_size': len(self.test_data)
            },
            'performance_metrics': {
                'accuracy': results['accuracy'],
                'partial_accuracy': results['partial_accuracy'],
                'avg_processing_time_ms': results['avg_processing_time'],
                'total_tests': results['total_tests'],
                'correct_predictions': results['correct_predictions'],
                'partially_correct': results['partially_correct'],
                'incorrect_predictions': results['incorrect_predictions'],
                'no_predictions': results['no_predictions']
            },
            'category_performance': dict(results['category_performance']),
            'error_analysis': dict(error_analysis),
            'top_errors': results['detailed_errors'][:100]  # Top 100 errors
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📊 Evaluation report saved to {filename}")


def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = SystematicVietnameseEvaluator()

    # Load current system
    from hybrid_vietnamese_processor import HybridVietnameseProcessor
    processor = HybridVietnameseProcessor()

    # Evaluate current system
    results = evaluator.evaluate_current_system(processor)

    # Analyze errors
    error_analysis = evaluator.analyze_error_patterns(results)

    # Generate improvement dataset
    improvement_dataset = evaluator.generate_improvement_dataset(results)

    # Print summary
    print(f"\n📊 SYSTEMATIC EVALUATION RESULTS:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Partial Accuracy: {results['partial_accuracy']:.2f}%")
    print(f"  Avg Processing Time: {results['avg_processing_time']:.2f}ms")
    print(f"  Total Errors: {len(results['detailed_errors']):,}")

    print(f"\n🔍 ERROR ANALYSIS:")
    for error_type, count in error_analysis['common_error_types'].items():
        print(f"  {error_type}: {count}")

    print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
    for suggestion in error_analysis['improvement_suggestions']:
        print(f"  - {suggestion}")

    # Save report
    evaluator.save_evaluation_report(results, error_analysis)

    return evaluator, results, error_analysis, improvement_dataset


if __name__ == "__main__":
    main()
