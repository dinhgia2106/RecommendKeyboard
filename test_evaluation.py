"""
Enhanced Testing and Evaluation System for Vietnamese Non-accented GPT
Comprehensive metrics and test suite for model evaluation
"""

from ml.models.gpt_model import load_model
from ml.tokenizer import get_tokenizer
from ml.inference import get_inference_engine, VietnameseNonAccentedInference
import os
import sys
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import torch
import time
from dataclasses import dataclass
import argparse

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TestMetrics:
    """Metrics for model evaluation"""
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    exact_match_accuracy: float = 0.0
    character_accuracy: float = 0.0
    syllable_accuracy: float = 0.0
    average_confidence: float = 0.0
    inference_time_ms: float = 0.0
    coverage: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'top1_accuracy': self.top1_accuracy,
            'top3_accuracy': self.top3_accuracy,
            'top5_accuracy': self.top5_accuracy,
            'exact_match_accuracy': self.exact_match_accuracy,
            'character_accuracy': self.character_accuracy,
            'syllable_accuracy': self.syllable_accuracy,
            'average_confidence': self.average_confidence,
            'inference_time_ms': self.inference_time_ms,
            'coverage': self.coverage
        }


class VietnameseTestSuite:
    """Comprehensive test suite for Vietnamese GPT model"""

    def __init__(self,
                 model_path: str = "checkpoints/vietnamese_non_accented_gpt_best.pth",
                 data_dir: str = "ml/data",
                 viet74k_path: str = "data/Viet74K.txt"):

        self.model_path = model_path
        self.data_dir = data_dir
        self.viet74k_path = viet74k_path

        # Load components
        print("🚀 Initializing Vietnamese Test Suite...")
        self.inference_engine = None
        self.tokenizer = None

        # Test cases
        self.basic_test_cases = []
        self.challenging_test_cases = []
        self.viet74k_test_cases = []
        self.context_test_cases = []

        # Results storage
        self.test_results = {}

        self._load_components()
        self._create_test_cases()

    def _load_components(self):
        """Load model and tokenizer"""
        try:
            self.inference_engine = get_inference_engine(
                model_path=self.model_path,
                data_dir=self.data_dir
            )
            self.tokenizer = get_tokenizer(self.data_dir)
            print("✅ Components loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading components: {e}")
            print("Will create test cases only")

    def _create_test_cases(self):
        """Create comprehensive test cases"""
        print("📝 Creating test cases...")

        # Basic test cases - common words
        self.basic_test_cases = [
            ("xinchao", "xin chào", []),
            ("chao", "chào", []),
            ("cam", "cảm", ["xin"]),
            ("on", "ơn", ["cảm"]),
            ("ban", "bạn", ["tôi", "là"]),
            ("toi", "tôi", []),
            ("la", "là", ["tôi"]),
            ("hoc", "học", ["đang"]),
            ("sinh", "sinh", ["học"]),
            ("viet", "việt", ["tiếng"]),
            ("nam", "nam", ["việt"]),
            ("dep", "đẹp", ["rất"]),
            ("yeu", "yêu", ["tôi"]),
            ("thuong", "thương", ["yêu"]),
            ("nha", "nhà", ["về"]),
            ("truong", "trường", ["đến"]),
            ("lam", "làm", ["đang"]),
            ("viec", "việc", ["làm"]),
            ("co", "có", ["tôi"]),
            ("khong", "không", ["có"]),
            # Keyboard-specific test cases (no spaces)
            ("moinguoi", "mọi người", []),  # "moi nguoi" → "mọi người"
            ("nguoiviet", "người việt", []),  # "nguoi viet" → "người việt"
            ("datnuoc", "đất nước", []),   # "dat nuoc" → "đất nước"
            ("hocsinh", "học sinh", []),   # "hoc sinh" → "học sinh"
            ("giaovien", "giáo viên", []),  # "giao vien" → "giáo viên"
            ("xinchao", "xin chào", []),   # "xin chao" → "xin chào"
            ("bongsu", "bông sứ", []),     # "bong su" → "bông sứ"
        ]

        # Challenging test cases - words with multiple possible mappings
        self.challenging_test_cases = [
            ("ma", ["ma", "mà", "má", "mả", "mã"], ["đây", "là"]),
            ("ban", ["ban", "bạn", "bàn"], ["cái"]),
            ("ca", ["ca", "cá", "cà"], ["hát"]),
            ("da", ["da", "dạ", "đã"], ["đã"]),
            ("co", ["co", "có", "cô", "cò"], ["tôi"]),
            ("do", ["do", "đỏ", "đo"], ["màu"]),
            ("an", ["an", "ăn", "ần"], ["đi"]),
            ("ong", ["ông", "ong"], ["bà"]),
            ("ba", ["ba", "bà", "bá", "bạ"], ["mẹ"]),
            ("me", ["mẹ", "mê", "mé"], ["ba"]),
            ("con", ["con", "còn", "cơn"], ["một"]),
            ("den", ["đen", "đến", "đèn"], ["màu"]),
            ("trang", ["trắng", "trang"], ["màu"]),
            ("do", ["đỏ", "do", "đo"], ["màu"]),
            ("xanh", ["xanh"], ["màu"]),
        ]

        # Load Viet74K test cases
        self._load_viet74k_test_cases()

        # Context-aware test cases
        self.context_test_cases = [
            ("hom", "hôm", ["ngày", "mai"]),
            ("nay", "nay", ["hôm"]),
            ("qua", "qua", ["hôm"]),
            ("roi", "rồi", ["xong"]),
            ("chua", "chưa", ["còn"]),
            ("da", "đã", ["tôi"]),
            ("se", "sẽ", ["tôi"]),
            ("dang", "đang", ["tôi"]),
            ("can", "cần", ["tôi"]),
            ("muon", "muốn", ["tôi"]),
        ]

        print(f"✅ Created test cases:")
        print(f"   Basic: {len(self.basic_test_cases)}")
        print(f"   Challenging: {len(self.challenging_test_cases)}")
        print(f"   Viet74K: {len(self.viet74k_test_cases)}")
        print(f"   Context: {len(self.context_test_cases)}")

    def _load_viet74k_test_cases(self):
        """Load test cases from Viet74K dictionary"""
        if not os.path.exists(self.viet74k_path):
            print(f"⚠️ Viet74K file not found at {self.viet74k_path}")
            return

        viet74k_samples = []
        try:
            with open(self.viet74k_path, 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]

            # Sample random words for testing
            if len(words) > 1000:
                viet74k_samples = random.sample(words, 1000)
            else:
                viet74k_samples = words

            # Create test cases
            for word in viet74k_samples:
                if len(word.split()) == 1 and len(word) >= 2:  # Single word
                    non_accented = self._remove_accents(word)
                    if non_accented != word:  # Only if different
                        self.viet74k_test_cases.append(
                            (non_accented, word, []))

        except Exception as e:
            print(f"❌ Error loading Viet74K test cases: {e}")

    def _remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents"""
        accent_map = {
            'àáạảãâầấậẩẫăằắặẳẵ': 'a',
            'èéẹẻẽêềếệểễ': 'e',
            'ìíịỉĩ': 'i',
            'òóọỏõôồốộổỗơờớợởỡ': 'o',
            'ùúụủũưừứựửữ': 'u',
            'ỳýỵỷỹ': 'y',
            'đ': 'd'
        }

        result = text
        for accented_chars, non_accented in accent_map.items():
            for char in accented_chars:
                result = result.replace(char, non_accented)

        return result

    def calculate_character_accuracy(self, predicted: str, target: str) -> float:
        """Calculate character-level accuracy"""
        if not predicted or not target:
            return 0.0

        # Convert to character lists
        pred_chars = list(predicted.lower())
        target_chars = list(target.lower())

        # Simple edit distance-based accuracy
        max_len = max(len(pred_chars), len(target_chars))
        if max_len == 0:
            return 1.0

        # Count matching characters at same positions
        matches = sum(1 for i in range(min(len(pred_chars), len(target_chars)))
                      if pred_chars[i] == target_chars[i])

        return matches / max_len

    def calculate_syllable_accuracy(self, predicted: str, target: str) -> float:
        """Calculate syllable-level accuracy"""
        pred_syllables = predicted.lower().split()
        target_syllables = target.lower().split()

        if not pred_syllables or not target_syllables:
            return 0.0

        max_len = max(len(pred_syllables), len(target_syllables))
        matches = sum(1 for i in range(min(len(pred_syllables), len(target_syllables)))
                      if pred_syllables[i] == target_syllables[i])

        return matches / max_len

    def evaluate_predictions(self,
                             predictions: List[Tuple[str, float, str]],
                             target: str,
                             targets_list: List[str] = None) -> Dict:
        """Evaluate predictions against target(s)"""
        if not predictions:
            return {
                'top1_hit': False,
                'top3_hit': False,
                'top5_hit': False,
                'exact_match': False,
                'char_accuracy': 0.0,
                'syllable_accuracy': 0.0,
                'confidence': 0.0
            }

        # Extract predicted words
        pred_words = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]

        # Check if target is string or list
        if targets_list:
            target_set = set(targets_list)
        else:
            target_set = {target}

        # Calculate hits
        top1_hit = len(pred_words) > 0 and pred_words[0] in target_set
        top3_hit = any(word in target_set for word in pred_words[:3])
        top5_hit = any(word in target_set for word in pred_words[:5])
        exact_match = len(pred_words) > 0 and pred_words[0] == target

        # Character and syllable accuracy (using first prediction)
        if pred_words:
            char_accuracy = self.calculate_character_accuracy(
                pred_words[0], target)
            syllable_accuracy = self.calculate_syllable_accuracy(
                pred_words[0], target)
            confidence = confidences[0] if confidences else 0.0
        else:
            char_accuracy = 0.0
            syllable_accuracy = 0.0
            confidence = 0.0

        return {
            'top1_hit': top1_hit,
            'top3_hit': top3_hit,
            'top5_hit': top5_hit,
            'exact_match': exact_match,
            'char_accuracy': char_accuracy,
            'syllable_accuracy': syllable_accuracy,
            'confidence': confidence
        }

    def run_test_set(self, test_cases: List, test_name: str) -> TestMetrics:
        """Run evaluation on a test set"""
        print(f"🧪 Running {test_name} tests ({len(test_cases)} cases)...")

        if not self.inference_engine:
            print("❌ Inference engine not available")
            return TestMetrics()

        results = []
        total_time = 0
        covered_cases = 0

        for i, test_case in enumerate(test_cases):
            if len(test_case) == 3:  # Basic format
                non_accented, target, context = test_case
                targets_list = None
            else:  # Challenging format with multiple targets
                non_accented, targets_list, context = test_case
                target = targets_list[0] if targets_list else ""

            # Measure inference time
            start_time = time.time()

            try:
                predictions = self.inference_engine.non_accented_to_words(
                    non_accented,
                    context=context,
                    max_suggestions=8
                )
                inference_time = (time.time() - start_time) * 1000  # ms
                total_time += inference_time
                covered_cases += 1

            except Exception as e:
                print(f"⚠️ Error in test case {i}: {e}")
                predictions = []
                inference_time = 0

            # Evaluate predictions
            eval_result = self.evaluate_predictions(
                predictions, target, targets_list)
            eval_result['inference_time'] = inference_time
            eval_result['test_case'] = {
                'non_accented': non_accented,
                'target': target,
                'context': context,
                'predictions': predictions[:5]  # Top 5 only
            }

            results.append(eval_result)

            # Progress report
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_cases)} cases...")

        # Calculate overall metrics
        if results:
            metrics = TestMetrics(
                top1_accuracy=sum(r['top1_hit']
                                  for r in results) / len(results),
                top3_accuracy=sum(r['top3_hit']
                                  for r in results) / len(results),
                top5_accuracy=sum(r['top5_hit']
                                  for r in results) / len(results),
                exact_match_accuracy=sum(r['exact_match']
                                         for r in results) / len(results),
                character_accuracy=sum(r['char_accuracy']
                                       for r in results) / len(results),
                syllable_accuracy=sum(r['syllable_accuracy']
                                      for r in results) / len(results),
                average_confidence=sum(r['confidence']
                                       for r in results) / len(results),
                inference_time_ms=total_time / len(results) if results else 0,
                coverage=covered_cases / len(test_cases)
            )
        else:
            metrics = TestMetrics()

        # Store detailed results
        self.test_results[test_name] = {
            'metrics': metrics.to_dict(),
            'detailed_results': results[:20],  # Store first 20 for analysis
            'summary': {
                'total_cases': len(test_cases),
                'covered_cases': covered_cases,
                'failed_cases': len(test_cases) - covered_cases
            }
        }

        print(f"✅ {test_name} completed:")
        print(f"   Top-1 Accuracy: {metrics.top1_accuracy:.3f}")
        print(f"   Top-3 Accuracy: {metrics.top3_accuracy:.3f}")
        print(f"   Character Accuracy: {metrics.character_accuracy:.3f}")
        print(f"   Avg Inference Time: {metrics.inference_time_ms:.2f}ms")

        return metrics

    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        print("🎯 Starting Full Evaluation Suite")
        print("=" * 60)

        # Run all test sets
        basic_metrics = self.run_test_set(self.basic_test_cases, "Basic Tests")
        challenging_metrics = self.run_test_set(
            self.challenging_test_cases, "Challenging Tests")
        context_metrics = self.run_test_set(
            self.context_test_cases, "Context Tests")

        # Run subset of Viet74K tests (due to size)
        if self.viet74k_test_cases:
            viet74k_subset = random.sample(self.viet74k_test_cases,
                                           min(500, len(self.viet74k_test_cases)))
            viet74k_metrics = self.run_test_set(
                viet74k_subset, "Viet74K Tests")
        else:
            viet74k_metrics = TestMetrics()

        # Aggregate metrics
        all_metrics = {
            'basic': basic_metrics.to_dict(),
            'challenging': challenging_metrics.to_dict(),
            'context': context_metrics.to_dict(),
            'viet74k': viet74k_metrics.to_dict()
        }

        # Overall weighted average
        weights = {'basic': 0.3, 'challenging': 0.3,
                   'context': 0.2, 'viet74k': 0.2}
        overall_metrics = {}

        for metric_name in ['top1_accuracy', 'top3_accuracy', 'character_accuracy']:
            weighted_sum = sum(all_metrics[test_type][metric_name] * weights[test_type]
                               for test_type in weights.keys())
            overall_metrics[metric_name] = weighted_sum

        self.test_results['overall'] = {
            'metrics': overall_metrics,
            'individual_test_types': all_metrics
        }

        return self.test_results

    def save_results(self, output_path: str = "test_results.json"):
        """Save test results to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False,
                          indent=2, default=str)
            print(f"📁 Test results saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def print_summary_report(self):
        """Print comprehensive summary report"""
        if 'overall' not in self.test_results:
            print("❌ No test results available")
            return

        print("\n" + "=" * 80)
        print("📊 VIETNAMESE GPT MODEL EVALUATION REPORT")
        print("=" * 80)

        overall = self.test_results['overall']['metrics']
        individual = self.test_results['overall']['individual_test_types']

        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Top-1 Accuracy: {overall['top1_accuracy']:.1%}")
        print(f"   Top-3 Accuracy: {overall['top3_accuracy']:.1%}")
        print(f"   Character Accuracy: {overall['character_accuracy']:.1%}")

        print(f"\n📈 BREAKDOWN BY TEST TYPE:")
        for test_type, metrics in individual.items():
            print(f"\n   {test_type.upper()}:")
            print(f"     Top-1: {metrics['top1_accuracy']:.1%}")
            print(f"     Top-3: {metrics['top3_accuracy']:.1%}")
            print(f"     Char Acc: {metrics['character_accuracy']:.1%}")
            print(f"     Avg Time: {metrics['inference_time_ms']:.1f}ms")

        print("\n" + "=" * 80)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Vietnamese GPT Model")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/vietnamese_non_accented_gpt_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="ml/data",
                        help="Data directory")
    parser.add_argument("--output", type=str, default="test_results.json",
                        help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with fewer cases")

    args = parser.parse_args()

    # Create test suite
    test_suite = VietnameseTestSuite(
        model_path=args.model_path,
        data_dir=args.data_dir
    )

    # Run evaluation
    if args.quick:
        print("🚀 Running Quick Evaluation...")
        # Run only basic tests for quick check
        test_suite.run_test_set(
            test_suite.basic_test_cases[:10], "Quick Basic Tests")
    else:
        print("🚀 Running Full Evaluation...")
        test_suite.run_full_evaluation()

    # Print report
    test_suite.print_summary_report()

    # Save results
    test_suite.save_results(args.output)

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
