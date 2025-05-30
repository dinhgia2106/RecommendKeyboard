#!/usr/bin/env python3
"""
Auto-Improvement System for Vietnamese Keyboard
Automatically apply AI-generated improvements và re-evaluate
"""

import json
import shutil
from typing import Dict, List
from datetime import datetime


class AutoImprovementSystem:
    """Hệ thống tự động cải tiến Vietnamese keyboard"""

    def __init__(self,
                 improvements_path: str = "ai_improvements.json",
                 processor_path: str = "ml/hybrid_vietnamese_processor.py"):
        self.improvements_path = improvements_path
        self.processor_path = processor_path

        # Load AI improvements
        self.improvements = self.load_improvements()

        # Backup original
        self.backup_original()

    def load_improvements(self) -> Dict:
        """Load AI-generated improvements"""
        try:
            with open(self.improvements_path, 'r', encoding='utf-8') as f:
                improvements = json.load(f)
            print(f"🔧 Loaded AI improvements")
            return improvements
        except Exception as e:
            print(f"❌ Error loading improvements: {e}")
            return {}

    def backup_original(self):
        """Backup original processor"""
        backup_path = f"{self.processor_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(self.processor_path, backup_path)
        print(f"💾 Backed up original processor to {backup_path}")

    def apply_vocabulary_improvements(self) -> int:
        """Apply vocabulary additions từ AI"""
        print("📚 Applying Vocabulary Improvements...")

        vocabulary_additions = self.improvements.get(
            'vocabulary_additions', [])

        if not vocabulary_additions:
            print("  No vocabulary improvements to apply")
            return 0

        # Read current processor
        with open(self.processor_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the core dictionaries section
        # High priority additions -> core_syllables
        # Medium priority -> core_compounds
        # Low priority -> core_sentences

        high_priority = [
            item for item in vocabulary_additions if item['priority'] == 'high']
        medium_priority = [
            item for item in vocabulary_additions if item['priority'] == 'medium']

        additions_made = 0

        # Add high priority single words to core_syllables
        if high_priority:
            syllable_additions = []
            for item in high_priority[:20]:  # Limit to top 20
                if item['category'] == 'single_word':
                    syllable_additions.append(
                        f"            '{item['input']}': '{item['expected']}',")
                    additions_made += 1

            if syllable_additions:
                # Find syllables section and add
                syllable_section_end = content.find("# Từ khác thường dùng")
                if syllable_section_end != -1:
                    insert_point = syllable_section_end
                    new_content = (content[:insert_point] +
                                   "\n            # AI-Generated additions (high priority)\n" +
                                   "\n".join(syllable_additions) + "\n\n            " +
                                   content[insert_point:])
                    content = new_content

        # Add medium priority to core_compounds
        if medium_priority:
            compound_additions = []
            for item in medium_priority[:15]:  # Limit to top 15
                if item['category'] == 'compound_word':
                    compound_additions.append(
                        f"            '{item['input']}': '{item['expected']}',")
                    additions_made += 1

            if compound_additions:
                # Find compounds section and add
                compound_section_end = content.find(
                    "# Giao tiếp cơ bản - THÊM MỚI")
                if compound_section_end != -1:
                    insert_point = compound_section_end
                    new_content = (content[:insert_point] +
                                   "# AI-Generated compounds (medium priority)\n            " +
                                   "\n            ".join(compound_additions) + "\n\n            " +
                                   content[insert_point:])
                    content = new_content

        # Write updated content
        if additions_made > 0:
            with open(self.processor_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Added {additions_made} vocabulary improvements")

        return additions_made

    def apply_segmentation_improvements(self) -> int:
        """Apply segmentation rule improvements"""
        print("🧠 Applying Segmentation Improvements...")

        segmentation_rules = self.improvements.get('segmentation_rules', [])

        if not segmentation_rules:
            print("  No segmentation improvements to apply")
            return 0

        # Read current processor
        with open(self.processor_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add smart segmentation rules
        # Find the _hybrid_segmentation method
        method_start = content.find(
            "def _hybrid_segmentation(self, input_text: str)")

        if method_start == -1:
            print("  ❌ Could not find segmentation method")
            return 0

        # Add AI-learned patterns before existing logic
        ai_rules = []
        for rule in segmentation_rules[:10]:  # Top 10 rules
            if rule['confidence'] > 0.7:
                pattern = rule['pattern']
                segments = rule['optimal_segments']

                ai_rules.append(f"""
        # AI-Learned pattern: {pattern}
        if len(input_text) == {sum(segments)} and len(segments) == {len(segments)}:
            # Apply learned segmentation pattern
            segments = {segments}
            pos = 0
            ai_result = []
            for seg_len in segments:
                if pos + seg_len <= len(input_text):
                    substring = input_text[pos:pos + seg_len]
                    vietnamese_word = self._lookup_word(substring)
                    ai_result.append(vietnamese_word)
                    pos += seg_len
            
            if len(ai_result) == len(segments):
                vietnamese_text = ' '.join(ai_result)
                confidence = {int(rule['confidence'] * 100)}
                return {{
                    'vietnamese_text': vietnamese_text,
                    'confidence': confidence,
                    'method': 'ai_learned_segmentation'
                }}
""")

        if ai_rules:
            # Insert AI rules at the beginning of the method
            method_end = content.find("result = []", method_start)
            if method_end != -1:
                new_content = (content[:method_end] +
                               "".join(ai_rules) + "\n        " +
                               content[method_end:])

                with open(self.processor_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"  ✅ Added {len(ai_rules)} segmentation rules")
                return len(ai_rules)

        return 0

    def run_full_improvement_cycle(self) -> Dict:
        """Run complete improvement cycle"""
        print("🚀 Running Full Auto-Improvement Cycle...")

        # Apply improvements
        vocab_additions = self.apply_vocabulary_improvements()
        segmentation_additions = self.apply_segmentation_improvements()

        # Re-evaluate system
        print("\n🧪 Re-evaluating Improved System...")
        from ml.systematic_evaluator import SystematicVietnameseEvaluator
        from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor

        # Reload improved processor
        import importlib
        import ml.hybrid_vietnamese_processor as processor_module
        importlib.reload(processor_module)

        processor = processor_module.HybridVietnameseProcessor()
        evaluator = SystematicVietnameseEvaluator()

        # Quick evaluation (sample of test data)
        test_sample = list(evaluator.test_data.items())[:1000]  # 1000 sample

        sample_results = {
            'total_tests': len(test_sample),
            'correct_predictions': 0,
            'partially_correct': 0,
            'incorrect_predictions': 0,
            'no_predictions': 0,
            'processing_times': []
        }

        import time
        for input_key, expected_data in test_sample:
            expected_vietnamese = expected_data['vietnamese']

            start_time = time.time()
            predictions = processor.process_text(input_key, max_suggestions=3)
            processing_time = time.time() - start_time
            sample_results['processing_times'].append(processing_time)

            if not predictions:
                sample_results['no_predictions'] += 1
                continue

            # Check if any prediction matches
            exact_match = any(pred['vietnamese_text'] ==
                              expected_vietnamese for pred in predictions)
            partial_match = any(expected_vietnamese in pred['vietnamese_text'] or
                                pred['vietnamese_text'] in expected_vietnamese for pred in predictions)

            if exact_match:
                sample_results['correct_predictions'] += 1
            elif partial_match:
                sample_results['partially_correct'] += 1
            else:
                sample_results['incorrect_predictions'] += 1

        # Calculate improvement metrics
        total = sample_results['total_tests']
        new_accuracy = sample_results['correct_predictions'] / total * 100
        new_partial_accuracy = (
            sample_results['correct_predictions'] + sample_results['partially_correct']) / total * 100
        avg_time = sum(sample_results['processing_times']) / \
            len(sample_results['processing_times']) * 1000

        # Compare với original (96.22% accuracy)
        original_accuracy = 96.22
        accuracy_improvement = new_accuracy - original_accuracy

        improvement_results = {
            'improvements_applied': {
                'vocabulary_additions': vocab_additions,
                'segmentation_rules': segmentation_additions
            },
            'performance_metrics': {
                'original_accuracy': original_accuracy,
                'new_accuracy': new_accuracy,
                'accuracy_improvement': accuracy_improvement,
                'new_partial_accuracy': new_partial_accuracy,
                'avg_processing_time_ms': avg_time
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }

        # Save improvement results
        with open('auto_improvement_results.json', 'w', encoding='utf-8') as f:
            json.dump(improvement_results, f, ensure_ascii=False, indent=2)

        print(f"\n📊 AUTO-IMPROVEMENT RESULTS:")
        print(f"  Vocabulary Additions: {vocab_additions}")
        print(f"  Segmentation Rules: {segmentation_additions}")
        print(f"  Original Accuracy: {original_accuracy:.2f}%")
        print(f"  New Accuracy: {new_accuracy:.2f}%")
        print(f"  Improvement: {accuracy_improvement:+.2f}%")
        print(f"  Processing Time: {avg_time:.2f}ms")

        return improvement_results


def main():
    """Main auto-improvement function"""
    auto_improver = AutoImprovementSystem()
    results = auto_improver.run_full_improvement_cycle()
    return auto_improver, results


if __name__ == "__main__":
    main()
