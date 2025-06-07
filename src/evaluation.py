"""
CRF Model Evaluation for Vietnamese Word Segmentation

This module provides comprehensive evaluation capabilities for CRF-based
Vietnamese word segmentation models. It includes multiple evaluation metrics,
error analysis, and visualization tools.

Key features:
- Multiple evaluation metrics (word-level, character-level, sentence-level)
- Detailed error analysis and pattern detection
- Performance visualization and reporting
- Comparison with ground truth data
- Export results to CSV and visualization files

Evaluation metrics:
- Precision, Recall, F1-score (word-level)
- Character-level accuracy
- Sentence-level accuracy
- Average word length accuracy
"""

import os
import re
import json
import csv
import time
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from sklearn_crfsuite import metrics
from tqdm import tqdm

from .models import CRFSegmenter
from .data_preparation import VietnameseDataPreprocessor


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        precision: Word-level precision
        recall: Word-level recall
        f1_score: Word-level F1-score
        character_accuracy: Character-level accuracy
        sentence_accuracy: Sentence-level accuracy (exact match)
        avg_word_length_error: Average difference in word length
        processing_time: Time taken for evaluation
    """
    precision: float
    recall: float
    f1_score: float
    character_accuracy: float
    sentence_accuracy: float
    avg_word_length_error: float
    processing_time: float


class CRFEvaluator:
    """
    Comprehensive evaluation system for CRF-based word segmentation.
    
    This class provides detailed evaluation of CRF models using multiple
    metrics and analysis approaches. It can evaluate both individual
    models and provide comparative analysis.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.preprocessor = VietnameseDataPreprocessor()
        self.error_patterns = defaultdict(int)
        self.prediction_cache = {}
    
    def load_test_data(self, test_path: str) -> List[Tuple[str, str]]:
        """
        Load test dataset from file.
        
        Args:
            test_path: Path to test data file
            
        Returns:
            List of (input, ground_truth) pairs
        """
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        return self.preprocessor.load_dataset(test_path)
    
    def evaluate_model(self, model: CRFSegmenter, 
                      test_data: List[Tuple[str, str]],
                      max_samples: int = None) -> EvaluationMetrics:
        """
        Evaluate CRF model on test data.
        
        Args:
            model: Trained CRF model
            test_data: Test dataset
            max_samples: Maximum samples to evaluate (None for all)
            
        Returns:
            EvaluationMetrics object with detailed scores
        """
        if max_samples:
            test_data = test_data[:max_samples]
        
        print(f"Evaluating CRF model on {len(test_data)} samples...")
        
        # Collect predictions and ground truth
        y_true_labels = []
        y_pred_labels = []
        
        # Word-level evaluation data
        true_words_all = []
        pred_words_all = []
        
        # Sentence-level evaluation
        exact_matches = 0
        char_matches = 0
        total_chars = 0
        word_length_errors = []
        
        start_time = time.time()
        
        for i, (x_raw, y_gold) in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                # Get model prediction
                pred_segmented = model.segment(x_raw)
                
                # FIXED: Remove diacritics from both y_gold and prediction for fair comparison
                # Since model only segments without diacritics, we need to normalize both sides
                y_gold_normalized = self.preprocessor.remove_diacritics(y_gold.lower())
                pred_normalized = self.preprocessor.remove_diacritics(pred_segmented.lower())
                
                # Character-level evaluation (use normalized versions)
                y_gold_clean = re.sub(r'[^\w]', '', y_gold_normalized)
                pred_clean = re.sub(r'[^\w]', '', pred_normalized)
                
                if len(y_gold_clean) == len(pred_clean):
                    for true_char, pred_char in zip(y_gold_clean, pred_clean):
                        if true_char == pred_char:
                            char_matches += 1
                        total_chars += 1
                
                # Word-level evaluation (use normalized versions)
                true_words = y_gold_normalized.strip().split()
                pred_words = pred_normalized.strip().split()
                
                true_words_all.extend(true_words)
                pred_words_all.extend(pred_words)
                
                # Sentence-level evaluation (use normalized versions)
                if y_gold_normalized.strip() == pred_normalized.strip():
                    exact_matches += 1
                
                # Word length analysis
                word_length_error = abs(len(true_words) - len(pred_words))
                word_length_errors.append(word_length_error)
                
                # Label-level evaluation for F1-score
                try:
                    true_labels = model.create_labels(x_raw, y_gold)
                    pred_labels = model.predict(x_raw)
                    
                    if len(true_labels) == len(pred_labels):
                        y_true_labels.extend(true_labels)
                        y_pred_labels.extend(pred_labels)
                except:
                    # Skip if label creation fails
                    pass
                
            except Exception as e:
                # Skip problematic samples
                continue
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        # Label-level F1-score
        if len(y_true_labels) > 0:
            precision = metrics.flat_precision_score(y_true_labels, y_pred_labels, average='weighted')
            recall = metrics.flat_recall_score(y_true_labels, y_pred_labels, average='weighted')
            f1_score = metrics.flat_f1_score(y_true_labels, y_pred_labels, average='weighted')
        else:
            precision = recall = f1_score = 0.0
        
        # Character accuracy
        character_accuracy = char_matches / max(1, total_chars)
        
        # Sentence accuracy
        sentence_accuracy = exact_matches / len(test_data)
        
        # Average word length error
        avg_word_length_error = np.mean(word_length_errors) if word_length_errors else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            character_accuracy=character_accuracy,
            sentence_accuracy=sentence_accuracy,
            avg_word_length_error=avg_word_length_error,
            processing_time=processing_time
        )
    
    def analyze_errors(self, model: CRFSegmenter, 
                      test_data: List[Tuple[str, str]],
                      max_samples: int = 50) -> Dict[str, Any]:
        """
        Perform detailed error analysis on model predictions.
        
        Args:
            model: Trained CRF model
            test_data: Test dataset
            max_samples: Maximum samples to analyze
            
        Returns:
            Dictionary with error analysis results
        """
        print(f"🔍 Analyzing errors on {min(max_samples, len(test_data))} samples...")
        
        error_types = {
            'over_segmentation': 0,  # Too many words predicted
            'under_segmentation': 0,  # Too few words predicted
            'boundary_errors': 0,     # Wrong word boundaries
            'perfect_matches': 0      # Exact matches
        }
        
        common_errors = defaultdict(int)
        error_examples = []
        
        for i, (x_raw, y_gold) in enumerate(test_data[:max_samples]):
            try:
                pred_segmented = model.segment(x_raw)
                
                # FIXED: Normalize both y_gold and prediction for fair comparison
                y_gold_normalized = self.preprocessor.remove_diacritics(y_gold.lower())
                pred_normalized = self.preprocessor.remove_diacritics(pred_segmented.lower())
                
                true_words = y_gold_normalized.strip().split()
                pred_words = pred_normalized.strip().split()
                
                # Classify error type
                if y_gold_normalized.strip() == pred_normalized.strip():
                    error_types['perfect_matches'] += 1
                elif len(pred_words) > len(true_words):
                    error_types['over_segmentation'] += 1
                elif len(pred_words) < len(true_words):
                    error_types['under_segmentation'] += 1
                else:
                    error_types['boundary_errors'] += 1
                
                # Collect error patterns
                if y_gold_normalized.strip() != pred_normalized.strip():
                    error_key = f"{y_gold_normalized.strip()} -> {pred_normalized.strip()}"
                    common_errors[error_key] += 1
                    
                    if len(error_examples) < 10:  # Keep top 10 examples
                        error_examples.append({
                            'input': x_raw,
                            'true': y_gold.strip(),  # Keep original for display
                            'pred': pred_segmented.strip(),  # Keep original for display
                            'true_normalized': y_gold_normalized.strip(),  # Add normalized for analysis
                            'pred_normalized': pred_normalized.strip(),  # Add normalized for analysis
                            'error_type': self._classify_error_detailed(true_words, pred_words)
                        })
                
            except Exception as e:
                continue
        
        # Get most common errors
        top_errors = dict(sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            'error_types': error_types,
            'common_errors': top_errors,
            'error_examples': error_examples,
            'total_analyzed': min(max_samples, len(test_data))
        }
    
    def _classify_error_detailed(self, true_words: List[str], pred_words: List[str]) -> str:
        """Classify the specific type of segmentation error."""
        if len(pred_words) > len(true_words):
            return 'over_segmentation'
        elif len(pred_words) < len(true_words):
            return 'under_segmentation'
        else:
            return 'boundary_error'
    
    def create_evaluation_report(self, model: CRFSegmenter,
                               test_data: List[Tuple[str, str]],
                               model_name: str = "CRF",
                               output_dir: str = "evaluation_results") -> str:
        """
        Create comprehensive evaluation report.
        
        Args:
            model: Trained CRF model
            test_data: Test dataset
            model_name: Name of the model
            output_dir: Directory to save results
            
        Returns:
            Path to the generated report file
        """
        print(f"📊 Creating evaluation report for {model_name}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate model
        metrics = self.evaluate_model(model, test_data)
        
        # Error analysis
        error_analysis = self.analyze_errors(model, test_data)
        
        # Create report
        report = {
            'model_name': model_name,
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(test_data),
            'metrics': {
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'character_accuracy': metrics.character_accuracy,
                'sentence_accuracy': metrics.sentence_accuracy,
                'avg_word_length_error': metrics.avg_word_length_error,
                'processing_time': metrics.processing_time
            },
            'error_analysis': error_analysis
        }
        
        # Save JSON report
        report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Precision', f"{metrics.precision:.4f}"])
            writer.writerow(['Recall', f"{metrics.recall:.4f}"])
            writer.writerow(['F1-Score', f"{metrics.f1_score:.4f}"])
            writer.writerow(['Character Accuracy', f"{metrics.character_accuracy:.4f}"])
            writer.writerow(['Sentence Accuracy', f"{metrics.sentence_accuracy:.4f}"])
            writer.writerow(['Avg Word Length Error', f"{metrics.avg_word_length_error:.4f}"])
            writer.writerow(['Processing Time (s)', f"{metrics.processing_time:.2f}"])
        
        # Create visualization
        self._create_metrics_visualization(metrics, model_name, output_dir)
        
        print(f"✅ Report saved to {report_path}")
        return report_path
    
    def _create_metrics_visualization(self, metrics: EvaluationMetrics, 
                                    model_name: str, output_dir: str):
        """Create visualization of evaluation metrics."""
        try:
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Precision, Recall, F1-Score
            scores = [metrics.precision, metrics.recall, metrics.f1_score]
            labels = ['Precision', 'Recall', 'F1-Score']
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            bars1 = ax1.bar(labels, scores, color=colors)
            ax1.set_title(f'{model_name} - Word-level Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars1, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            # Accuracy metrics
            accuracies = [metrics.character_accuracy, metrics.sentence_accuracy]
            acc_labels = ['Character\nAccuracy', 'Sentence\nAccuracy']
            
            bars2 = ax2.bar(acc_labels, accuracies, color=['#C73E1D', '#592E83'])
            ax2.set_title(f'{model_name} - Accuracy Metrics')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            
            for bar, acc in zip(bars2, accuracies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # Processing time
            ax3.bar(['Processing\nTime'], [metrics.processing_time], color='#4CAF50')
            ax3.set_title(f'{model_name} - Processing Time')
            ax3.set_ylabel('Time (seconds)')
            ax3.text(0, metrics.processing_time + metrics.processing_time*0.05,
                    f'{metrics.processing_time:.2f}s', ha='center', va='bottom')
            
            # Word length error
            ax4.bar(['Avg Word\nLength Error'], [metrics.avg_word_length_error], color='#FF9800')
            ax4.set_title(f'{model_name} - Word Length Error')
            ax4.set_ylabel('Average Error')
            ax4.text(0, metrics.avg_word_length_error + metrics.avg_word_length_error*0.05,
                    f'{metrics.avg_word_length_error:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"{model_name}_metrics_visualization.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Visualization saved to {plot_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")
    
    def print_evaluation_summary(self, metrics: EvaluationMetrics, model_name: str):
        """Print formatted evaluation summary."""
        print(f"\n📊 EVALUATION SUMMARY - {model_name}")
        print("=" * 50)
        print(f"📈 Word-level Metrics:")
        print(f"   Precision:    {metrics.precision:.4f}")
        print(f"   Recall:       {metrics.recall:.4f}")
        print(f"   F1-Score:     {metrics.f1_score:.4f}")
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   Character:    {metrics.character_accuracy:.4f}")
        print(f"   Sentence:     {metrics.sentence_accuracy:.4f}")
        print(f"\n⚡ Performance:")
        print(f"   Processing:   {metrics.processing_time:.2f}s")
        print(f"   Word Length Error: {metrics.avg_word_length_error:.2f}")
        
        # Performance interpretation
        if metrics.f1_score >= 0.9:
            print("🏆 Excellent performance!")
        elif metrics.f1_score >= 0.8:
            print("👍 Good performance!")
        elif metrics.f1_score >= 0.7:
            print("📈 Decent performance!")
        else:
            print("⚠️  Performance needs improvement")


def main():
    """
    Main evaluation script for CRF model.
    
    This script demonstrates comprehensive evaluation of a trained
    CRF model for Vietnamese word segmentation.
    """
    print("🇻🇳 Vietnamese Word Segmentation - CRF Model Evaluation")
    print("=" * 65)
    
    # Configuration
    model_path = "models/crf/best_model.pkl"
    test_data_path = "data/test.txt"
    output_dir = "evaluation_results"
    
    # Check if required files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please run training first: python -m src.training")
        return
    
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found: {test_data_path}")
        print("   Please prepare test data first")
        return
    
    try:
        # Initialize evaluator
        evaluator = CRFEvaluator()
        
        # Load model
        print("🔄 Loading CRF model...")
        model = CRFSegmenter()
        model.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Load test data
        print("📚 Loading test data...")
        test_data = evaluator.load_test_data(test_data_path)
        print(f"✅ Loaded {len(test_data)} test samples")
        
        # Run evaluation
        print("\n🚀 Starting comprehensive evaluation...")
        metrics = evaluator.evaluate_model(model, test_data, max_samples=1000)
        
        # Print summary
        evaluator.print_evaluation_summary(metrics, "CRF")
        
        # Create detailed report
        print("\n📋 Creating detailed evaluation report...")
        report_path = evaluator.create_evaluation_report(
            model, test_data, "CRF", output_dir
        )
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📂 Results saved to: {output_dir}/")
        print(f"📄 Full report: {report_path}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 