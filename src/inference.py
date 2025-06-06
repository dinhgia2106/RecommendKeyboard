"""
CRF-based Vietnamese Word Segmentation Inference

This module provides inference capabilities for the trained CRF model.
It handles model loading, text preprocessing, and batch processing for
Vietnamese word segmentation tasks.

Main features:
- Single text segmentation
- Batch processing
- Model performance monitoring
- Input validation and preprocessing

Example usage:
    inference = CRFInference('models/crf/best_model.pkl')
    result = inference.segment("xinchao")  # Returns "xin chao"
"""

import os
import json
import pickle
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .models import CRFSegmenter


@dataclass
class SegmentationResult:
    """
    Result object for word segmentation output.
    
    Attributes:
        input_text: Original unsegmented text
        segmented_text: Text with word boundaries (best result)
        processing_time: Time taken for segmentation (seconds)
        confidence_score: Optional confidence measure for best result
        candidates: List of (segmented_text, confidence) for multiple suggestions
    """
    input_text: str
    segmented_text: str
    processing_time: float
    confidence_score: float = None
    candidates: List[Tuple[str, float]] = None


class CRFInference:
    """
    CRF-based inference engine for Vietnamese word segmentation.
    
    This class provides a high-level interface for using trained CRF models
    to segment Vietnamese text. It handles model loading, preprocessing,
    and provides both single and batch inference capabilities.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained CRF model (.pkl file)
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load the trained CRF model and its metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"🔄 Loading CRF model from {self.model_path}...")
        
        # Load model
        self.model = CRFSegmenter()
        self.model.load(self.model_path)
        
        # Load metadata if available
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                print(f"📋 Loaded model metadata: F1={self.metadata.get('test_f1', 'N/A'):.4f}")
        
        print("✅ CRF model loaded successfully!")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for segmentation.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for segmentation
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove existing spaces (for consistency)
        text = text.replace(' ', '')
        
        # Convert to lowercase for better model performance
        text = text.lower()
        
        # Remove special characters but keep Vietnamese characters
        # This is a simple version - could be enhanced based on requirements
        cleaned_chars = []
        for char in text:
            if char.isalnum() or ord(char) > 127:  # Keep alphanumeric and Vietnamese chars
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)
    
    def segment(self, text: str) -> SegmentationResult:
        """
        Segment a single text input.
        
        Args:
            text: Input text to segment
            
        Returns:
            SegmentationResult with segmented text and metadata
        """
        if not self.model:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        if not text:
            return SegmentationResult("", "", 0.0)
        
        # Preprocess
        original_text = text
        preprocessed_text = self.preprocess_text(text)
        
        if not preprocessed_text:
            return SegmentationResult(original_text, "", 0.0)
        
        # Segment with timing
        start_time = time.time()
        segmented_text = self.model.segment(preprocessed_text)
        processing_time = time.time() - start_time
        
        return SegmentationResult(
            input_text=original_text,
            segmented_text=segmented_text,
            processing_time=processing_time
        )
    
    def segment_multiple(self, text: str, n_suggestions: int = 5) -> SegmentationResult:
        """
        Segment a single text input with multiple suggestions.
        
        Args:
            text: Input text to segment
            n_suggestions: Number of segmentation suggestions to generate
            
        Returns:
            SegmentationResult with multiple candidates
        """
        if not self.model:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        if not text:
            return SegmentationResult("", "", 0.0, candidates=[("", 1.0)])
        
        # Preprocess
        original_text = text
        preprocessed_text = self.preprocess_text(text)
        
        if not preprocessed_text:
            return SegmentationResult(original_text, "", 0.0, candidates=[("", 1.0)])
        
        # Segment with timing
        start_time = time.time()
        candidates = self.model.segment_multiple(preprocessed_text, n_suggestions)
        processing_time = time.time() - start_time
        
        # Filter out duplicate suggestions, keeping the one with the highest confidence
        unique_candidates = {}
        for segmented_text, confidence in candidates:
            if segmented_text not in unique_candidates or confidence > unique_candidates[segmented_text]:
                unique_candidates[segmented_text] = confidence
        
        # Sort by confidence score in descending order
        sorted_candidates = sorted(unique_candidates.items(), key=lambda item: item[1], reverse=True)
        
        # Assign filtered candidates
        candidates = sorted_candidates
        
        # Best result is the first one (highest confidence)
        best_result = candidates[0][0] if candidates else ""
        best_confidence = candidates[0][1] if candidates else 0.0
        
        return SegmentationResult(
            input_text=original_text,
            segmented_text=best_result,
            processing_time=processing_time,
            confidence_score=best_confidence,
            candidates=candidates
        )
    
    def batch_segment(self, texts: List[str]) -> List[SegmentationResult]:
        """
        Segment multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of SegmentationResult objects
        """
        if not self.model:
            raise ValueError("Model not loaded!")
        
        results = []
        total_start_time = time.time()
        
        print(f"🔄 Processing {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            try:
                result = self.segment(text)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - total_start_time
                    avg_time = elapsed / (i + 1)
                    print(f"📊 Processed {i+1}/{len(texts)} texts (avg: {avg_time:.4f}s/text)")
                    
            except Exception as e:
                print(f"❌ Error processing text {i}: {e}")
                # Add empty result for failed processing
                results.append(SegmentationResult(text, "", 0.0))
        
        total_time = time.time() - total_start_time
        print(f"✅ Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_path': self.model_path,
            'model_type': 'CRF',
            'is_loaded': self.model is not None,
            'metadata': self.metadata
        }
        
        if self.model and hasattr(self.model, 'feature_extractor'):
            feature_extractor = self.model.feature_extractor
            info['feature_info'] = {
                'has_dictionary': feature_extractor.dictionary is not None,
                'dictionary_size': len(feature_extractor.dictionary) if feature_extractor.dictionary else 0
            }
        
        return info
    
    def demo_examples(self) -> List[SegmentationResult]:
        """
        Run demo segmentation on example texts.
        
        Returns:
            List of demo results
        """
        demo_texts = [
            "xinchao",
            "toilasinhhvien", 
            "moibandenquannuocvietnam",
            "chungtoicunglamviec",
            "homnaylaicuoituan",
            "toisethamsinhvienvietnam"
        ]
        
        print("🎯 Running demo examples...")
        results = []
        
        for text in demo_texts:
            result = self.segment(text)
            results.append(result)
            print(f"   '{result.input_text}' → '{result.segmented_text}' ({result.processing_time:.4f}s)")
        
        return results


class PerformanceMonitor:
    """
    Monitor performance metrics for the inference system.
    
    This class tracks various metrics including processing speed,
    accuracy (if ground truth available), and system performance.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0,
            'texts_per_second': 0.0,
            'error_count': 0
        }
    
    def update(self, results: List[SegmentationResult]):
        """
        Update statistics with new results.
        
        Args:
            results: List of segmentation results
        """
        for result in results:
            self.stats['total_processed'] += 1
            self.stats['total_time'] += result.processing_time
            
            if not result.segmented_text:  # Consider empty results as errors
                self.stats['error_count'] += 1
        
        # Update derived statistics
        if self.stats['total_processed'] > 0:
            self.stats['avg_processing_time'] = self.stats['total_time'] / self.stats['total_processed']
            self.stats['texts_per_second'] = self.stats['total_processed'] / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        return self.stats.copy()
    
    def print_stats(self):
        """Print formatted performance statistics."""
        print("\n📊 PERFORMANCE STATISTICS")
        print("=" * 40)
        print(f"Total texts processed: {self.stats['total_processed']}")
        print(f"Total processing time: {self.stats['total_time']:.2f}s")
        print(f"Average time per text: {self.stats['avg_processing_time']:.4f}s")
        print(f"Processing speed: {self.stats['texts_per_second']:.1f} texts/second")
        print(f"Error rate: {self.stats['error_count']}/{self.stats['total_processed']} ({100*self.stats['error_count']/max(1, self.stats['total_processed']):.1f}%)")


def main():
    """
    Main inference demonstration script.
    
    This script shows how to use the CRF inference system for
    Vietnamese word segmentation with various examples.
    """
    print("🇻🇳 Vietnamese Word Segmentation - CRF Inference Demo")
    print("=" * 60)
    
    # Initialize inference system
    model_path = "models/crf/best_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please run training first: python -m src.training")
        return
    
    try:
        # Load model
        inference = CRFInference(model_path)
        
        # Show model info
        model_info = inference.get_model_info()
        print("\n📋 Model Information:")
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Dictionary Size: {model_info.get('feature_info', {}).get('dictionary_size', 'N/A')}")
        if model_info['metadata']:
            print(f"   Test F1-score: {model_info['metadata'].get('test_f1', 'N/A'):.4f}")
        
        # Run demo examples
        print("\n🎯 Demo Examples:")
        print("-" * 30)
        demo_results = inference.demo_examples()
        
        # Performance monitoring
        monitor = PerformanceMonitor()
        monitor.update(demo_results)
        monitor.print_stats()
        
        # Interactive mode
        print("\n🔧 Interactive Mode (type 'quit' to exit):")
        print("💡 Use 'multi:<text>' for multiple suggestions")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("Enter text to segment: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Check if user wants multiple suggestions
                if user_input.startswith('multi:'):
                    text_to_segment = user_input[6:].strip()
                    if text_to_segment:
                        result = inference.segment_multiple(text_to_segment, n_suggestions=5)
                        print(f"📍 Multiple suggestions for '{result.input_text}':")
                        for i, (candidate, confidence) in enumerate(result.candidates, 1):
                            print(f"   {i}. '{candidate}' (confidence: {confidence:.3f})")
                        print(f"Processing time: {result.processing_time:.4f}s")
                        print()
                else:
                    result = inference.segment(user_input)
                    print(f"Result: '{result.segmented_text}' ({result.processing_time:.4f}s)")
                    print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Failed to initialize inference system: {e}")


if __name__ == "__main__":
    main() 