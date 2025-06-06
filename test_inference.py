"""
Test script for CRF-based Vietnamese Word Segmentation

This script tests the CRF inference capabilities and demonstrates
various usage patterns for the Vietnamese word segmentation system.
"""

import os
import sys
import time
from typing import List

# Add src to path for imports
sys.path.append('src')

from src.inference import CRFInference, PerformanceMonitor


def test_crf_model():
    """Test the CRF model with various inputs."""
    print("🧪 Testing CRF Model for Vietnamese Word Segmentation")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please run training first: python -m src.training")
        return False
    
    try:
        # Initialize inference
        print("🔄 Loading CRF model...")
        inference = CRFInference(model_path)
        
        # Test cases
        test_cases = [
            # Basic cases
            "xinchao",
            "toilasinhhvien",
            "moibandenquannuocvietnam",
            
            # Compound words
            "chungtoicunglamviec",
            "homnaylaicuoituan",
            "toisethamsinhvienvietnam",
            
            # Longer sequences
            "hanchoivasinhvienuongnghehocbuoisang",
            "nguoivietnamratthichanucomvapho",
            
            # Edge cases
            "a",           # Single character
            "ab",          # Two characters
            "",            # Empty string
            "123",         # Numbers
            "abc123def"    # Mixed alphanumeric
        ]
        
        print("\n🎯 Test Results:")
        print("-" * 50)
        
        results = []
        for i, test_input in enumerate(test_cases, 1):
            result = inference.segment(test_input)
            results.append(result)
            
            status = "✅" if result.segmented_text else "⚠️ "
            print(f"{i:2}. {status} '{result.input_text}' → '{result.segmented_text}' ({result.processing_time:.4f}s)")
        
        # Performance statistics
        monitor = PerformanceMonitor()
        monitor.update(results)
        monitor.print_stats()
        
        # Model information
        print("\n📋 Model Information:")
        print("-" * 30)
        model_info = inference.get_model_info()
        print(f"Model Type: {model_info['model_type']}")
        print(f"Dictionary Size: {model_info.get('feature_info', {}).get('dictionary_size', 'N/A')}")
        if model_info['metadata']:
            print(f"Test F1-Score: {model_info['metadata'].get('test_f1', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n🔄 Testing Batch Processing")
    print("-" * 40)
    
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Model not found for batch testing")
        return False
    
    try:
        inference = CRFInference(model_path)
        
        # Batch test cases
        batch_texts = [
            "xinchao",
            "toilasinhhvien",
            "moibandenquannuocvietnam",
            "chungtoicunglamviec",
            "homnaylaicuoituan"
        ]
        
        print(f"Processing {len(batch_texts)} texts in batch...")
        start_time = time.time()
        results = inference.batch_segment(batch_texts)
        total_time = time.time() - start_time
        
        print("\nBatch Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. '{result.input_text}' → '{result.segmented_text}'")
        
        print(f"\nBatch Performance:")
        print(f"Total time: {total_time:.4f}s")
        print(f"Average per text: {total_time/len(batch_texts):.4f}s")
        print(f"Throughput: {len(batch_texts)/total_time:.1f} texts/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔍 Testing Edge Cases")
    print("-" * 30)
    
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Model not found for edge case testing")
        return False
    
    try:
        inference = CRFInference(model_path)
        
        edge_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("a", "Single character"),
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Very long single word"),
            ("123456789", "Numbers only"),
            ("!@#$%^&*()", "Special characters"),
            ("AaAaAaAa", "Mixed case"),
            ("vietnamvietnamvietnamvietnamvietnam", "Repetitive text")
        ]
        
        print("Edge Case Results:")
        for test_input, description in edge_cases:
            try:
                result = inference.segment(test_input)
                status = "✅" if result.segmented_text is not None else "❌"
                print(f"{status} {description}: '{test_input}' → '{result.segmented_text}'")
            except Exception as e:
                print(f"❌ {description}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False


def main():
    """Main test function."""
    print("🇻🇳 Vietnamese Word Segmentation - CRF Model Tests")
    print("=" * 70)
    
    # Run tests
    tests = [
        ("Basic CRF Model Test", test_crf_model),
        ("Batch Processing Test", test_batch_processing),
        ("Edge Cases Test", test_edge_cases)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print(f"{'='*70}")
        
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {100*passed/total:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 