#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇻🇳 VIETNAMESE WORD SEGMENTATION DEMO
=====================================

Dự án tách từ tiếng Việt tự động - Demo showcasing tất cả tính năng

Author: Vietnamese NLP Team
"""

import os
import sys
import time

# Add src to path for imports
sys.path.append('src')

from src.inference import CRFInference

def print_header(title):
    """In header đẹp cho demo"""
    print("\n" + "="*60)
    print(f"🇻🇳 {title}")
    print("="*60)

def print_section(title):
    """In section header"""
    print(f"\n📍 {title}")
    print("-"*50)

def interactive_demo():
    """Run interactive demo for Vietnamese word segmentation."""
    print("🇻🇳 Vietnamese Word Segmentation - Enhanced Interactive Demo")
    print("=" * 70)
    print("Enter Vietnamese text without spaces and diacritics")
    print("Examples: xinchao, sonha, toilasinhhvien, moibandenquannuocvietnam")
    print("Commands: 'multi' for multiple suggestions, 'quit' to exit")
    print("-" * 70)
    
    # Try to load enhanced model first, fallback to basic model
    enhanced_model_paths = [
        "models/crf_structure/best_model.pkl",
        "models/crf_enhanced/best_model.pkl", 
        "models/crf_full_enhanced/best_model.pkl"
    ]
    
    basic_model_paths = [
        "models/crf_large/best_model.pkl",
        "models/crf/best_model.pkl"
    ]
    
    model_path = None
    is_enhanced = False
    
    # Check for enhanced models first
    for path in enhanced_model_paths:
        if os.path.exists(path):
            model_path = path
            is_enhanced = True
            break
    
    # Fallback to basic models
    if not model_path:
        for path in basic_model_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path:
        print("❌ No trained model found!")
        print("🏋️ Train a model first:")
        print("   python train_large_corpus.py --structure-aware")
        print("   python train_large_corpus.py")
        return
    
    try:
        print(f"🔄 Loading model from {model_path}...")
        
        if is_enhanced:
            # Load enhanced model with Vietnamese dictionary
            sys.path.append('src')
            from src.models import ContextAwareCRFSegmenter, create_vietnamese_dictionary_from_data
            
            dict_files = ["data/train.txt", "data/Viet74K_clean.txt"]
            vietnamese_dict = create_vietnamese_dictionary_from_data(dict_files)
            
            model = ContextAwareCRFSegmenter(vietnamese_dict=vietnamese_dict)
            model.load(model_path)
            
            print(f"✅ Enhanced model loaded with {len(vietnamese_dict):,} dictionary words!")
            print("🧠 Features: Context-aware suggestions, meaningfulness scoring")
        else:
            # Load basic model
            inference = CRFInference(model_path)
            model = None
            
            model_info = inference.get_model_info()
            print(f"✅ Basic model loaded!")
            print(f"   Model Type: {model_info['model_type']}")
            if model_info['metadata']:
                print(f"   F1-Score: {model_info['metadata'].get('test_f1', 'N/A'):.4f}")
        
        print()
        
        # Interactive loop
        while True:
            try:
                # Get user input
                user_input = input("🔤 Enter text (or 'multi' for suggestions): ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("Please enter some text to segment")
                    continue
                
                # Check for multiple suggestions command
                if user_input.lower() == 'multi':
                    if not is_enhanced:
                        print("❌ Multiple suggestions require enhanced model")
                        print("🏋️ Train with: python train_large_corpus.py --structure-aware")
                        continue
                    
                    multi_input = input("🔤 Enter text for multiple suggestions: ").strip()
                    if not multi_input:
                        continue
                    
                    print(f"⏳ Generating suggestions for: '{multi_input}'...")
                    
                    if hasattr(model, 'segment_with_context'):
                        suggestions = model.segment_with_context(multi_input, n_best=5)
                        
                        print("🎯 Suggestions (ranked by meaningfulness):")
                        for i, (suggestion, score) in enumerate(suggestions, 1):
                            if score >= 0.8:
                                status = "🟢 High"
                            elif score >= 0.6:
                                status = "🟡 Medium"
                            else:
                                status = "🔴 Low"
                            
                            print(f"  {i}. {status}: '{suggestion}' ({score:.3f})")
                    else:
                        result = model.segment(multi_input)
                        print(f"✅ Result: '{result}'")
                    
                    print()
                    continue
                
                # Process regular input
                print(f"⏳ Processing: '{user_input}'...")
                
                if is_enhanced and model:
                    # Use enhanced model
                    if hasattr(model, 'segment_smart'):
                        result = model.segment_smart(user_input)
                        print(f"✅ Smart Result: '{result}'")
                    else:
                        result = model.segment(user_input)
                        print(f"✅ Result: '{result}'")
                    
                    # Show top suggestion with score if available
                    if hasattr(model, 'segment_with_context'):
                        suggestions = model.segment_with_context(user_input, n_best=1)
                        if suggestions:
                            _, score = suggestions[0]
                            print(f"📊 Confidence: {score:.3f}")
                else:
                    # Use basic model
                    result = inference.segment(user_input)
                    print(f"📤 Input:  '{result.input_text}'")
                    print(f"📥 Output: '{result.segmented_text}'")
                    print(f"⏱️  Time:   {result.processing_time:.4f} seconds")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
                
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")

def batch_demo():
    """Demonstrate batch processing capabilities."""
    print("\n🔄 Batch Processing Demo")
    print("-" * 40)
    
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Model not found for batch demo")
        return
    
    try:
        inference = CRFInference(model_path)
        
        # Sample Vietnamese texts
        sample_texts = [
            "xinchao",
            "toilasinhhvien",
            "moibandenquannuocvietnam", 
            "chungtoicunglamviec",
            "homnaylaicuoituan",
            "toisethamsinhvienvietnam",
            "hanchoivasinhvienuongnghehocbuoisang",
            "nguoivietnamratthichanucomvapho"
        ]
        
        print(f"Processing {len(sample_texts)} sample texts...")
        print()
        
        # Process batch
        results = inference.batch_segment(sample_texts)
        
        # Display results
        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"{i:2}. '{result.input_text}' → '{result.segmented_text}'")
        
    except Exception as e:
        print(f"❌ Batch demo failed: {e}")

def examples_demo():
    """Show various examples of word segmentation."""
    print("\n🎯 Example Demonstrations")
    print("-" * 35)
    
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Model not found for examples demo")
        return
    
    try:
        inference = CRFInference(model_path)
        
        # Categorized examples
        example_categories = {
            "Basic Greetings": [
                "xinchao",
                "chaoban",
                "tambietnhe"
            ],
            "Common Phrases": [
                "toilasinhhvien",
                "bantengi",
                "chungtoicunglamviec"
            ],
            "Longer Sentences": [
                "moibandenquannuocvietnam",
                "homnaytroisangdepqua",
                "toisethamsinhvienvietnam"
            ],
            "Complex Examples": [
                "hanchoivasinhvienuongnghehocbuoisang",
                "nguoivietnamratthichanucomvapho",
                "chungtoicanphaihoanthanhduannaytrongthoigianngan"
            ]
        }
        
        for category, examples in example_categories.items():
            print(f"\n{category}:")
            print("-" * len(category))
            
            for example in examples:
                result = inference.segment(example)
                print(f"  '{result.input_text}' → '{result.segmented_text}'")
        
    except Exception as e:
        print(f"❌ Examples demo failed: {e}")

def performance_demo():
    """Demonstrate model performance characteristics."""
    print("\n📊 Performance Demonstration")
    print("-" * 35)
    
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Model not found for performance demo")
        return
    
    try:
        inference = CRFInference(model_path)
        
        # Test different text lengths
        test_lengths = [
            ("Short (5 chars)", "xinch"),
            ("Medium (15 chars)", "toilasinhhvien"),
            ("Long (30 chars)", "moibandenquannuocvietnamchao"),
            ("Very Long (50+ chars)", "hanchoivasinhvienuongnghehocbuoisangvatoidentuquevietnam")
        ]
        
        print("Processing speed by text length:")
        print()
        
        for description, text in test_lengths:
            # Multiple runs for better timing
            times = []
            for _ in range(5):
                result = inference.segment(text)
                times.append(result.processing_time)
            
            avg_time = sum(times) / len(times)
            chars_per_sec = len(text) / avg_time if avg_time > 0 else float('inf')
            
            print(f"{description:15}: {avg_time:.4f}s avg ({chars_per_sec:.0f} chars/sec)")
            print(f"                 Result: '{result.segmented_text}'")
            print()
    
    except Exception as e:
        print(f"❌ Performance demo failed: {e}")

def main():
    """Main demo function with menu."""
    print("🇻🇳 Vietnamese Word Segmentation - CRF Demo")
    print("=" * 70)
    print("This demo showcases the CRF-based Vietnamese word segmentation system.")
    print()
    
    while True:
        print("Choose a demo option:")
        print("1. Interactive demo (type your own text)")
        print("2. Batch processing demo")  
        print("3. Example demonstrations")
        print("4. Performance demonstration")
        print("5. Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                interactive_demo()
            elif choice == '2':
                batch_demo()
            elif choice == '3':
                examples_demo()
            elif choice == '4':
                performance_demo()
            elif choice == '5':
                print("👋 Thank you for using the Vietnamese Word Segmentation demo!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
            if choice in ['1', '2', '3', '4']:
                input("\nPress Enter to return to menu...")
                print("\n" + "="*70)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 