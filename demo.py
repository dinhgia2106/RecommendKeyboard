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
    print("🇻🇳 Vietnamese Word Segmentation - Interactive Demo")
    print("=" * 60)
    print("Enter Vietnamese text without spaces and diacritics")
    print("Examples: xinchao, toilasinhhvien, moibandenquannuocvietnam")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 60)
    
    # Load model
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run training first: python -m src.training")
        return
    
    try:
        print("🔄 Loading CRF model...")
        inference = CRFInference(model_path)
        
        # Show model information
        model_info = inference.get_model_info()
        print(f"✅ Model loaded successfully!")
        print(f"   Model Type: {model_info['model_type']}")
        if model_info['metadata']:
            print(f"   F1-Score: {model_info['metadata'].get('test_f1', 'N/A'):.4f}")
        print()
        
        # Interactive loop
        while True:
            try:
                user_input = input("📝 Enter text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("Please enter some text to segment")
                    continue
                
                # Segment text
                result = inference.segment(user_input)
                
                # Display results
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