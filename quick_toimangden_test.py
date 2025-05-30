#!/usr/bin/env python3
"""
Quick test for toimangden debug
"""

from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor


def test_toimangden():
    print("🔍 Debugging toimangden specifically")
    processor = HybridVietnameseProcessor()

    input_text = "toimangden"
    results = processor.process_text(input_text, max_suggestions=5)

    print(f"📝 Input: {input_text}")
    print(f"📊 Found {len(results)} suggestions:")

    for i, result in enumerate(results, 1):
        print(
            f"  {i}. '{result['vietnamese_text']}' ({result['confidence']}%) - {result['method']}")

    # Check if in AI learned patterns
    if input_text in processor.ai_learned_patterns:
        pattern = processor.ai_learned_patterns[input_text]
        print(f"\n🤖 AI Pattern found: {pattern}")
    else:
        print(f"\n❌ Not found in AI learned patterns")

    # Check core dictionaries
    print(f"\n🔍 Dictionary checks:")
    print(
        f"  Core syllables: toi={processor.core_syllables.get('toi', 'Not found')}")
    print(
        f"  Core syllables: man={processor.core_syllables.get('man', 'Not found')}")
    print(
        f"  Core syllables: mang={processor.core_syllables.get('mang', 'Not found')}")
    print(
        f"  Core syllables: den={processor.core_syllables.get('den', 'Not found')}")


if __name__ == "__main__":
    test_toimangden()
