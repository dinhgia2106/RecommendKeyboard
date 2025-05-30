#!/usr/bin/env python3
"""
Test multiple suggestions với enhanced processor
"""

from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor


def test_multiple_suggestions():
    """Test nhiều gợi ý cho một input"""
    print("🚀 Testing Multiple Suggestions System\n")

    processor = HybridVietnameseProcessor()

    test_cases = [
        "toihocbai",
        "toilasinhvien",
        "xinchao",
        "homnaytoilam",
        "camon",
        "maytinh",
        "dienthoai",
        "ancom",
        "dihoc",
        "xemphim",
        # Test cases với nhiều possibilities hơn
        "motngaydepmoi",  # Should have multiple variations
        "todihocsinhlong",  # Ambiguous segmentation
        "vietmamtutiet",   # Multiple possible meanings
        "hohocbaitap",     # Similar to known patterns
        "ancomtruahomnay",  # Multiple combinations possible
        "toidemden"  # Added for the new test case
    ]

    for input_text in test_cases:
        print(f"📝 Input: '{input_text}'")
        results = processor.process_text(input_text, max_suggestions=10)

        print(f"   💡 Found {len(results)} suggestions:")
        for i, result in enumerate(results, 1):
            method_icon = get_method_icon(result['method'])
            print(
                f"   {i:2d}. {result['vietnamese_text']} {method_icon} ({result['confidence']}%) [{result['method']}]")
        print()


def get_method_icon(method: str) -> str:
    """Get icon for method"""
    icons = {
        'core_sentence': '🎯',
        'core_compound': '🔗',
        'corpus_trigram': '⭐',
        'corpus_bigram': '💫',
        'extended_compound': '📦',
        'extended_word': '📚',
        'hybrid_segmentation': '🧠',
        'aggressive_segmentation': '⚡',
        'conservative_segmentation': '🏛️',
        'simple_segmentation': '📝',
        'partial_compound': '🔗',
        'partial_extended': '📦',
        'corpus_trigram_partial': '⭐',
        'corpus_bigram_partial': '💫'
    }
    return icons.get(method, '⚡')


if __name__ == "__main__":
    test_multiple_suggestions()
