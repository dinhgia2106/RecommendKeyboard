#!/usr/bin/env python3
"""
Test AI Improvements for Vietnamese Keyboard
Kiểm tra AI-driven approach thay vì manual approach
"""

from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor


def test_ai_driven_improvements():
    """Test AI-driven improvements cho các cases problematics"""
    print("🤖 Testing AI-Driven Improvements")
    print("=" * 50)

    processor = HybridVietnameseProcessor()

    test_cases = [
        ("toidemden", "tôi đem đến"),
        ("toitangban", "tôi tặng bạn"),
        ("toidicho", "tôi đi chợ"),
        ("toiluubai", "tôi lưu bài"),
        ("toiguibai", "tôi gửi bài"),
        ("toidocbao", "tôi đọc báo"),
        ("toixemtv", "tôi xem TV"),
        ("toinghetho", "tôi nghe thơ")
    ]

    for input_text, expected in test_cases:
        print(f"\n📝 Testing: {input_text} → {expected}")
        results = processor.process_text(input_text, max_suggestions=5)

        if results:
            for i, result in enumerate(results, 1):
                status = "✅" if result['vietnamese_text'] == expected else "❌"
                print(
                    f"  {i}. {result['vietnamese_text']} ({result['confidence']}%) - {result['method']} {status}")

            # Check if expected in results
            expected_found = any(r['vietnamese_text'] ==
                                 expected for r in results)
            if expected_found:
                print(f"  🎯 SUCCESS: Expected result found!")
            else:
                print(f"  ⚠️ Expected '{expected}' not found in suggestions")
        else:
            print("  ❌ No suggestions generated")

    print(f"\n📊 System Statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")


if __name__ == "__main__":
    test_ai_driven_improvements()
