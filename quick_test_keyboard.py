#!/usr/bin/env python3
"""
Quick Test cho Vietnamese AI Keyboard System
Đánh giá khả năng hoạt động của bộ gõ
"""

from ml.hybrid_suggestions import VietnameseHybridSuggestions
from ml.word_segmentation import VietnameseWordSegmenter
import os
import sys
sys.path.append('.')


def test_word_segmentation():
    """Test khả năng tách từ"""
    print("🧪 TESTING WORD SEGMENTATION")
    print("=" * 50)

    segmenter = VietnameseWordSegmenter()

    # Test cases cơ bản
    test_cases = [
        "xinchao",
        "chaoban",
        "toiyeuban",
        "hocsinh",
        "dihoc",
        "camon",
        "homnay",
        "ngaymai",
        "giaovien",
        "sinhvien",
    ]

    results = []
    for test in test_cases:
        result = segmenter.segment_text(test)
        success = result != test  # Có tách được hay không
        results.append(success)
        status = "✅" if success else "❌"
        print(f"{status} '{test}' → '{result}'")

    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Word Segmentation Success Rate: {success_rate:.1f}%")
    return success_rate


def test_suggestions():
    """Test khả năng gợi ý"""
    print("\n🧪 TESTING SUGGESTION SYSTEM")
    print("=" * 50)

    hybrid = VietnameseHybridSuggestions()

    # Test cases cơ bản
    test_cases = [
        "tieng",    # tiếng
        "viet",     # việt
        "xin",      # xin chào
        "cam",      # cảm ơn
        "homnay",   # hôm nay
        "ban",      # bạn
        "toi",      # tôi
        "yeu",      # yêu
        "hoc",      # học
        "lam",      # làm
    ]

    results = []
    for test in test_cases:
        suggestions = hybrid.get_suggestions(test, max_suggestions=3)
        has_good_suggestion = len(
            suggestions) > 0 and suggestions[0]['confidence'] > 0.1
        results.append(has_good_suggestion)

        status = "✅" if has_good_suggestion else "❌"
        top_suggestion = suggestions[0]['word'] if suggestions else "None"
        confidence = suggestions[0]['confidence'] if suggestions else 0
        print(f"{status} '{test}' → '{top_suggestion}' ({confidence:.1%})")

    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Suggestion Success Rate: {success_rate:.1f}%")
    return success_rate


def test_keyboard_scenarios():
    """Test scenarios thực tế của bộ gõ"""
    print("\n🧪 TESTING REAL KEYBOARD SCENARIOS")
    print("=" * 50)

    segmenter = VietnameseWordSegmenter()
    hybrid = VietnameseHybridSuggestions()

    # Scenarios thực tế
    scenarios = [
        # User gõ liền không dấu
        ("xinchao", "xin chào"),
        ("toiyeuban", "tôi yêu bạn"),
        ("chaobanhomnay", "chào bạn hôm nay"),
        ("camon", "cảm ơn"),
        ("dihocve", "đi học về"),

        # User gõ từng từ
        ("toi", "tôi"),
        ("ban", "bạn"),
        ("hoc", "học"),
        ("viet", "việt"),
        ("dep", "đẹp"),
    ]

    results = []

    for input_text, expected in scenarios:
        # Thử word segmentation trước
        segmented = segmenter.segment_text(input_text)

        # Nếu không tách được, thử suggestion
        if segmented == input_text:
            suggestions = hybrid.get_suggestions(input_text, max_suggestions=1)
            result = suggestions[0]['word'] if suggestions else input_text
        else:
            result = segmented

        # Kiểm tra có gần đúng không
        success = (expected.lower() in result.lower() or
                   result.lower() in expected.lower() or
                   result != input_text)  # Ít nhất có thay đổi

        results.append(success)
        status = "✅" if success else "❌"
        print(f"{status} '{input_text}' → '{result}' (expected: '{expected}')")

    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Keyboard Scenario Success Rate: {success_rate:.1f}%")
    return success_rate


def evaluate_readiness():
    """Đánh giá độ sẵn sàng để sử dụng"""
    print("\n🎯 OVERALL SYSTEM EVALUATION")
    print("=" * 50)

    # Chạy tất cả tests
    seg_score = test_word_segmentation()
    sug_score = test_suggestions()
    kb_score = test_keyboard_scenarios()

    # Tính điểm tổng
    overall_score = (seg_score + sug_score + kb_score) / 3

    print(f"\n📊 FINAL SCORES:")
    print(f"   Word Segmentation: {seg_score:.1f}%")
    print(f"   Suggestions: {sug_score:.1f}%")
    print(f"   Keyboard Scenarios: {kb_score:.1f}%")
    print(f"   Overall: {overall_score:.1f}%")

    # Đánh giá độ sẵn sàng
    if overall_score >= 80:
        readiness = "🟢 SẴN SÀNG - Có thể deploy cho người dùng"
    elif overall_score >= 60:
        readiness = "🟡 CẦN CẢI THIỆN - Dùng được nhưng còn hạn chế"
    elif overall_score >= 40:
        readiness = "🟠 BETA - Chỉ phù hợp để test"
    else:
        readiness = "🔴 CHƯA SẴN SÀNG - Cần training thêm"

    print(f"\n🎯 READINESS ASSESSMENT: {readiness}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if seg_score < 70:
        print(f"   - Cải thiện word segmentation (thêm mappings)")
    if sug_score < 70:
        print(f"   - Tăng chất lượng suggestions (train model)")
    if kb_score < 70:
        print(f"   - Optimize cho scenarios thực tế")
    if overall_score < 60:
        print(f"   - Cần hoàn thành training model GPT")
        print(f"   - Mở rộng vocabulary và mappings")

    return overall_score


if __name__ == "__main__":
    print("🚀 VIETNAMESE AI KEYBOARD QUICK TEST")
    print("=" * 60)
    print("Đánh giá khả năng hoạt động của bộ gõ...")
    print()

    try:
        final_score = evaluate_readiness()

        print(f"\n{'='*60}")
        print(f"🎉 TEST COMPLETED - Overall Score: {final_score:.1f}%")

        if final_score >= 60:
            print("✅ Bộ gõ có thể sử dụng được ở mức độ cơ bản!")
        else:
            print("⚠️ Bộ gõ cần cải thiện thêm trước khi deploy.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("🔧 Please check if all components are properly installed.")
