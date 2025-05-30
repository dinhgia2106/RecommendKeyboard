#!/usr/bin/env python3
"""
Model Quality Assessment
Test thực tế chất lượng model với various inputs
"""

from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor
from ml.contextual_processor import ContextualVietnameseProcessor
from ml.semantic_validator import SemanticValidator
from demo_context_aware_keyboard import ContextAwareKeyboard


def test_simple_cases():
    """Test các cases đơn giản"""
    print("🧪 TESTING SIMPLE CASES")
    print("=" * 50)

    processor = HybridVietnameseProcessor()

    simple_cases = [
        ('mot', ['một']),
        ('toi', ['tôi']),
        ('ban', ['bạn']),
        ('nha', ['nhà']),
        ('di', ['đi']),
        ('an', ['ăn']),
        ('hoc', ['học']),
        ('lam', ['làm']),
        ('viet', ['viết']),
        ('doc', ['đọc'])
    ]

    total_tests = len(simple_cases)
    passed_tests = 0

    for input_text, expected in simple_cases:
        results = processor.process_text(input_text, max_suggestions=5)

        print(f"\n📝 Input: '{input_text}'")
        print(f"   Expected: {expected}")
        print(f"   Got: ", end="")

        if results:
            actual = [r['vietnamese_text'] for r in results]
            print(f"{actual}")

            # Check if any expected result is in top suggestions
            if any(exp in actual for exp in expected):
                print(f"   ✅ PASS")
                passed_tests += 1
            else:
                print(f"   ❌ FAIL")

            # Show confidence levels
            best = results[0]
            print(
                f"   Best: '{best['vietnamese_text']}' ({best['confidence']}%)")

        else:
            print("No suggestions")
            print(f"   ❌ FAIL")

    success_rate = (passed_tests / total_tests) * 100
    print(f"\n📊 SIMPLE CASES RESULTS:")
    print(f"   Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")

    return success_rate


def test_complex_cases():
    """Test các cases phức tạp"""
    print(f"\n🧪 TESTING COMPLEX CASES")
    print("=" * 50)

    processor = HybridVietnameseProcessor()

    complex_cases = [
        ('toidemden', ['tôi đem đến']),
        ('toilambai', ['tôi làm bài']),
        ('toimangden', ['tôi mang đến']),
        ('anhdichuyen', ['anh đi chuyển']),
        ('emhocbai', ['em học bài']),
        ('chungtoilam', ['chúng tôi làm']),
        ('banvietbai', ['bạn viết bài']),
        ('cogiaoday', ['cô giáo dạy']),
        ('thaygiaoday', ['thầy giáo dạy']),
        ('hocsinhhoc', ['học sinh học'])
    ]

    total_tests = len(complex_cases)
    passed_tests = 0

    for input_text, expected in complex_cases:
        results = processor.process_text(input_text, max_suggestions=5)

        print(f"\n📝 Input: '{input_text}'")
        print(f"   Expected: {expected}")

        if results:
            actual = [r['vietnamese_text'] for r in results]
            print(f"   Got: {actual}")

            # Check if any expected result is in top suggestions
            if any(exp in actual for exp in expected):
                print(f"   ✅ PASS")
                passed_tests += 1
            else:
                print(f"   ❌ FAIL")

            # Show confidence levels
            best = results[0]
            print(
                f"   Best: '{best['vietnamese_text']}' ({best['confidence']}%)")

        else:
            print(f"   Got: No suggestions")
            print(f"   ❌ FAIL")

    success_rate = (passed_tests / total_tests) * 100
    print(f"\n📊 COMPLEX CASES RESULTS:")
    print(f"   Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")

    return success_rate


def test_problematic_cases():
    """Test các cases có vấn đề"""
    print(f"\n🧪 TESTING PROBLEMATIC CASES")
    print("=" * 50)

    processor = HybridVietnameseProcessor()

    problematic_cases = [
        ('mot', ['một'], 'Too few suggestions'),
        ('chocacban', ['cho các bạn'], 'Ambiguous without context'),
        ('diancomtua', ['đi ăn cơm'], 'Unclear segmentation'),
        ('abcdefgh', [], 'Nonsense input'),
        ('xyzabc', [], 'Random letters'),
        ('123456', [], 'Numbers'),
        ('', [], 'Empty input')
    ]

    for input_text, expected, issue in problematic_cases:
        print(f"\n📝 Input: '{input_text}' - Issue: {issue}")

        if input_text:  # Skip empty input
            results = processor.process_text(input_text, max_suggestions=5)

            if results:
                print(f"   Suggestions: {len(results)}")
                for i, result in enumerate(results, 1):
                    print(
                        f"     {i}. '{result['vietnamese_text']}' ({result['confidence']}%)")
            else:
                print(f"   No suggestions")

        else:
            print(f"   Skipped empty input")


def test_contextual_improvements():
    """Test contextual improvements"""
    print(f"\n🧪 TESTING CONTEXTUAL IMPROVEMENTS")
    print("=" * 50)

    keyboard = ContextAwareKeyboard()

    test_cases = [
        {
            'context': 'xin chào hôm nay tôi đem đến',
            'input': 'chocacban',
            'expected': 'cho các bạn'
        },
        {
            'context': '',
            'input': 'chocacban',
            'expected': 'ambiguous'
        },
        {
            'context': 'tôi đi học',
            'input': 'mot',
            'expected': 'một'
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}:")
        print(f"   Context: '{case['context']}'")
        print(f"   Input: '{case['input']}'")

        keyboard.clear_context()
        if case['context']:
            keyboard.update_context(case['context'])

        suggestions = keyboard.get_suggestions(
            case['input'], max_suggestions=3)

        if suggestions:
            print(f"   Results:")
            for j, suggestion in enumerate(suggestions, 1):
                priority = "🔥" if suggestion['priority'] == 'high' else "🔧"
                print(
                    f"     {j}. {priority} '{suggestion['vietnamese_text']}' ({suggestion['confidence']:.1f}%)")
        else:
            print(f"   No suggestions")


def analyze_specific_mot_case():
    """Analyze tại sao 'mot' case weak"""
    print(f"\n🔍 ANALYZING 'MOT' CASE SPECIFICALLY")
    print("=" * 50)

    processor = HybridVietnameseProcessor()

    print("📝 Testing various similar inputs:")

    similar_inputs = ['mot', 'moi', 'mon', 'moc', 'mod', 'mou', 'moy']

    for input_text in similar_inputs:
        results = processor.process_text(input_text, max_suggestions=5)

        print(f"\n  '{input_text}' →")
        if results:
            for i, result in enumerate(results, 1):
                print(
                    f"    {i}. '{result['vietnamese_text']}' ({result['confidence']}%)")
        else:
            print("    No suggestions")

    print(f"\n🔍 Deep analysis of 'mot':")
    results = processor.process_text('mot', max_suggestions=10)

    if results:
        print(f"  Total suggestions: {len(results)}")
        for i, result in enumerate(results, 1):
            print(
                f"    {i}. '{result['vietnamese_text']}' ({result['confidence']}%) - {result['method']}")

    # Check if in dictionaries
    print(f"\n📚 Dictionary coverage:")
    if hasattr(processor, 'core_syllables') and 'mot' in processor.core_syllables:
        print(
            f"  ✅ Found in core_syllables: {processor.core_syllables['mot']}")
    else:
        print(f"  ❌ Not in core_syllables")

    if hasattr(processor, 'extended_syllables') and 'mot' in processor.extended_syllables:
        print(
            f"  ✅ Found in extended_syllables: {processor.extended_syllables['mot']}")
    else:
        print(f"  ❌ Not in extended_syllables")


def overall_assessment():
    """Overall model quality assessment"""
    print(f"\n🏆 OVERALL MODEL QUALITY ASSESSMENT")
    print("=" * 50)

    # Run all tests
    simple_rate = test_simple_cases()
    complex_rate = test_complex_cases()

    # Calculate overall score
    overall_score = (simple_rate + complex_rate) / 2

    print(f"\n📊 FINAL ASSESSMENT:")
    print(f"   Simple Cases: {simple_rate:.1f}%")
    print(f"   Complex Cases: {complex_rate:.1f}%")
    print(f"   Overall Score: {overall_score:.1f}%")

    # Assessment categories
    if overall_score >= 90:
        grade = "🏆 EXCELLENT"
    elif overall_score >= 80:
        grade = "✅ GOOD"
    elif overall_score >= 70:
        grade = "⚠️ ACCEPTABLE"
    elif overall_score >= 60:
        grade = "❌ POOR"
    else:
        grade = "💥 TERRIBLE"

    print(f"   Grade: {grade}")

    print(f"\n💡 RECOMMENDATIONS:")
    if overall_score < 80:
        print("   • Model needs significant improvement")
        print("   • Focus on basic syllable coverage")
        print("   • Improve pattern matching algorithms")
    else:
        print("   • Model performs reasonably well")
        print("   • Context-aware processing is a major improvement")
        print("   • Continue enhancing edge cases")


if __name__ == "__main__":
    overall_assessment()
    test_problematic_cases()
    test_contextual_improvements()
    analyze_specific_mot_case()
