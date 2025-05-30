#!/usr/bin/env python3
"""
Analyze chocacban case - Tại sao model gợi ý kỳ lạ
"""

from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor


def analyze_chocacban_issue():
    """Analyze tại sao chocacban cho kết quả kỳ lạ"""
    print("🔍 ANALYZING CHOCACBAN ISSUE")
    print("=" * 60)

    processor = HybridVietnameseProcessor()
    input_text = "chocacban"

    print(f"📝 Input: {input_text}")
    print(f"📊 Length: {len(input_text)} chars")

    # Get all suggestions
    results = processor.process_text(input_text, max_suggestions=5)

    print(f"\n🤖 Current Results:")
    for i, result in enumerate(results, 1):
        method_indicator = "🚀" if 'phobert' in result['method'] else "🔧"
        print(
            f"  {i}. '{result['vietnamese_text']}' ({result['confidence']}%) - {result['method']} {method_indicator}")
        if 'phobert_score' in result:
            print(f"     PhoBERT score: {result['phobert_score']:.1f}")

    print(f"\n🔍 PROBLEM ANALYSIS:")

    # 1. Check if in dictionaries
    print("1. Dictionary Coverage:")
    if input_text in processor.core_sentences:
        print(
            f"   ✅ Found in core_sentences: {processor.core_sentences[input_text]}")
    elif input_text in processor.core_compounds:
        print(
            f"   ✅ Found in core_compounds: {processor.core_compounds[input_text]}")
    elif input_text in processor.ai_learned_patterns:
        print(
            f"   ✅ Found in ai_learned_patterns: {processor.ai_learned_patterns[input_text]}")
    else:
        print("   ❌ NOT found in any exact match dictionary")

    # 2. Segmentation analysis
    print("\n2. Possible Segmentations:")
    possible_meanings = [
        ("cho cá bạn", "chờ cá bạn", "wait for fish friend"),
        ("chợ cá bạn", "chợ cá bạn", "fish market friend"),
        ("chọn cá bạn", "chọn cá bạn", "choose fish friend"),
        ("chó cả bạn", "chó cả bạn", "dog of friend"),
        ("chò cà bạn", "chờ cà bạn", "wait eggplant friend")
    ]

    for seg_input, seg_output, meaning in possible_meanings:
        print(f"   • {seg_input} → {seg_output} ({meaning})")

    # 3. Check individual syllables
    print("\n3. Individual Syllable Coverage:")
    syllables = ['cho', 'ca', 'ban', 'choc', 'cac', 'ban']
    for syl in syllables:
        if syl in processor.core_syllables:
            print(f"   ✅ '{syl}' → '{processor.core_syllables[syl]}'")
        elif syl in processor.extended_syllables:
            print(
                f"   🔶 '{syl}' → '{processor.extended_syllables[syl]}' (extended)")
        else:
            print(f"   ❌ '{syl}' not found")

    # 4. Pattern analysis
    print("\n4. Pattern Analysis:")
    print(
        f"   • Length: {len(input_text)} chars (không match toi+verb+object pattern)")
    print(f"   • Structure: Không clear semantic structure")
    print(f"   • Context: Thiếu context để determine intent")

    print(f"\n💡 WHY MODEL BEHAVES THIS WAY:")
    print("   🔧 Model tries best với available patterns")
    print("   🔧 'chóc ác ban' có thể từ aggressive_segmentation")
    print("   🔧 'chợ ca c ba n' từ partial matches")
    print("   🔧 PhoBERT enhances based on Vietnamese language patterns")
    print("   🔧 Without clear intent, model falls back to closest matches")


def suggest_improvements():
    """Suggest improvements cho ambiguous cases"""
    print(f"\n🚀 SUGGESTED IMPROVEMENTS:")
    print("=" * 60)

    print("1. 📊 Context-Aware Processing:")
    print("   • Sử dụng previous words để determine intent")
    print("   • Example: 'tôi đi chocacban' → likely 'chợ cá bạn'")

    print("\n2. 🤖 Enhanced PhoBERT Integration:")
    print("   • Better word segmentation với VnCoreNLP")
    print("   • Use PhoBERT's fill-mask for ambiguous cases")
    print("   • Contextual embedding similarity")

    print("\n3. 📚 Expanded Training Data:")
    print("   • Add more compound patterns beyond toi+verb+object")
    print("   • Include food/market related patterns")
    print("   • User feedback learning")

    print("\n4. 🎯 Intent Recognition:")
    print("   • Semantic category classification")
    print("   • Most common Vietnamese phrases")
    print("   • Frequency-based suggestions")

    print("\n5. 🔄 Interactive Refinement:")
    print("   • Ask user for clarification on ambiguous inputs")
    print("   • Learn from user selections")
    print("   • Adaptive personalization")


def test_alternative_approaches():
    """Test alternative approaches cho chocacban"""
    print(f"\n🧪 TESTING ALTERNATIVE APPROACHES:")
    print("=" * 60)

    # Manual better suggestions
    better_suggestions = [
        ("chợ cá bạn", 85, "semantic_analysis"),
        ("chờ cả bạn", 80, "phonetic_matching"),
        ("chọn cá bạn", 75, "contextual_inference"),
        ("cho cả bạn", 70, "simplified_segmentation")
    ]

    print("🎯 Better Suggestions (manual analysis):")
    for i, (text, conf, method) in enumerate(better_suggestions, 1):
        print(f"  {i}. '{text}' ({conf}%) - {method}")

    print(f"\n📈 Improvement Opportunities:")
    print("  • Semantic understanding: chợ (market) + cá (fish) makes sense")
    print("  • Phonetic similarity: 'cho' sound similar to 'chờ', 'chọn'")
    print("  • Context patterns: Vietnamese compound words")
    print("  • User intent modeling: Most likely meanings")


def main():
    """Main analysis"""
    analyze_chocacban_issue()
    suggest_improvements()
    test_alternative_approaches()

    print(f"\n🏆 CONCLUSION:")
    print("Model KHÔNG tệ - đây là ambiguous case cần context!")
    print("PhoBERT vẫn rất powerful cho clear patterns.")
    print("Cần enhance cho ambiguous cases với better segmentation.")


if __name__ == "__main__":
    main()
