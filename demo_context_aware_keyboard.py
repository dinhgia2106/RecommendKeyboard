#!/usr/bin/env python3
"""
Demo Context-Aware Vietnamese Keyboard
Integration của contextual processing với keyboard interface
"""

from typing import List, Dict
from ml.contextual_processor import ContextualVietnameseProcessor
from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor
from ml.semantic_validator import SemanticValidator
from ml.ambiguous_case_handler import AmbiguousCaseHandler


class ContextAwareKeyboard:
    """Context-aware Vietnamese keyboard với full integration"""

    def __init__(self):
        print("🚀 Initializing Context-Aware Vietnamese Keyboard...")

        # Initialize all processors
        self.contextual_processor = ContextualVietnameseProcessor()
        self.base_processor = HybridVietnameseProcessor()
        self.semantic_validator = SemanticValidator()
        self.ambiguous_handler = AmbiguousCaseHandler()

        # Context management
        self.context_buffer = ""
        self.max_context_length = 50  # words

        print("✅ Context-Aware Keyboard ready!")

    def update_context(self, text: str):
        """Update context buffer với new text"""
        words = text.split()
        all_words = self.context_buffer.split() + words

        # Keep only recent context
        if len(all_words) > self.max_context_length:
            all_words = all_words[-self.max_context_length:]

        self.context_buffer = " ".join(all_words)
        print(
            f"📝 Context updated: '{self.context_buffer[-60:]}{'...' if len(self.context_buffer) > 60 else ''}'")

    def get_suggestions(self, input_text: str, max_suggestions: int = 5) -> List[Dict]:
        """Get suggestions với full contextual processing"""
        print(f"\n🧠 Getting suggestions for: '{input_text}'")

        all_suggestions = []

        # 1. Try contextual processing first (highest priority)
        if self.context_buffer:
            contextual_suggestions = self.contextual_processor.process_with_context(
                self.context_buffer, input_text, max_suggestions=3)

            if contextual_suggestions:
                print(
                    f"✅ Found {len(contextual_suggestions)} contextual suggestions")
                for suggestion in contextual_suggestions:
                    suggestion['priority'] = 'high'
                all_suggestions.extend(contextual_suggestions)

        # 2. Try base processing (medium priority)
        base_suggestions = self.base_processor.process_text(
            input_text, max_suggestions=5)
        for suggestion in base_suggestions:
            suggestion['priority'] = 'medium'
        all_suggestions.extend(base_suggestions)

        # 3. Try ambiguous case handling (low priority)
        if self.ambiguous_handler.is_ambiguous(input_text):
            ambiguous_suggestions = self.ambiguous_handler.handle_ambiguous_case(
                input_text)
            for suggestion in ambiguous_suggestions:
                suggestion['priority'] = 'low'
            all_suggestions.extend(ambiguous_suggestions)

        # 4. Apply semantic validation
        validated_suggestions = self.semantic_validator.filter_suggestions(
            all_suggestions, min_confidence=40.0, max_suggestions=max_suggestions)

        # 5. Sort by priority and confidence
        final_suggestions = self._rank_suggestions(validated_suggestions)

        return final_suggestions[:max_suggestions]

    def _rank_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Rank suggestions by priority và confidence"""
        priority_weights = {
            'high': 1000,    # Contextual suggestions
            'medium': 100,   # Base processing
            'low': 10        # Ambiguous handling
        }

        for suggestion in suggestions:
            priority = suggestion.get('priority', 'medium')
            suggestion['final_score'] = (
                priority_weights[priority] + suggestion['confidence']
            )

        # Remove duplicates, keeping highest scored
        seen = {}
        for suggestion in suggestions:
            text = suggestion['vietnamese_text']
            if text not in seen or suggestion['final_score'] > seen[text]['final_score']:
                seen[text] = suggestion

        unique_suggestions = list(seen.values())
        unique_suggestions.sort(key=lambda x: x['final_score'], reverse=True)

        return unique_suggestions

    def clear_context(self):
        """Clear context buffer"""
        self.context_buffer = ""
        print("🗑️ Context cleared")

    def get_context_info(self) -> Dict:
        """Get current context information"""
        return {
            'context': self.context_buffer,
            'word_count': len(self.context_buffer.split()) if self.context_buffer else 0,
            'last_10_words': ' '.join(self.context_buffer.split()[-10:]) if self.context_buffer else ''
        }


def demo_context_scenarios():
    """Demo various context scenarios"""
    print("🎯 DEMO: CONTEXT-AWARE VIETNAMESE KEYBOARD")
    print("=" * 70)

    keyboard = ContextAwareKeyboard()

    # Scenario 1: Presentation context
    print(f"\n📖 SCENARIO 1: Presentation Context")
    print("-" * 40)

    keyboard.clear_context()
    keyboard.update_context("xin chào hôm nay tôi đem đến")

    suggestions = keyboard.get_suggestions("chocacban")
    print(f"\n🎯 Suggestions for 'chocacban':")
    for i, suggestion in enumerate(suggestions, 1):
        priority_icon = "🔥" if suggestion['priority'] == 'high' else "🔧" if suggestion['priority'] == 'medium' else "💡"
        print(
            f"  {i}. {priority_icon} '{suggestion['vietnamese_text']}' ({suggestion['confidence']:.1f}%) - {suggestion['method']}")

    # Scenario 2: Cooking context
    print(f"\n🍳 SCENARIO 2: Cooking Context")
    print("-" * 40)

    keyboard.clear_context()
    keyboard.update_context("hôm nay tôi nấu món ăn ngon")

    suggestions = keyboard.get_suggestions("chocacban")
    print(f"\n🎯 Suggestions for 'chocacban':")
    for i, suggestion in enumerate(suggestions, 1):
        priority_icon = "🔥" if suggestion['priority'] == 'high' else "🔧" if suggestion['priority'] == 'medium' else "💡"
        print(
            f"  {i}. {priority_icon} '{suggestion['vietnamese_text']}' ({suggestion['confidence']:.1f}%) - {suggestion['method']}")

    # Scenario 3: No context (ambiguous)
    print(f"\n❓ SCENARIO 3: No Context (Ambiguous)")
    print("-" * 40)

    keyboard.clear_context()

    suggestions = keyboard.get_suggestions("chocacban")
    print(f"\n🎯 Suggestions for 'chocacban':")
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            priority_icon = "🔥" if suggestion['priority'] == 'high' else "🔧" if suggestion['priority'] == 'medium' else "💡"
            print(
                f"  {i}. {priority_icon} '{suggestion['vietnamese_text']}' ({suggestion['confidence']:.1f}%) - {suggestion['method']}")
    else:
        print("  💡 No confident suggestions - better to ask for clarification")

    # Scenario 4: Progressive context building
    print(f"\n🔄 SCENARIO 4: Progressive Context Building")
    print("-" * 40)

    keyboard.clear_context()

    # Build context gradually
    steps = [
        ("xin chào", "toilamviec"),
        ("xin chào tôi làm việc", "tatcaban"),
        ("xin chào tôi làm việc tất cả bạn", "haitruong"),
        ("xin chào tôi làm việc tất cả bạn hai trường", "chocacban")
    ]

    for context_add, test_input in steps:
        keyboard.update_context(context_add)
        suggestions = keyboard.get_suggestions(test_input, max_suggestions=2)

        print(f"\n  Input: '{test_input}'")
        if suggestions:
            best = suggestions[0]
            priority_icon = "🔥" if best['priority'] == 'high' else "🔧" if best['priority'] == 'medium' else "💡"
            print(
                f"  Best: {priority_icon} '{best['vietnamese_text']}' ({best['confidence']:.1f}%)")
        else:
            print(f"  Result: No suggestions")


def demonstrate_context_evolution():
    """Demonstrate how context evolves suggestions"""
    print(f"\n🔬 CONTEXT EVOLUTION ANALYSIS")
    print("=" * 70)

    keyboard = ContextAwareKeyboard()

    input_text = "chocacban"
    contexts = [
        "",  # No context
        "tôi đi",  # Minimal context
        "tôi đi chợ",  # Shopping context
        "tôi đi chợ mua cá",  # Specific shopping
        "hôm nay tôi đem đến",  # Sharing context
        "xin chào hôm nay tôi đem đến"  # Full presentation context
    ]

    print(f"🧪 Input: '{input_text}'")
    print(f"📊 How context changes suggestions:")

    for i, context in enumerate(contexts):
        keyboard.clear_context()
        if context:
            keyboard.update_context(context)

        suggestions = keyboard.get_suggestions(input_text, max_suggestions=2)

        context_display = f"'{context}'" if context else "No context"
        print(f"\n  {i+1}. Context: {context_display}")

        if suggestions:
            best = suggestions[0]
            priority_icon = "🔥" if best['priority'] == 'high' else "🔧" if best['priority'] == 'medium' else "💡"
            print(
                f"     Best: {priority_icon} '{best['vietnamese_text']}' ({best['confidence']:.1f}%) - {best['method']}")
        else:
            print(f"     Result: No suggestions")

    print(f"\n💡 INSIGHT:")
    print("Context dramatically improves suggestion quality!")
    print("From ambiguous → meaningful with proper context.")


if __name__ == "__main__":
    demo_context_scenarios()
    demonstrate_context_evolution()
