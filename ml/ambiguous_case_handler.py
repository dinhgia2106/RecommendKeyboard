#!/usr/bin/env python3
"""
Ambiguous Case Handler cho Vietnamese Keyboard
Xử lý các cases không rõ ràng như chocacban
"""

from typing import List, Dict, Optional


class AmbiguousCaseHandler:
    """Handler cho ambiguous Vietnamese input cases"""

    def __init__(self):
        # Common ambiguous patterns và likely interpretations
        self.ambiguous_patterns = {
            # Market/Food patterns
            'chocacban': [
                ('chợ cá bạn', 85, 'semantic_market_pattern'),
                ('chờ cả bạn', 80, 'phonetic_action_pattern'),
                ('chọn cá bạn', 75, 'action_food_pattern'),
                ('cho cả bạn', 70, 'give_action_pattern')
            ],
            'diancomtua': [
                ('đi ăn cơm tự', 85, 'action_food_pattern'),
                ('đi án cơm tưa', 70, 'phonetic_variation')
            ],
            'muacangghe': [
                ('mua cà nghe', 80, 'buy_item_pattern'),
                ('mua căng ghế', 75, 'buy_furniture_pattern')
            ]
        }

        # Common semantic categories
        self.semantic_categories = {
            'food_market': ['chợ', 'cá', 'cơm', 'thịt', 'rau', 'mua', 'bán'],
            'social_action': ['chờ', 'gặp', 'cả', 'bạn', 'mọi', 'người'],
            'daily_action': ['đi', 'về', 'ăn', 'ngủ', 'làm', 'học']
        }

        print("🔧 Ambiguous Case Handler initialized")

    def handle_ambiguous_case(self, input_text: str) -> List[Dict]:
        """Handle ambiguous cases với better suggestions"""
        input_text = input_text.lower().strip()

        # Check exact patterns
        if input_text in self.ambiguous_patterns:
            suggestions = []
            for text, confidence, method in self.ambiguous_patterns[input_text]:
                suggestions.append({
                    'vietnamese_text': text,
                    'confidence': confidence,
                    'method': method,
                    'source': 'ambiguous_handler'
                })
            return suggestions

        # Semantic-based suggestions
        semantic_suggestions = self._generate_semantic_suggestions(input_text)
        if semantic_suggestions:
            return semantic_suggestions

        # Phonetic-based suggestions
        phonetic_suggestions = self._generate_phonetic_suggestions(input_text)
        return phonetic_suggestions

    def _generate_semantic_suggestions(self, input_text: str) -> List[Dict]:
        """Generate suggestions based on semantic analysis"""
        suggestions = []

        # Check for food/market patterns
        if any(food_word in input_text for food_word in ['cho', 'ca', 'com', 'ban']):
            if 'cho' in input_text and 'ca' in input_text:
                suggestions.append({
                    'vietnamese_text': 'chợ cá bạn',
                    'confidence': 85,
                    'method': 'semantic_food_market',
                    'source': 'semantic_analysis'
                })

        # Check for social action patterns
        if any(social_word in input_text for social_word in ['cho', 'ban', 'ca']):
            if 'cho' in input_text and 'ban' in input_text:
                suggestions.append({
                    'vietnamese_text': 'chờ cả bạn',
                    'confidence': 80,
                    'method': 'semantic_social_action',
                    'source': 'semantic_analysis'
                })

        return suggestions[:3]

    def _generate_phonetic_suggestions(self, input_text: str) -> List[Dict]:
        """Generate suggestions based on phonetic similarity"""
        suggestions = []

        # Common phonetic substitutions trong Vietnamese
        phonetic_rules = {
            'cho': ['chờ', 'chợ', 'chọn', 'cho'],
            'ca': ['cả', 'cá', 'ca'],
            'ban': ['bạn', 'ban', 'băn']
        }

        # Apply phonetic rules
        if input_text.startswith('cho') and 'ca' in input_text and 'ban' in input_text:
            base_patterns = [
                'chợ cá bạn',
                'chờ cả bạn',
                'chọn cá bạn',
                'cho cả bạn'
            ]

            for i, pattern in enumerate(base_patterns):
                confidence = 85 - (i * 5)  # Decreasing confidence
                suggestions.append({
                    'vietnamese_text': pattern,
                    'confidence': confidence,
                    'method': 'phonetic_matching',
                    'source': 'phonetic_analysis'
                })

        return suggestions

    def is_ambiguous(self, input_text: str) -> bool:
        """Check if input is likely ambiguous"""
        input_text = input_text.lower().strip()

        # Known ambiguous patterns
        if input_text in self.ambiguous_patterns:
            return True

        # Heuristics for ambiguous detection
        ambiguous_indicators = [
            len(input_text) >= 8,  # Long compounds often ambiguous
            input_text.count('c') >= 2,  # Multiple 'c' sounds
            not input_text.startswith('toi'),  # Not clear toi+verb+object
            any(char in input_text for char in [
                'ch', 'ca', 'co'])  # Ambiguous sounds
        ]

        return sum(ambiguous_indicators) >= 2

    def get_statistics(self) -> Dict:
        """Get handler statistics"""
        return {
            'known_ambiguous_patterns': len(self.ambiguous_patterns),
            'semantic_categories': len(self.semantic_categories),
            'total_known_variations': sum(len(patterns) for patterns in self.ambiguous_patterns.values())
        }


def test_ambiguous_handler():
    """Test ambiguous case handler"""
    print("🧪 Testing Ambiguous Case Handler")
    print("=" * 50)

    handler = AmbiguousCaseHandler()

    test_cases = [
        'chocacban',
        'diancomtua',
        'muacangghe',
        'unknown_case'
    ]

    for test_input in test_cases:
        print(f"\n📝 Input: {test_input}")

        if handler.is_ambiguous(test_input):
            print("  🔍 Detected as AMBIGUOUS")
            suggestions = handler.handle_ambiguous_case(test_input)

            if suggestions:
                for i, sug in enumerate(suggestions, 1):
                    print(
                        f"  {i}. {sug['vietnamese_text']} ({sug['confidence']}%) - {sug['method']}")
            else:
                print("  ❌ No specific suggestions")
        else:
            print("  ✅ Not ambiguous - regular processing")

    # Statistics
    stats = handler.get_statistics()
    print(f"\n📊 Handler Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_ambiguous_handler()
