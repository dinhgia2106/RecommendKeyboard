"""
Minimal AI Recommender for Vietnamese AI Keyboard
Delegates to ML components for suggestions
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from ml.word_segmentation import VietnameseWordSegmenter
    from ml.hybrid_suggestions import VietnameseHybridSuggestions
except ImportError as e:
    print(f"Warning: Could not import ML components: {e}")
    VietnameseWordSegmenter = None
    VietnameseHybridSuggestions = None


class AIRecommender:
    """
    AI Recommender that uses the ML components for suggestions
    """

    def __init__(self):
        """Initialize AI Recommender"""
        self.segmenter = None
        self.hybrid_suggestions = None
        self.stats = {
            'ai_engine_available': False,
            'ai_vocab_size': 0,
            'ai_device': 'CPU',
            'total_predictions': 0,
            'successful_predictions': 0
        }

        self._initialize_components()

    def _initialize_components(self):
        """Initialize ML components"""
        try:
            if VietnameseWordSegmenter:
                self.segmenter = VietnameseWordSegmenter()
                print("✅ Word Segmenter initialized")

            if VietnameseHybridSuggestions:
                self.hybrid_suggestions = VietnameseHybridSuggestions()
                print("✅ Hybrid Suggestions initialized")

            self.stats['ai_engine_available'] = True
            self.stats['ai_vocab_size'] = 268  # Mapping count

        except Exception as e:
            print(f"Warning: Failed to initialize ML components: {e}")
            self.stats['ai_engine_available'] = False

    def get_suggestions(self, text: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Get suggestions for input text"""
        if not text:
            return []

        suggestions = []
        self.stats['total_predictions'] += 1

        try:
            if self.hybrid_suggestions:
                # Use hybrid suggestions
                results = self.hybrid_suggestions.get_suggestions(
                    text, max_suggestions=max_suggestions)

                for result in results:
                    suggestions.append({
                        'word': result.get('word', ''),
                        'confidence': result.get('confidence', 0.0),
                        'source': result.get('method', 'hybrid'),
                        'score': result.get('confidence', 0.0)
                    })

                if suggestions:
                    self.stats['successful_predictions'] += 1

        except Exception as e:
            print(f"Error getting suggestions: {e}")

        # Fallback suggestions if no results
        if not suggestions:
            suggestions = [
                {'word': text, 'confidence': 0.1,
                    'source': 'fallback', 'score': 0.1}
            ]

        return suggestions[:max_suggestions]

    def segment_text(self, text: str) -> str:
        """Segment text into words"""
        if not text:
            return ""

        try:
            if self.segmenter:
                return self.segmenter.segment_text(text)
        except Exception as e:
            print(f"Error segmenting text: {e}")

        # Fallback: return original text
        return text

    def get_statistics(self) -> Dict[str, Any]:
        """Get AI engine statistics"""
        return self.stats.copy()

    def is_available(self) -> bool:
        """Check if AI engine is available"""
        return self.stats['ai_engine_available']

    def smart_recommend(self, user_input: str, context: List[str] = None, max_suggestions: int = 5) -> List[tuple]:
        """Smart recommendations compatible with UI expectations"""
        if not user_input:
            return []

        suggestions = self.get_suggestions(user_input, max_suggestions)

        # Convert to format expected by UI: (text, confidence, method)
        results = []
        for suggestion in suggestions:
            text = suggestion.get('word', '')
            confidence = suggestion.get('confidence', 0.0)
            method = suggestion.get('source', 'hybrid')
            results.append((text, confidence, method))

        return results

    def update_context(self, text: str):
        """Update context (placeholder for UI compatibility)"""
        # In a full implementation, this would update the context
        pass

    def clear_context(self):
        """Clear context (placeholder for UI compatibility)"""
        # In a full implementation, this would clear the context
        pass


# Global instance
_ai_recommender_instance = None


def get_ai_recommender() -> AIRecommender:
    """Get global AI Recommender instance"""
    global _ai_recommender_instance

    if _ai_recommender_instance is None:
        _ai_recommender_instance = AIRecommender()

    return _ai_recommender_instance
