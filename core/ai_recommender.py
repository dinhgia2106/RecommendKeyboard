"""
AI-powered Vietnamese Non-Accented Recommender
Uses trained GPT model for intelligent word prediction
"""

from .text_processor import TextProcessor
import os
import sys
from typing import List, Tuple, Dict, Optional
import time

# Add ml module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.inference import get_inference_engine, predict_vietnamese_words
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML module not available: {e}")
    ML_AVAILABLE = False


class AIRecommender:
    """AI-powered Vietnamese Non-Accented Recommender"""

    def __init__(
        self,
        model_path: str = "checkpoints/vietnamese_non_accented_gpt_best.pth",
        data_dir: str = "ml/data",
        fallback_to_simple: bool = True
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.fallback_to_simple = fallback_to_simple

        # Initialize components
        self.text_processor = TextProcessor()
        self.inference_engine = None

        # Context tracking
        self.context_words = []
        self.max_context_length = 5

        # Performance tracking
        self.recommendations_served = 0
        self.total_response_time = 0.0

        # Initialize AI engine
        self.initialize_ai_engine()

    def initialize_ai_engine(self):
        """Initialize AI inference engine"""
        if not ML_AVAILABLE:
            print("⚠️  ML components not available. Using fallback mode.")
            return

        try:
            print("🤖 Initializing AI inference engine...")
            self.inference_engine = get_inference_engine(
                model_path=self.model_path,
                data_dir=self.data_dir
            )
            print("✅ AI inference engine loaded successfully!")

            # Print engine statistics
            stats = self.inference_engine.get_statistics()
            print(f"📊 Engine stats: {stats}")

        except Exception as e:
            print(f"❌ Failed to initialize AI engine: {e}")
            if not self.fallback_to_simple:
                raise
            print("🔄 Falling back to simple recommendations")

    def recommend(
        self,
        user_input: str,
        context: List[str] = None,
        max_suggestions: int = 8,
        use_context: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Get word recommendations for pinyin input

        Args:
            user_input: User's pinyin input
            context: Context words (optional)
            max_suggestions: Maximum number of suggestions
            use_context: Whether to use context for predictions

        Returns:
            List of (word, confidence, method) tuples
        """
        start_time = time.time()

        # Clean input
        clean_input = self.text_processor.clean_text(
            user_input.strip().lower())
        if not clean_input:
            return []

        # Use provided context or internal context
        if use_context:
            prediction_context = context if context else self.context_words
        else:
            prediction_context = []

        recommendations = []

        # Try AI predictions first
        if self.inference_engine:
            try:
                recommendations = self.inference_engine.non_accented_to_words(
                    non_accented_input=clean_input,
                    context=prediction_context,
                    max_suggestions=max_suggestions,
                    use_model=True,
                    temperature=0.8
                )
            except Exception as e:
                print(f"AI prediction error: {e}")

        # Fallback to simple recommendations
        if not recommendations and self.fallback_to_simple:
            recommendations = self._get_simple_recommendations(
                clean_input, max_suggestions
            )

        # Update performance tracking
        response_time = time.time() - start_time
        self.recommendations_served += 1
        self.total_response_time += response_time

        return recommendations

    def _get_simple_recommendations(
        self,
        user_input: str,
        max_suggestions: int
    ) -> List[Tuple[str, float, str]]:
        """Simple fallback recommendations without AI"""

        # Basic Vietnamese word mappings for common inputs
        simple_mappings = {
            'xinchao': [('xin chào', 0.9), ('xin cháo', 0.3)],
            'chao': [('chào', 0.9), ('cháo', 0.4)],
            'xin': [('xin', 0.9), ('xinh', 0.3)],
            'cam': [('cảm', 0.8), ('cam', 0.7), ('cắm', 0.3)],
            'on': [('ơn', 0.8), ('ôn', 0.4)],
            'ban': [('bạn', 0.9), ('ban', 0.6)],
            'toi': [('tôi', 0.9), ('tới', 0.4)],
            'la': [('là', 0.9), ('lá', 0.5), ('lạ', 0.3)],
            'hoc': [('học', 0.9), ('hóc', 0.2)],
            'sinh': [('sinh', 0.9), ('xinh', 0.3)],
            'vien': [('viên', 0.8), ('viện', 0.7)],
            'dai': [('dài', 0.7), ('đài', 0.6), ('dãi', 0.3)],
            'hoc': [('học', 0.9), ('hóc', 0.2)],
            'truong': [('trường', 0.8), ('trưởng', 0.7)]
        }

        # Get suggestions
        suggestions = simple_mappings.get(user_input, [])

        # Convert to standard format
        recommendations = []
        for word, confidence in suggestions[:max_suggestions]:
            recommendations.append((word, confidence, "simple"))

        return recommendations

    def update_context(self, selected_word: str):
        """Update context with selected word"""
        if selected_word and len(selected_word.strip()) > 0:
            # Tokenize selected word (might be multiple words)
            words = self.text_processor.tokenize(selected_word)

            # Add to context
            self.context_words.extend(words)

            # Keep only recent context
            if len(self.context_words) > self.max_context_length:
                self.context_words = self.context_words[-self.max_context_length:]

            # Update AI engine context learning
            if self.inference_engine:
                try:
                    self.inference_engine.update_context_learning(
                        selected_word, self.context_words
                    )
                except Exception as e:
                    print(f"Context learning update error: {e}")

    def clear_context(self):
        """Clear current context"""
        self.context_words = []

    def get_context(self) -> List[str]:
        """Get current context words"""
        return self.context_words.copy()

    def get_statistics(self) -> Dict:
        """Get recommender statistics"""
        base_stats = {
            'recommendations_served': self.recommendations_served,
            'avg_response_time': (
                self.total_response_time / max(self.recommendations_served, 1)
            ),
            'context_length': len(self.context_words),
            'current_context': self.context_words,
            'ai_engine_available': self.inference_engine is not None,
            'ml_module_available': ML_AVAILABLE
        }

        # Add AI engine stats if available
        if self.inference_engine:
            try:
                ai_stats = self.inference_engine.get_statistics()
                base_stats.update({
                    f'ai_{key}': value for key, value in ai_stats.items()
                })
            except Exception as e:
                base_stats['ai_stats_error'] = str(e)

        return base_stats

    def benchmark_performance(self, test_cases: List[str] = None):
        """Benchmark recommendation performance"""
        if test_cases is None:
            test_cases = [
                'xinchao', 'chao', 'cam', 'on', 'ban',
                'toi', 'la', 'hoc', 'sinh', 'vien'
            ]

        print(
            f"🔄 Benchmarking AI Recommender with {len(test_cases)} test cases...")

        total_time = 0
        successful_predictions = 0

        for test_input in test_cases:
            start_time = time.time()

            try:
                recommendations = self.recommend(test_input, max_suggestions=5)
                prediction_time = time.time() - start_time
                total_time += prediction_time

                if recommendations:
                    successful_predictions += 1
                    print(f"  ✅ {test_input}: {len(recommendations)} suggestions "
                          f"({prediction_time*1000:.1f}ms)")

                    # Show top suggestion
                    top_word, confidence, method = recommendations[0]
                    print(f"      → {top_word} ({confidence:.2%}, {method})")
                else:
                    print(f"  ❌ {test_input}: No suggestions")

            except Exception as e:
                print(f"  💥 {test_input}: Error - {e}")

        avg_time = total_time / len(test_cases)
        success_rate = successful_predictions / len(test_cases)

        print(f"\n📊 Benchmark Results:")
        print(f"  Average response time: {avg_time*1000:.2f}ms")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Predictions per second: {1/avg_time:.1f}")

        # Additional AI engine benchmark if available
        if self.inference_engine:
            try:
                self.inference_engine.benchmark_performance(
                    test_cases, iterations=10)
            except Exception as e:
                print(f"AI engine benchmark error: {e}")

    def smart_recommend(
        self,
        user_input: str,
        context: List[str] = None,
        max_suggestions: int = 8
    ) -> List[Tuple[str, float, str]]:
        """Smart recommendation with enhanced context handling"""
        # This is the main method that should be called from UI
        return self.recommend(
            user_input=user_input,
            context=context,
            max_suggestions=max_suggestions,
            use_context=True
        )

    def update_preferences(self, selected_word: str, context: List[str]):
        """Update user preferences based on selection"""
        # Update context
        self.update_context(selected_word)

        # Additional preference learning could be implemented here
        pass


# Create global instance
_ai_recommender = None


def get_ai_recommender(
    model_path: str = "checkpoints/vietnamese_non_accented_gpt_best.pth",
    data_dir: str = "ml/data"
) -> AIRecommender:
    """Get global AI recommender instance"""
    global _ai_recommender
    if _ai_recommender is None:
        _ai_recommender = AIRecommender(model_path, data_dir)
    return _ai_recommender
