"""
Inference Engine for Vietnamese Non-Accented GPT
Provides prediction interface for UI integration
"""

import torch
import os
import json
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import time

from .models.gpt_model import load_model, GPTConfig
from .tokenizer import VietnameseNonAccentedTokenizer, get_tokenizer


class VietnameseNonAccentedInference:
    """Inference engine for Vietnamese Non-Accented predictions"""

    def __init__(
        self,
        model_path: str = "checkpoints/vietnamese_non_accented_gpt_best.pth",
        data_dir: str = "ml/data",
        device: str = "auto"
    ):
        self.model_path = model_path
        self.data_dir = data_dir

        # Set device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.config = None

        # Performance tracking
        self.cache = {}
        self.prediction_count = 0
        self.cache_hits = 0

        # Load model and tokenizer
        self.load_components()

    def load_components(self):
        """Load model and tokenizer"""
        print(f"Loading inference components...")

        # Load tokenizer
        try:
            self.tokenizer = get_tokenizer(self.data_dir)
            print(
                f"Tokenizer loaded: {self.tokenizer.get_vocab_size()} vocab size")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        # Load model
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                self.model = self.model.to(self.device)
                self.model.eval()

                # Get config from checkpoint
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.config = checkpoint.get('config')

                print(f"Model loaded from {self.model_path}")
                print(f"Model parameters: {self.model.get_num_params():,}")
                print(f"Device: {self.device}")

            except Exception as e:
                print(f"Error loading model: {e}")
                print("Will use tokenizer-only predictions")
                self.model = None
        else:
            print(f"Model not found at {self.model_path}")
            print("Will use tokenizer-only predictions")
            self.model = None

    def non_accented_to_words(
        self,
        non_accented_input: str,
        context: List[str] = None,
        max_suggestions: int = 8,
        use_model: bool = True,
        temperature: float = 0.8
    ) -> List[Tuple[str, float, str]]:
        """
        Convert non-accented input to Vietnamese word suggestions

        Args:
            non_accented_input: Non-accented style input (e.g., "xinchao")
            context: Previous words for context
            max_suggestions: Maximum number of suggestions
            use_model: Whether to use the neural model
            temperature: Sampling temperature for model

        Returns:
            List of (word, confidence, method) tuples
        """
        self.prediction_count += 1

        # Create cache key
        cache_key = f"{non_accented_input}_{str(context)}_{max_suggestions}_{use_model}_{temperature}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        start_time = time.time()
        suggestions = []

        # Method 1: Tokenizer-based predictions (frequency-based)
        tokenizer_suggestions = self._get_tokenizer_suggestions(
            non_accented_input, max_suggestions
        )

        # Method 2: Model-based predictions (context-aware)
        model_suggestions = []
        if use_model and self.model is not None and context:
            model_suggestions = self._get_model_suggestions(
                non_accented_input, context, max_suggestions, temperature
            )

        # Combine and rank suggestions
        suggestions = self._combine_suggestions(
            tokenizer_suggestions,
            model_suggestions,
            max_suggestions
        )

        # Cache result
        self.cache[cache_key] = suggestions

        # Log performance
        inference_time = time.time() - start_time
        if self.prediction_count % 100 == 0:
            print(f"Inference #{self.prediction_count}: {inference_time:.3f}s, "
                  f"Cache hit rate: {self.cache_hits/self.prediction_count:.2%}")

        return suggestions

    def _get_tokenizer_suggestions(
        self,
        non_accented_input: str,
        max_suggestions: int
    ) -> List[Tuple[str, float, str]]:
        """Get suggestions from tokenizer (frequency-based)"""
        if not self.tokenizer:
            return []

        candidates = self.tokenizer.non_accented_to_candidates(
            non_accented_input, max_suggestions)

        # Convert to standard format
        suggestions = []
        for word, confidence in candidates:
            suggestions.append((word, confidence, "frequency"))

        return suggestions

    def _get_model_suggestions(
        self,
        non_accented_input: str,
        context: List[str],
        max_suggestions: int,
        temperature: float
    ) -> List[Tuple[str, float, str]]:
        """Get suggestions from neural model (context-aware)"""
        if not self.model or not self.tokenizer:
            return []

        try:
            # Prepare context
            # Last 5 words
            context_words = context[-5:] if context else ["<sos>"]

            # Encode context
            context_ids = self.tokenizer.encode_sequence(
                context_words,
                max_length=self.config.block_size - 1 if self.config else 31
            )
            context_tensor = torch.tensor([context_ids], device=self.device)

            # Get model predictions
            with torch.no_grad():
                top_indices, probs = self.model.predict_next_words(
                    context_tensor,
                    num_predictions=max_suggestions * 2,  # Get more for filtering
                    temperature=temperature
                )

            # Decode predictions and filter by pinyin
            suggestions = []
            for idx, prob in zip(top_indices[0], probs[0]):
                word = self.tokenizer.decode_token(idx.item())

                # Check if word matches non-accented input
                word_non_accented = self.tokenizer.word_to_non_accented_map(
                    word)
                if word_non_accented and self._non_accented_matches(non_accented_input, word_non_accented):
                    confidence = prob.item()
                    suggestions.append((word, confidence, "model"))

                if len(suggestions) >= max_suggestions:
                    break

            return suggestions

        except Exception as e:
            print(f"Model prediction error: {e}")
            return []

    def _non_accented_matches(self, input_non_accented: str, word_non_accented: str) -> bool:
        """Check if input non-accented matches word non-accented"""
        # Simple fuzzy matching
        return (
            input_non_accented.lower() == word_non_accented.lower() or
            word_non_accented.lower().startswith(input_non_accented.lower()) or
            input_non_accented.lower() in word_non_accented.lower()
        )

    def _combine_suggestions(
        self,
        tokenizer_suggestions: List[Tuple[str, float, str]],
        model_suggestions: List[Tuple[str, float, str]],
        max_suggestions: int
    ) -> List[Tuple[str, float, str]]:
        """Combine and rank suggestions from different methods"""

        # Create word -> (confidence, method) mapping
        word_scores = {}

        # Add tokenizer suggestions
        for word, conf, method in tokenizer_suggestions:
            word_scores[word] = (conf * 0.7, method)  # Weight tokenizer lower

        # Add model suggestions (higher priority)
        for word, conf, method in model_suggestions:
            if word in word_scores:
                # Combine scores if word exists in both
                existing_conf, existing_method = word_scores[word]
                combined_conf = conf * 0.8 + existing_conf * 0.2  # Favor model
                word_scores[word] = (combined_conf, "combined")
            else:
                # Full model confidence
                word_scores[word] = (conf * 1.0, method)

        # Sort by confidence
        sorted_words = sorted(
            word_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )

        # Return top suggestions
        suggestions = []
        for word, (conf, method) in sorted_words[:max_suggestions]:
            suggestions.append((word, conf, method))

        return suggestions

    def update_context_learning(self, selected_word: str, context: List[str]):
        """Update model's context learning (placeholder for future implementation)"""
        # This could be used for online learning or preference adaptation
        pass

    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        return {
            'prediction_count': self.prediction_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.prediction_count, 1),
            'cache_size': len(self.cache),
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'device': str(self.device),
            'vocab_size': self.tokenizer.get_vocab_size() if self.tokenizer else 0
        }

    def clear_cache(self):
        """Clear prediction cache"""
        self.cache.clear()
        print("Prediction cache cleared")

    def benchmark_performance(self, test_cases: List[str], iterations: int = 100):
        """Benchmark prediction performance"""
        print(
            f"Benchmarking with {len(test_cases)} test cases, {iterations} iterations each")

        total_time = 0
        total_predictions = 0

        for non_accented in test_cases:
            case_start = time.time()

            for _ in range(iterations):
                suggestions = self.non_accented_to_words(
                    non_accented, max_suggestions=5)
                total_predictions += len(suggestions)

            case_time = time.time() - case_start
            total_time += case_time

            print(f"  {non_accented}: {case_time:.3f}s ({iterations} iterations)")

        avg_time_per_prediction = total_time / (len(test_cases) * iterations)

        print(f"\nBenchmark Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(
            f"  Average time per prediction: {avg_time_per_prediction*1000:.2f}ms")
        print(f"  Predictions per second: {1/avg_time_per_prediction:.1f}")
        print(f"  Cache hit rate: {self.cache_hits/self.prediction_count:.2%}")


# Global inference instance
_inference_engine = None


def get_inference_engine(
    model_path: str = "checkpoints/vietnamese_non_accented_gpt_best.pth",
    data_dir: str = "ml/data",
    device: str = "auto"
) -> VietnameseNonAccentedInference:
    """Get global inference engine instance"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = VietnameseNonAccentedInference(
            model_path, data_dir, device)
    return _inference_engine


def predict_vietnamese_words(
    non_accented_input: str,
    context: List[str] = None,
    max_suggestions: int = 8
) -> List[Tuple[str, float, str]]:
    """Convenient function for getting Vietnamese word predictions"""
    engine = get_inference_engine()
    return engine.non_accented_to_words(non_accented_input, context, max_suggestions)
