"""
Recommender Engine - Core của hệ thống gợi ý
Version 2.1 - Production Ready với Performance Optimization
"""

from typing import List, Tuple, Dict, Set
import re
import math
import time
from collections import defaultdict
from .text_processor import TextProcessor
from .dictionary import Dictionary


class AdvancedRecommender:
    def __init__(self, data_dir: str = "data"):
        self.text_processor = TextProcessor()
        self.dictionary = Dictionary(data_dir)
        
        # Enhanced frequency tracking
        self.word_frequency: Dict[str, int] = {}
        self.bigram_freq: Dict[Tuple[str, str], int] = {}
        self.trigram_freq: Dict[Tuple[str, str, str], int] = {}
        self.fourgram_freq: Dict[Tuple[str, str, str, str], int] = {}
        
        # Performance optimization features
        self.recommendation_cache: Dict[str, List[Tuple[str, float, str]]] = {}
        self.split_cache: Dict[str, List[Tuple[List[str], float]]] = {}
        self.last_recommendation_time: Dict[str, float] = {}
        
        # Advanced features
        self.word_embeddings: Dict[str, List[float]] = {}
        self.user_preferences: Dict[str, float] = {}
        self.context_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # Performance settings
        self.max_cache_size = 1000
        self.cache_timeout = 300  # 5 minutes
        self.min_debounce_time = 0.1  # 100ms debounce
        self.max_processing_time = 0.03  # 30ms max processing time
        
        # Performance metrics
        self.prediction_accuracy: Dict[str, float] = {}
        self.response_times: List[float] = []
        
        self._build_advanced_frequency_tables()
    
    def _build_advanced_frequency_tables(self):
        """
        Xây dựng bảng tần suất nâng cao với optimization
        """
        print("Building optimized frequency tables...")
        start_time = time.time()
        
        # Reset counters
        self.word_frequency.clear()
        self.bigram_freq.clear()
        self.trigram_freq.clear()
        self.fourgram_freq.clear()
        
        all_texts = list(self.dictionary.phrases) + list(self.dictionary.words)
        
        for text in all_texts:
            words = self.text_processor.tokenize(text)
            if not words:
                continue
            
            # Unigram frequency
            for word in words:
                self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
            
            # Bigram frequency
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
            
            # Trigram frequency
            for i in range(len(words) - 2):
                trigram = (words[i], words[i + 1], words[i + 2])
                self.trigram_freq[trigram] = self.trigram_freq.get(trigram, 0) + 1
            
            # 4-gram frequency (limited for performance)
            if len(words) <= 10:  # Only for shorter phrases
                for i in range(len(words) - 3):
                    fourgram = (words[i], words[i + 1], words[i + 2], words[i + 3])
                    self.fourgram_freq[fourgram] = self.fourgram_freq.get(fourgram, 0) + 1
        
        build_time = time.time() - start_time
        print(f"Optimized frequency tables built in {build_time:.2f}s: {len(self.word_frequency)} words, "
              f"{len(self.bigram_freq)} bigrams, {len(self.trigram_freq)} trigrams, "
              f"{len(self.fourgram_freq)} 4-grams")
    
    def _clean_old_cache(self):
        """
        Dọn dẹp cache cũ để tối ưu memory
        """
        current_time = time.time()
        
        # Clean recommendation cache
        if len(self.recommendation_cache) > self.max_cache_size:
            # Remove oldest 20% of cache
            to_remove = len(self.recommendation_cache) // 5
            oldest_keys = sorted(self.last_recommendation_time.items(), key=lambda x: x[1])[:to_remove]
            for key, _ in oldest_keys:
                self.recommendation_cache.pop(key, None)
                self.last_recommendation_time.pop(key, None)
        
        # Clean split cache
        if len(self.split_cache) > self.max_cache_size:
            # Keep only most recent half
            recent_keys = list(self.split_cache.keys())[-self.max_cache_size//2:]
            new_split_cache = {k: self.split_cache[k] for k in recent_keys}
            self.split_cache = new_split_cache
        
        # Clean context cache
        if len(self.context_cache) > self.max_cache_size:
            # Keep only most recent half
            recent_keys = list(self.context_cache.keys())[-self.max_cache_size//2:]
            new_context_cache = {k: self.context_cache[k] for k in recent_keys}
            self.context_cache = new_context_cache
    
    def advanced_text_splitting_optimized(self, text: str) -> List[Tuple[List[str], float]]:
        """
        Optimized advanced text splitting với timeout và caching
        """
        # Check cache first
        if text in self.split_cache:
            return self.split_cache[text]
        
        start_time = time.time()
        text = text.lower().strip()
        if not text:
            return []
        
        # Limit processing time for long texts
        if len(text) > 20:  # For very long texts, use simpler approach
            return self._simple_text_splitting(text)
        
        n = len(text)
        # DP table with timeout
        dp = [[] for _ in range(n + 1)]
        dp[n] = [([], 0.0)]
        
        # Backwards DP with timeout
        for i in range(n - 1, -1, -1):
            # Check timeout
            if time.time() - start_time > self.max_processing_time:
                break
                
            best_score = float('-inf')
            best_words = []
            
            # Try reasonable word lengths (optimized range)
            max_word_len = min(10, n - i)  # Reduced from 15
            for j in range(i + 1, i + max_word_len + 1):
                substring = text[i:j]
                
                # Quick word score calculation
                word_score = self._calculate_word_score_fast(substring)
                
                if word_score > 0:
                    if dp[j]:
                        continuation_words, continuation_score = dp[j][0]
                        total_score = word_score + continuation_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_words = [substring] + continuation_words
            
            if best_words:
                dp[i] = [(best_words, best_score)]
        
        # Extract solutions (limited for performance)
        solutions = []
        if dp[0]:
            words, score = dp[0][0]
            solutions.append((words, score))
        
        # Cache result
        self.split_cache[text] = solutions
        self._clean_old_cache()
        
        return solutions
    
    def _simple_text_splitting(self, text: str) -> List[Tuple[List[str], float]]:
        """
        Simple text splitting for long texts
        """
        # Try common breakpoints
        common_splits = []
        
        # Try splitting every 3-6 characters
        for split_len in [3, 4, 5, 6]:
            words = []
            for i in range(0, len(text), split_len):
                words.append(text[i:i+split_len])
            if words:
                score = sum(self._calculate_word_score_fast(w) for w in words)
                common_splits.append((words, score))
        
        return common_splits[:3] if common_splits else [(list(text), 1.0)]
    
    def _calculate_word_score_fast(self, word: str) -> float:
        """
        Fast word score calculation
        """
        # Quick lookup in word frequency
        if word in self.word_frequency:
            return 10.0 + math.log(self.word_frequency[word] + 1)
        
        # Check if it's a substring of known words (quick check)
        for known_word in list(self.word_frequency.keys())[:100]:  # Limit check
            if word in known_word:
                return 2.0
        
        # Default score based on length
        if 3 <= len(word) <= 6:
            return 1.0
        elif 2 <= len(word) <= 8:
            return 0.5
        else:
            return 0.1
    
    def smart_recommend_optimized(self, user_input: str, context: List[str] = None, max_suggestions: int = 8) -> List[Tuple[str, float, str]]:
        """
        Optimized smart recommendation với caching và debounce
        """
        if not user_input:
            return []
        
        # Create cache key
        context_key = "_".join(context) if context else ""
        cache_key = f"{user_input}_{context_key}_{max_suggestions}"
        
        # Check cache first
        current_time = time.time()
        if cache_key in self.recommendation_cache:
            # Check if not too old
            if cache_key in self.last_recommendation_time:
                if current_time - self.last_recommendation_time[cache_key] < self.cache_timeout:
                    return self.recommendation_cache[cache_key]
        
        start_time = time.time()
        recommendations = []
        
        try:
            # Strategy 1: Dictionary lookup (optimized)
            dict_results = self.dictionary.search_comprehensive(user_input, max_results=max_suggestions)
            for result, confidence, match_type in dict_results[:5]:  # Limit to top 5
                user_boost = self.user_preferences.get(result, 0) * 0.1
                adjusted_confidence = min(confidence + user_boost, 1.0)
                recommendations.append((result, adjusted_confidence, f"dict_{match_type}"))
            
            # Strategy 2: Advanced text splitting (with timeout)
            if len(user_input) >= 4:  # Only for longer inputs
                split_results = self.advanced_text_splitting_optimized(user_input)
                for words, score in split_results[:3]:  # Limit to top 3
                    if len(words) > 1:
                        phrase = " ".join(words)
                        normalized_score = min(score / 20.0, 0.95)
                        recommendations.append((phrase, normalized_score, "advanced_split"))
            
            # Strategy 3: Context predictions (optimized)
            if context and len(recommendations) < max_suggestions:
                context_preds = self.enhanced_context_prediction_fast(context, max_suggestions // 3)
                for word, score in context_preds:
                    if user_input:
                        combined = f"{user_input} {word}"
                        recommendations.append((combined, score * 0.7, "context_extend"))
            
            # Strategy 4: Pattern matching (quick patterns only)
            if len(user_input) >= 5:
                pattern_results = self._pattern_matching_fast(user_input)
                recommendations.extend(pattern_results[:2])  # Limit to top 2
            
        except Exception as e:
            print(f"Error in recommendation: {e}")
            # Fallback to simple dictionary search
            dict_results = self.dictionary.search_comprehensive(user_input, max_results=3)
            for result, confidence, match_type in dict_results:
                recommendations.append((result, confidence, f"dict_{match_type}"))
        
        # Remove duplicates and sort (optimized)
        unique_recs = {}
        for text, confidence, rec_type in recommendations:
            key = text.lower().strip()
            if key not in unique_recs or confidence > unique_recs[key][1]:
                unique_recs[key] = (text, confidence, rec_type)
        
        final_recs = list(unique_recs.values())
        final_recs.sort(key=lambda x: x[1], reverse=True)
        final_recs = final_recs[:max_suggestions]
        
        # Cache result
        self.recommendation_cache[cache_key] = final_recs
        self.last_recommendation_time[cache_key] = current_time
        
        # Track performance
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        if len(self.response_times) > 100:  # Keep last 100 measurements
            self.response_times = self.response_times[-100:]
        
        return final_recs
    
    def enhanced_context_prediction_fast(self, context: List[str], max_predictions: int = 5) -> List[Tuple[str, float]]:
        """
        Fast context prediction với caching
        """
        if not context:
            return []
        
        context_key = " ".join(context[-2:])  # Use last 2 words only for speed
        
        # Check cache first
        if context_key in self.context_cache:
            cached = self.context_cache[context_key]
            return cached[:max_predictions]
        
        predictions = defaultdict(float)
        
        # Simplified prediction (faster)
        # Use only bigram for speed
        if context:
            last_word = context[-1]
            for bigram, freq in list(self.bigram_freq.items())[:500]:  # Limit iterations
                if bigram[0] == last_word:
                    next_word = bigram[1]
                    score = freq / max(sum(self.bigram_freq.values()), 1)
                    predictions[next_word] += score
        
        # Convert to list and sort
        prediction_list = [(word, score) for word, score in predictions.items()]
        prediction_list.sort(key=lambda x: x[1], reverse=True)
        
        # Cache result
        self.context_cache[context_key] = prediction_list
        
        return prediction_list[:max_predictions]
    
    def _pattern_matching_fast(self, user_input: str) -> List[Tuple[str, float, str]]:
        """
        Fast pattern matching với limited patterns
        """
        results = []
        input_lower = user_input.lower()
        
        # Quick patterns only
        quick_patterns = {
            r'toi.*hoc': ['tôi học'],
            r'xin.*chao': ['xin chào'],
            r'chuc.*mung': ['chúc mừng'],
            r'cam.*on': ['cảm ơn']
        }
        
        for pattern, suggestions in quick_patterns.items():
            if re.search(pattern, input_lower):
                for suggestion in suggestions:
                    similarity = 0.8  # Fixed similarity for speed
                    results.append((suggestion, similarity, "pattern_match"))
                break  # Only match first pattern for speed
        
        return results
    
    # Backward compatibility methods
    def smart_recommend(self, user_input: str, context: List[str] = None, max_suggestions: int = 8) -> List[Tuple[str, float, str]]:
        """Main recommendation method - optimized"""
        return self.smart_recommend_optimized(user_input, context, max_suggestions)
    
    def advanced_text_splitting(self, text: str) -> List[Tuple[List[str], float]]:
        """Backward compatibility method"""
        return self.advanced_text_splitting_optimized(text)
    
    def enhanced_context_prediction(self, context: List[str], max_predictions: int = 5) -> List[Tuple[str, float]]:
        """Backward compatibility method"""
        return self.enhanced_context_prediction_fast(context, max_predictions)
    
    def update_user_preferences(self, chosen_text: str, context: List[str] = None):
        """
        Optimized user learning
        """
        words = self.text_processor.tokenize(chosen_text)
        
        # Quick updates only
        for word in words:
            self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
            self.user_preferences[word] = self.user_preferences.get(word, 0) + 0.1
        
        # Update only bigrams for speed
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
        
        # Clear relevant caches
        keys_to_remove = [k for k in self.recommendation_cache.keys() if chosen_text.lower() in k.lower()]
        for key in keys_to_remove[:10]:  # Limit cleanup
            self.recommendation_cache.pop(key, None)
            self.last_recommendation_time.pop(key, None)
    
    def get_performance_stats(self) -> Dict[str, any]:
        """
        Get performance statistics
        """
        avg_response_time = sum(self.response_times) / max(len(self.response_times), 1)
        
        return {
            "avg_response_time_ms": avg_response_time * 1000,
            "cache_sizes": {
                "recommendations": len(self.recommendation_cache),
                "splits": len(self.split_cache),
                "context": len(self.context_cache)
            },
            "performance_settings": {
                "max_processing_time_ms": self.max_processing_time * 1000,
                "min_debounce_time_ms": self.min_debounce_time * 1000,
                "cache_timeout_s": self.cache_timeout
            }
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Enhanced statistics with performance data
        """
        base_stats = {
            "word_count": len(self.word_frequency),
            "bigram_count": len(self.bigram_freq),
            "trigram_count": len(self.trigram_freq),
            "fourgram_count": len(self.fourgram_freq),
            "user_preferences": len(self.user_preferences),
            "cache_size": len(self.context_cache),
            "dictionary_stats": self.dictionary.get_stats(),
            "top_words": sorted(self.word_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_preferences": sorted(self.user_preferences.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        # Add performance stats
        base_stats.update(self.get_performance_stats())
        
        return base_stats


# Backward compatibility - create alias
class Recommender(AdvancedRecommender):
    """Backward compatibility alias"""
    
    def recommend_smart(self, user_input: str, context: List[str] = None, max_suggestions: int = 5) -> List[Tuple[str, float, str]]:
        """Backward compatibility method"""
        return self.smart_recommend(user_input, context, max_suggestions)
    
    def update_user_choice(self, chosen_text: str, context: List[str] = None):
        """Backward compatibility method"""
        return self.update_user_preferences(chosen_text, context)


if __name__ == "__main__":
    # Test Advanced Recommender
    recommender = AdvancedRecommender()
    
    print("=== ADVANCED RECOMMENDER TEST ===")
    print(f"Statistics: {recommender.get_statistics()}")
    
    test_cases = [
        "xinchao",
        "toihoctiengviet",
        "anhyeuem",
        "chucmungnamoi",
        "camonnhieu"
    ]
    
    for test in test_cases:
        print(f"\nInput: '{test}'")
        recommendations = recommender.smart_recommend(test, max_suggestions=3)
        for i, (text, confidence, rec_type) in enumerate(recommendations, 1):
            print(f"  {i}. {text} (confidence: {confidence:.3f}, type: {rec_type})")
            
        # Simulate user choosing first recommendation
        if recommendations:
            chosen = recommendations[0][0]
            recommender.update_user_preferences(chosen)
            print(f"  → User chose: '{chosen}'")
    
    print(f"\nFinal statistics: {recommender.get_statistics()}") 