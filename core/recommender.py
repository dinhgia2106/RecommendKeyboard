"""
Recommender Engine - Core của hệ thống gợi ý
Version 2.0 - Enhanced với advanced features
"""

from typing import List, Tuple, Dict, Set
import re
import math
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
        self.fourgram_freq: Dict[Tuple[str, str, str, str], int] = {}  # New!
        
        # Advanced features
        self.word_embeddings: Dict[str, List[float]] = {}  # Placeholder for embeddings
        self.user_preferences: Dict[str, float] = {}  # User learning weights
        self.context_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # Performance metrics
        self.prediction_accuracy: Dict[str, float] = {}
        self.response_times: List[float] = []
        
        self._build_advanced_frequency_tables()
    
    def _build_advanced_frequency_tables(self):
        """
        Xây dựng bảng tần suất nâng cao với 4-gram
        """
        print("Building advanced frequency tables...")
        
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
            
            # 4-gram frequency (New!)
            for i in range(len(words) - 3):
                fourgram = (words[i], words[i + 1], words[i + 2], words[i + 3])
                self.fourgram_freq[fourgram] = self.fourgram_freq.get(fourgram, 0) + 1
        
        print(f"Enhanced frequency tables: {len(self.word_frequency)} words, "
              f"{len(self.bigram_freq)} bigrams, {len(self.trigram_freq)} trigrams, "
              f"{len(self.fourgram_freq)} 4-grams")
    
    def advanced_text_splitting(self, text: str) -> List[Tuple[List[str], float]]:
        """
        Advanced text splitting với dynamic programming và scoring
        """
        text = text.lower().strip()
        if not text:
            return []
        
        n = len(text)
        # DP table: dp[i] = [(best_words_from_i, score)]
        dp = [[] for _ in range(n + 1)]
        dp[n] = [([], 0.0)]
        
        # Backwards DP
        for i in range(n - 1, -1, -1):
            best_score = float('-inf')
            best_words = []
            
            # Try all possible word lengths from position i
            for j in range(i + 1, min(i + 15, n + 1)):  # Max word length 15
                substring = text[i:j]
                
                # Calculate word score
                word_score = self._calculate_word_score(substring)
                
                if word_score > 0:  # Valid word
                    # Get best continuation from position j
                    if dp[j]:
                        continuation_words, continuation_score = dp[j][0]
                        total_score = word_score + continuation_score
                        
                        # Apply length bonus/penalty
                        length_bonus = self._calculate_length_bonus(len(substring))
                        total_score += length_bonus
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_words = [substring] + continuation_words
            
            if best_words:
                dp[i] = [(best_words, best_score)]
        
        # Extract multiple solutions
        solutions = []
        if dp[0]:
            words, score = dp[0][0]
            solutions.append((words, score))
        
        # Add alternative splits by trying different starting positions
        for start_len in range(2, min(8, len(text))):
            if start_len < len(text):
                first_part = text[:start_len]
                remaining = text[start_len:]
                
                first_score = self._calculate_word_score(first_part)
                if first_score > 0:
                    remaining_splits = self.advanced_text_splitting(remaining)
                    for remaining_words, remaining_score in remaining_splits[:2]:
                        total_words = [first_part] + remaining_words
                        total_score = first_score + remaining_score
                        solutions.append((total_words, total_score))
        
        # Sort by score and return top 5
        solutions.sort(key=lambda x: x[1], reverse=True)
        return solutions[:5]
    
    def _calculate_word_score(self, word: str) -> float:
        """
        Tính điểm cho một từ
        """
        # Check if word exists in dictionary
        exact_matches = self.dictionary.find_exact_match(word)
        if exact_matches:
            base_score = 10.0  # High score for exact match
            
            # Add frequency bonus
            freq_bonus = math.log(self.word_frequency.get(word, 1) + 1)
            
            # Add user preference bonus
            pref_bonus = self.user_preferences.get(word, 0)
            
            return base_score + freq_bonus + pref_bonus
        
        # Check fuzzy matches
        fuzzy_matches = self.dictionary.find_fuzzy_match(word, threshold=0.8)
        if fuzzy_matches:
            best_match, similarity = fuzzy_matches[0]
            return similarity * 5.0  # Lower score for fuzzy match
        
        # Minimum score for very short words
        if len(word) <= 2:
            return 1.0
        
        return 0.0  # No match
    
    def _calculate_length_bonus(self, length: int) -> float:
        """
        Tính bonus dựa trên độ dài từ
        """
        if 3 <= length <= 6:
            return 2.0  # Sweet spot
        elif 2 <= length <= 8:
            return 1.0  # Good
        elif length >= 9:
            return -1.0  # Too long penalty
        else:
            return -2.0  # Too short penalty
    
    def enhanced_context_prediction(self, context: List[str], max_predictions: int = 5) -> List[Tuple[str, float]]:
        """
        Enhanced context prediction với 4-gram models
        """
        if not context:
            return []
        
        predictions = defaultdict(float)
        context_key = " ".join(context[-3:])  # Use last 3 words for caching
        
        # Check cache first
        if context_key in self.context_cache:
            cached = self.context_cache[context_key]
            return cached[:max_predictions]
        
        # 4-gram prediction (highest priority)
        if len(context) >= 3:
            last_three = tuple(context[-3:])
            for fourgram, freq in self.fourgram_freq.items():
                if fourgram[:3] == last_three:
                    next_word = fourgram[3]
                    score = freq / max(sum(self.fourgram_freq.values()), 1)
                    predictions[next_word] += score * 4.0  # High weight
        
        # Trigram prediction
        if len(context) >= 2:
            last_two = tuple(context[-2:])
            for trigram, freq in self.trigram_freq.items():
                if trigram[:2] == last_two:
                    next_word = trigram[2]
                    score = freq / max(sum(self.trigram_freq.values()), 1)
                    predictions[next_word] += score * 2.0  # Medium weight
        
        # Bigram prediction
        if context:
            last_word = context[-1]
            for bigram, freq in self.bigram_freq.items():
                if bigram[0] == last_word:
                    next_word = bigram[1]
                    score = freq / max(sum(self.bigram_freq.values()), 1)
                    predictions[next_word] += score * 1.0  # Lower weight
        
        # Convert to list and sort
        prediction_list = [(word, score) for word, score in predictions.items()]
        prediction_list.sort(key=lambda x: x[1], reverse=True)
        
        # Cache result
        self.context_cache[context_key] = prediction_list
        
        return prediction_list[:max_predictions]
    
    def smart_recommend(self, user_input: str, context: List[str] = None, max_suggestions: int = 8) -> List[Tuple[str, float, str]]:
        """
        Enhanced smart recommendation với multiple strategies
        """
        if not user_input:
            return []
        
        recommendations = []
        
        # Strategy 1: Dictionary lookup (exact/fuzzy)
        dict_results = self.dictionary.search_comprehensive(user_input, max_results=max_suggestions)
        for result, confidence, match_type in dict_results:
            # Boost confidence với user preferences
            user_boost = self.user_preferences.get(result, 0) * 0.1
            adjusted_confidence = min(confidence + user_boost, 1.0)
            recommendations.append((result, adjusted_confidence, f"dict_{match_type}"))
        
        # Strategy 2: Advanced text splitting
        split_results = self.advanced_text_splitting(user_input)
        for words, score in split_results:
            if len(words) > 1:  # Multi-word phrases
                phrase = " ".join(words)
                normalized_score = min(score / 20.0, 0.95)  # Normalize to [0, 0.95]
                recommendations.append((phrase, normalized_score, "advanced_split"))
        
        # Strategy 3: Context-aware predictions
        if context:
            # Try combining input with context predictions
            context_preds = self.enhanced_context_prediction(context, max_suggestions // 2)
            for word, score in context_preds:
                # Create combined suggestions
                if user_input:
                    # Try input + predicted word
                    combined1 = f"{user_input} {word}"
                    recommendations.append((combined1, score * 0.7, "context_extend"))
                    
                    # Try predicted word + input (less common but possible)
                    combined2 = f"{word} {user_input}"
                    recommendations.append((combined2, score * 0.5, "context_prepend"))
                else:
                    recommendations.append((word, score, "context_predict"))
        
        # Strategy 4: Pattern matching (new!)
        pattern_results = self._pattern_matching(user_input)
        recommendations.extend(pattern_results)
        
        # Remove duplicates and sort
        unique_recs = {}
        for text, confidence, rec_type in recommendations:
            key = text.lower().strip()
            if key not in unique_recs or confidence > unique_recs[key][1]:
                unique_recs[key] = (text, confidence, rec_type)
        
        # Convert back to list and sort by confidence
        final_recs = list(unique_recs.values())
        final_recs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply final scoring adjustments
        final_recs = self._apply_final_scoring(final_recs, user_input, context)
        
        return final_recs[:max_suggestions]
    
    def _pattern_matching(self, user_input: str) -> List[Tuple[str, float, str]]:
        """
        Pattern-based matching cho các patterns phổ biến
        """
        results = []
        input_lower = user_input.lower()
        
        # Common patterns
        patterns = {
            r'toi.*hoc': ['tôi học', 'tôi đi học', 'tôi học bài'],
            r'toi.*yeu': ['tôi yêu', 'tôi yêu em', 'tôi yêu bạn'],
            r'xin.*chao': ['xin chào', 'xin chào mọi người'],
            r'chuc.*mung': ['chúc mừng', 'chúc mừng sinh nhật', 'chúc mừng năm mới'],
            r'cam.*on': ['cảm ơn', 'cảm ơn bạn', 'cảm ơn nhiều'],
            r'xin.*loi': ['xin lỗi', 'xin lỗi nhé'],
            r'di.*hoc': ['đi học', 'tôi đi học', 'đi học thôi'],
            r'di.*choi': ['đi chơi', 'đi chơi nào', 'đi chơi thôi'],
            r'an.*com': ['ăn cơm', 'ăn cơm chưa', 'đi ăn cơm'],
            r'hom.*nay': ['hôm nay', 'hôm nay thế nào']
        }
        
        for pattern, suggestions in patterns.items():
            if re.search(pattern, input_lower):
                for suggestion in suggestions:
                    # Calculate confidence based on input similarity
                    similarity = self.text_processor.calculate_similarity(input_lower, 
                                                                        self.text_processor.remove_accents(suggestion))
                    if similarity > 0.5:
                        results.append((suggestion, similarity * 0.8, "pattern_match"))
        
        return results
    
    def _apply_final_scoring(self, recommendations: List[Tuple[str, float, str]], 
                           user_input: str, context: List[str] = None) -> List[Tuple[str, float, str]]:
        """
        Apply final scoring adjustments
        """
        scored_recs = []
        
        for text, confidence, rec_type in recommendations:
            final_score = confidence
            
            # Length-based scoring
            input_len = len(user_input)
            text_len = len(self.text_processor.remove_accents(text).replace(" ", ""))
            
            if input_len > 0:
                length_ratio = text_len / input_len
                if 1.0 <= length_ratio <= 2.0:  # Good length ratio
                    final_score *= 1.1
                elif length_ratio > 3.0:  # Too long
                    final_score *= 0.8
            
            # Frequency-based scoring
            words = self.text_processor.tokenize(text)
            if words:
                avg_freq = sum(self.word_frequency.get(word, 1) for word in words) / len(words)
                freq_boost = min(math.log(avg_freq + 1) * 0.05, 0.2)
                final_score += freq_boost
            
            # Context relevance scoring
            if context and words:
                context_relevance = self._calculate_context_relevance(words, context)
                final_score += context_relevance * 0.1
            
            # User preference scoring
            user_pref = sum(self.user_preferences.get(word, 0) for word in words) / max(len(words), 1)
            final_score += user_pref * 0.05
            
            # Cap the final score
            final_score = min(final_score, 1.0)
            
            scored_recs.append((text, final_score, rec_type))
        
        return scored_recs
    
    def _calculate_context_relevance(self, words: List[str], context: List[str]) -> float:
        """
        Calculate how relevant words are to the given context
        """
        if not context or not words:
            return 0.0
        
        relevance = 0.0
        
        # Check for bigram/trigram matches with context
        for i, word in enumerate(words):
            # Check if word appears in context
            if word in context:
                relevance += 1.0
            
            # Check bigram matches
            if context:
                last_context_word = context[-1]
                if (last_context_word, word) in self.bigram_freq:
                    relevance += 2.0
        
        return relevance / len(words)
    
    def update_user_preferences(self, chosen_text: str, context: List[str] = None):
        """
        Enhanced user learning với preference tracking
        """
        words = self.text_processor.tokenize(chosen_text)
        
        # Update word frequencies
        for word in words:
            self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
            # Update user preferences (positive reinforcement)
            self.user_preferences[word] = self.user_preferences.get(word, 0) + 0.1
        
        # Update n-gram frequencies
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
        
        for i in range(len(words) - 2):
            trigram = (words[i], words[i + 1], words[i + 2])
            self.trigram_freq[trigram] = self.trigram_freq.get(trigram, 0) + 1
        
        for i in range(len(words) - 3):
            fourgram = (words[i], words[i + 1], words[i + 2], words[i + 3])
            self.fourgram_freq[fourgram] = self.fourgram_freq.get(fourgram, 0) + 1
        
        # Update context-based learning
        if context and words:
            # Learn context transitions
            last_context_word = context[-1] if context else None
            if last_context_word:
                bigram = (last_context_word, words[0])
                self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
        
        # Add to dictionary if not exists
        if len(words) == 1:
            self.dictionary.add_word(chosen_text)
        else:
            self.dictionary.add_phrase(chosen_text)
        
        # Clear cache to force refresh
        self.context_cache.clear()
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get detailed statistics about the recommender
        """
        return {
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