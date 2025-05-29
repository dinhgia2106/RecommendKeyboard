"""
Recommender Engine - Core của hệ thống gợi ý
"""

from typing import List, Tuple, Dict, Set
from .text_processor import TextProcessor
from .dictionary import Dictionary


class Recommender:
    def __init__(self, data_dir: str = "data"):
        self.text_processor = TextProcessor()
        self.dictionary = Dictionary(data_dir)
        
        # Frequency của các từ (để ranking)
        self.word_frequency: Dict[str, int] = {}
        
        # Context tracking để dự đoán từ tiếp theo
        self.bigram_freq: Dict[Tuple[str, str], int] = {}
        self.trigram_freq: Dict[Tuple[str, str, str], int] = {}
        
        self._build_frequency_tables()
    
    def _build_frequency_tables(self):
        """
        Xây dựng bảng tần suất từ dữ liệu có sẵn
        """
        # Xây dựng frequency từ phrases
        for phrase in self.dictionary.phrases:
            words = self.text_processor.tokenize(phrase)
            
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
        
        print(f"Built frequency tables: {len(self.word_frequency)} words, "
              f"{len(self.bigram_freq)} bigrams, {len(self.trigram_freq)} trigrams")
    
    def split_continuous_text(self, text: str) -> List[List[str]]:
        """
        Tách văn bản liên tục thành các từ có thể
        Ví dụ: "xinchaomoinguoi" -> [["xin", "chao", "moi", "nguoi"], ["xinchao", "moinguoi"], ...]
        """
        text = text.lower().strip()
        if not text:
            return []
        
        # Dynamic programming để tìm cách tách tốt nhất
        def find_best_splits(s: str, start: int = 0, memo: Dict = None) -> List[List[str]]:
            if memo is None:
                memo = {}
            
            if start == len(s):
                return [[]]
            
            if start in memo:
                return memo[start]
            
            results = []
            
            # Thử tất cả các substring từ vị trí hiện tại
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                
                # Kiểm tra xem substring có trong từ điển không
                matches = self.dictionary.find_exact_match(substring)
                if matches or len(substring) <= 3:  # Cho phép từ ngắn
                    # Đệ quy với phần còn lại
                    remaining_splits = find_best_splits(s, end, memo)
                    for split in remaining_splits:
                        # Chọn từ tốt nhất từ matches hoặc substring gốc
                        best_word = matches[0] if matches else substring
                        results.append([best_word] + split)
            
            memo[start] = results
            return results
        
        # Lấy tất cả các cách tách có thể
        all_splits = find_best_splits(text)
        
        # Lọc và sắp xếp theo độ tin cậy
        scored_splits = []
        for split in all_splits:
            score = self._calculate_split_score(split)
            scored_splits.append((split, score))
        
        # Sắp xếp theo score giảm dần
        scored_splits.sort(key=lambda x: x[1], reverse=True)
        
        # Trả về top 5 splits
        return [split for split, score in scored_splits[:5]]
    
    def _calculate_split_score(self, words: List[str]) -> float:
        """
        Tính điểm cho một cách tách từ
        """
        if not words:
            return 0.0
        
        score = 0.0
        
        # Điểm dựa trên frequency của từ
        for word in words:
            word_score = self.word_frequency.get(word, 0)
            score += word_score
        
        # Điểm dựa trên bigram
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigram_score = self.bigram_freq.get(bigram, 0)
            score += bigram_score * 2  # Bigram quan trọng hơn
        
        # Penalty cho từ quá ngắn hoặc quá dài
        avg_length = sum(len(word) for word in words) / len(words)
        if avg_length < 2:
            score *= 0.5
        elif avg_length > 8:
            score *= 0.8
        
        # Bonus cho số từ hợp lý
        if 2 <= len(words) <= 4:
            score *= 1.2
        
        return score
    
    def recommend_from_input(self, user_input: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Gợi ý từ input của user
        """
        if not user_input:
            return []
        
        suggestions = []
        
        # 1. Tìm exact matches và fuzzy matches
        search_results = self.dictionary.search_comprehensive(user_input, max_results=max_suggestions * 2)
        
        for result, confidence, match_type in search_results:
            # Adjust confidence based on match type
            adjusted_confidence = confidence
            if match_type == "exact":
                adjusted_confidence = 1.0
            elif match_type == "prefix":
                adjusted_confidence = 0.9
            elif match_type == "fuzzy":
                adjusted_confidence = confidence * 0.8
            else:  # contains
                adjusted_confidence = confidence * 0.6
            
            suggestions.append((result, adjusted_confidence))
        
        # 2. Thử tách văn bản liên tục
        splits = self.split_continuous_text(user_input)
        for split in splits:
            if len(split) > 1:  # Chỉ quan tâm đến cụm từ
                phrase = " ".join(split)
                split_score = self._calculate_split_score(split)
                # Normalize split score
                normalized_score = min(split_score / 100.0, 0.95)
                suggestions.append((phrase, normalized_score))
        
        # 3. Loại bỏ duplicate và sắp xếp
        unique_suggestions = {}
        for suggestion, confidence in suggestions:
            if suggestion not in unique_suggestions or confidence > unique_suggestions[suggestion]:
                unique_suggestions[suggestion] = confidence
        
        # Chuyển về list và sắp xếp
        final_suggestions = [(text, conf) for text, conf in unique_suggestions.items()]
        final_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return final_suggestions[:max_suggestions]
    
    def predict_next_word(self, context: List[str], max_predictions: int = 3) -> List[Tuple[str, float]]:
        """
        Dự đoán từ tiếp theo dựa trên context
        """
        if not context:
            return []
        
        predictions = {}
        
        # Sử dụng trigram nếu có đủ context
        if len(context) >= 2:
            last_two = (context[-2], context[-1])
            for trigram, freq in self.trigram_freq.items():
                if trigram[:2] == last_two:
                    next_word = trigram[2]
                    score = freq / sum(self.trigram_freq.values())
                    predictions[next_word] = predictions.get(next_word, 0) + score
        
        # Sử dụng bigram
        if context:
            last_word = context[-1]
            for bigram, freq in self.bigram_freq.items():
                if bigram[0] == last_word:
                    next_word = bigram[1]
                    score = freq / sum(self.bigram_freq.values())
                    predictions[next_word] = predictions.get(next_word, 0) + score * 0.7
        
        # Chuyển về list và sắp xếp
        prediction_list = [(word, score) for word, score in predictions.items()]
        prediction_list.sort(key=lambda x: x[1], reverse=True)
        
        return prediction_list[:max_predictions]
    
    def recommend_smart(self, user_input: str, context: List[str] = None, max_suggestions: int = 5) -> List[Tuple[str, float, str]]:
        """
        Gợi ý thông minh kết hợp nhiều phương pháp
        """
        recommendations = []
        
        # 1. Recommendations từ input
        input_recs = self.recommend_from_input(user_input, max_suggestions)
        for text, confidence in input_recs:
            recommendations.append((text, confidence, "input_based"))
        
        # 2. Predictions từ context (nếu có)
        if context:
            context_preds = self.predict_next_word(context, max_suggestions // 2)
            for word, score in context_preds:
                # Combine với input nếu có
                if user_input:
                    combined = f"{user_input} {word}"
                    recommendations.append((combined, score * 0.8, "context_pred"))
                else:
                    recommendations.append((word, score, "context_pred"))
        
        # 3. Loại bỏ duplicate và sắp xếp
        unique_recs = {}
        for text, confidence, rec_type in recommendations:
            key = text.lower().strip()
            if key not in unique_recs or confidence > unique_recs[key][1]:
                unique_recs[key] = (text, confidence, rec_type)
        
        # Chuyển về list và sắp xếp
        final_recs = list(unique_recs.values())
        final_recs.sort(key=lambda x: x[1], reverse=True)
        
        return final_recs[:max_suggestions]
    
    def update_user_choice(self, chosen_text: str, context: List[str] = None):
        """
        Cập nhật model dựa trên lựa chọn của user (learning)
        """
        # Thêm vào từ điển nếu chưa có
        words = self.text_processor.tokenize(chosen_text)
        
        # Cập nhật word frequency
        for word in words:
            self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
        
        # Cập nhật bigram frequency
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
        
        # Nếu có context, cập nhật với context
        if context and words:
            last_context_word = context[-1] if context else None
            if last_context_word:
                bigram = (last_context_word, words[0])
                self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
        
        # Thêm vào từ điển
        if len(words) == 1:
            self.dictionary.add_word(chosen_text)
        else:
            self.dictionary.add_phrase(chosen_text)


if __name__ == "__main__":
    # Test Recommender
    recommender = Recommender()
    
    test_cases = [
        "xinchao",
        "toihoc",
        "moinguoi",
        "xinchaomoinguoi",
        "chucmung"
    ]
    
    for test in test_cases:
        print(f"\nInput: '{test}'")
        recommendations = recommender.recommend_smart(test, max_suggestions=3)
        for i, (text, confidence, rec_type) in enumerate(recommendations, 1):
            print(f"  {i}. {text} (confidence: {confidence:.3f}, type: {rec_type})") 