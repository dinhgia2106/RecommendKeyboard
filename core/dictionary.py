"""
Dictionary module để quản lý từ điển tiếng Việt
"""

import os
from typing import List, Set, Dict, Tuple
from .text_processor import TextProcessor


class Dictionary:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.text_processor = TextProcessor()
        
        # Lưu trữ từ điển
        self.words: Set[str] = set()
        self.phrases: List[str] = []
        
        # Từ điển để tìm kiếm nhanh (key: từ không dấu, value: list từ có dấu)
        self.no_accent_to_accent: Dict[str, List[str]] = {}
        
        # Cache để tăng tốc
        self.search_cache: Dict[str, List[str]] = {}
        
        self.load_dictionary()
    
    def load_dictionary(self):
        """
        Load từ điển từ files
        """
        # Load từ đơn
        words_file = os.path.join(self.data_dir, "words.txt")
        if os.path.exists(words_file):
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.words.add(word)
                        
                        # Tạo mapping từ không dấu -> có dấu
                        no_accent = self.text_processor.remove_accents(word)
                        if no_accent not in self.no_accent_to_accent:
                            self.no_accent_to_accent[no_accent] = []
                        self.no_accent_to_accent[no_accent].append(word)
        
        # Load cụm từ
        phrases_file = os.path.join(self.data_dir, "phrases.txt")
        if os.path.exists(phrases_file):
            with open(phrases_file, 'r', encoding='utf-8') as f:
                for line in f:
                    phrase = line.strip()
                    if phrase:
                        self.phrases.append(phrase)
                        
                        # Tạo mapping cho cụm từ
                        no_accent = self.text_processor.remove_accents(phrase)
                        if no_accent not in self.no_accent_to_accent:
                            self.no_accent_to_accent[no_accent] = []
                        self.no_accent_to_accent[no_accent].append(phrase)
        
        print(f"Loaded {len(self.words)} words and {len(self.phrases)} phrases")
    
    def find_exact_match(self, query: str) -> List[str]:
        """
        Tìm kiếm chính xác (exact match)
        """
        query_no_accent = self.text_processor.remove_accents(query.lower())
        
        if query_no_accent in self.no_accent_to_accent:
            return self.no_accent_to_accent[query_no_accent].copy()
        
        return []
    
    def find_fuzzy_match(self, query: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Tìm kiếm gần đúng với threshold
        """
        query_no_accent = self.text_processor.remove_accents(query.lower())
        results = []
        
        for key, values in self.no_accent_to_accent.items():
            similarity = self.text_processor.calculate_similarity(query_no_accent, key)
            if similarity >= threshold:
                for value in values:
                    results.append((value, similarity))
        
        # Sắp xếp theo độ tương tự giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def find_prefix_match(self, query: str) -> List[str]:
        """
        Tìm kiếm theo prefix (đầu từ)
        """
        query_no_accent = self.text_processor.remove_accents(query.lower())
        results = []
        
        for key, values in self.no_accent_to_accent.items():
            if key.startswith(query_no_accent):
                results.extend(values)
        
        return results
    
    def find_contains_match(self, query: str) -> List[str]:
        """
        Tìm kiếm chứa chuỗi con
        """
        query_no_accent = self.text_processor.remove_accents(query.lower())
        results = []
        
        for key, values in self.no_accent_to_accent.items():
            if query_no_accent in key:
                results.extend(values)
        
        return results
    
    def search_comprehensive(self, query: str, max_results: int = 10) -> List[Tuple[str, float, str]]:
        """
        Tìm kiếm toàn diện với nhiều phương pháp
        Returns: List[(result, confidence, match_type)]
        """
        if not query:
            return []
        
        # Check cache
        cache_key = f"{query}_{max_results}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        results = []
        query_lower = query.lower()
        
        # 1. Exact match (confidence = 1.0)
        exact_matches = self.find_exact_match(query_lower)
        for match in exact_matches:
            results.append((match, 1.0, "exact"))
        
        # 2. Prefix match (confidence = 0.9)
        if len(results) < max_results:
            prefix_matches = self.find_prefix_match(query_lower)
            for match in prefix_matches:
                if match not in [r[0] for r in results]:
                    results.append((match, 0.9, "prefix"))
        
        # 3. Fuzzy match (confidence = similarity score)
        if len(results) < max_results:
            fuzzy_matches = self.find_fuzzy_match(query_lower, threshold=0.6)
            for match, similarity in fuzzy_matches:
                if match not in [r[0] for r in results]:
                    results.append((match, similarity, "fuzzy"))
        
        # 4. Contains match (confidence = 0.5)
        if len(results) < max_results:
            contains_matches = self.find_contains_match(query_lower)
            for match in contains_matches:
                if match not in [r[0] for r in results]:
                    results.append((match, 0.5, "contains"))
        
        # Sắp xếp theo confidence giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Giới hạn số lượng kết quả
        results = results[:max_results]
        
        # Cache kết quả
        self.search_cache[cache_key] = results
        
        return results
    
    def add_word(self, word: str):
        """
        Thêm từ mới vào từ điển
        """
        if word and word not in self.words:
            self.words.add(word)
            
            # Update mapping
            no_accent = self.text_processor.remove_accents(word)
            if no_accent not in self.no_accent_to_accent:
                self.no_accent_to_accent[no_accent] = []
            self.no_accent_to_accent[no_accent].append(word)
            
            # Clear cache
            self.search_cache.clear()
    
    def add_phrase(self, phrase: str):
        """
        Thêm cụm từ mới vào từ điển
        """
        if phrase and phrase not in self.phrases:
            self.phrases.append(phrase)
            
            # Update mapping
            no_accent = self.text_processor.remove_accents(phrase)
            if no_accent not in self.no_accent_to_accent:
                self.no_accent_to_accent[no_accent] = []
            self.no_accent_to_accent[no_accent].append(phrase)
            
            # Clear cache
            self.search_cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Lấy thống kê từ điển
        """
        return {
            "total_words": len(self.words),
            "total_phrases": len(self.phrases),
            "total_mappings": len(self.no_accent_to_accent),
            "cache_size": len(self.search_cache)
        }


if __name__ == "__main__":
    # Test Dictionary
    dict_obj = Dictionary()
    
    print("Dictionary stats:", dict_obj.get_stats())
    
    test_queries = ["xinchao", "xin", "chao", "toihoc", "moinguoi"]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = dict_obj.search_comprehensive(query, max_results=5)
        for result, confidence, match_type in results:
            print(f"  {result} (confidence: {confidence:.2f}, type: {match_type})") 