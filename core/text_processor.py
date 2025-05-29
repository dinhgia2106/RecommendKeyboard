"""
Text Processor cho tiếng Việt
Xử lý các tác vụ cơ bản: normalize, tokenize, remove accents
"""

import re
import unicodedata
from typing import List, Tuple


class TextProcessor:
    def __init__(self):
        # Bảng chuyển đổi các ký tự tiếng Việt có dấu về không dấu (sửa lỗi)
        self.vietnamese_map = {
            'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd',
            # Thêm các ký tự khác
            'ă': 'a', 'â': 'a', 'ê': 'e', 'ô': 'o', 'ơ': 'o', 'ư': 'u'
        }
        
        # Pattern để tách từ (cải thiện)
        self.word_pattern = re.compile(r'\b[\w\u00C0-\u1EF9]+\b')
    
    def remove_accents(self, text: str) -> str:
        """
        Loại bỏ dấu tiếng Việt khỏi văn bản
        """
        result = ""
        for char in text.lower():
            if char in self.vietnamese_map:
                result += self.vietnamese_map[char]
            else:
                result += char
        return result
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize văn bản: lowercase, loại bỏ ký tự đặc biệt
        """
        # Chuyển về lowercase
        text = text.lower().strip()
        
        # Loại bỏ các ký tự không cần thiết
        text = re.sub(r'[^\w\s]', '', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tách văn bản thành các từ
        """
        normalized = self.normalize_text(text)
        words = self.word_pattern.findall(normalized)
        return [word for word in words if word]
    
    def split_without_spaces(self, text: str) -> List[str]:
        """
        Tách chuỗi không dấu, không khoảng cách thành các từ có thể
        Ví dụ: "xinchao" -> ["xin", "chao"] hoặc ["xinchao"]
        """
        text = text.lower().strip()
        if not text:
            return []
        
        # Trường hợp đơn giản: trả về toàn bộ chuỗi
        possible_splits = [text]
        
        # Thử tách thành 2 phần
        for i in range(1, len(text)):
            part1 = text[:i]
            part2 = text[i:]
            if len(part1) >= 2 and len(part2) >= 2:
                possible_splits.append([part1, part2])
        
        # Thử tách thành 3 phần
        for i in range(2, len(text)-2):
            for j in range(i+2, len(text)):
                part1 = text[:i]
                part2 = text[i:j]
                part3 = text[j:]
                if len(part1) >= 2 and len(part2) >= 2 and len(part3) >= 2:
                    possible_splits.append([part1, part2, part3])
        
        return possible_splits
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Tính độ tương tự giữa 2 chuỗi (Levenshtein distance normalized)
        """
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(str1.lower(), str2.lower())
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def extract_consonants(self, text: str) -> str:
        """
        Trích xuất phụ âm đầu của từ (để hỗ trợ tìm kiếm gần đúng)
        Ví dụ: "xin chào" -> "xch"
        """
        words = self.tokenize(text)
        consonants = ""
        
        vowels = set('aeiouàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự')
        
        for word in words:
            if word:
                # Lấy ký tự đầu tiên (thường là phụ âm)
                consonants += word[0]
        
        return consonants


if __name__ == "__main__":
    # Test basic functionality
    processor = TextProcessor()
    
    test_cases = [
        "xin chào",
        "xinchao",
        "tôi học tiếng việt",
        "toihoctiengviet"
    ]
    
    for test in test_cases:
        print(f"Original: {test}")
        print(f"No accents: {processor.remove_accents(test)}")
        print(f"Tokenized: {processor.tokenize(test)}")
        print(f"Consonants: {processor.extract_consonants(test)}")
        print("---") 