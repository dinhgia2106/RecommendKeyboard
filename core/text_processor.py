"""
Minimal Text Processor for Vietnamese AI Keyboard
Simple text processing utilities
"""

import re
from typing import List, Dict, Any


class TextProcessor:
    """
    Minimal text processor for basic Vietnamese text operations
    """

    def __init__(self):
        """Initialize text processor"""
        pass

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        if not text:
            return []

        # Basic word tokenization
        words = text.strip().split()
        return [word for word in words if word]

    def remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents (basic version)"""
        if not text:
            return ""

        # Basic accent removal mapping
        accent_map = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }

        result = ""
        for char in text.lower():
            result += accent_map.get(char, char)

        return result

    def get_context(self, text: str, position: int, window_size: int = 5) -> List[str]:
        """Get context words around position"""
        words = self.tokenize(text)
        if not words or position < 0:
            return []

        start = max(0, position - window_size)
        end = min(len(words), position + window_size + 1)

        return words[start:end]

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features"""
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'char_count': 0,
                'has_accents': False
            }

        words = self.tokenize(text)
        has_accents = any(
            char in 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ' for char in text)

        return {
            'length': len(text),
            'word_count': len(words),
            'char_count': len([c for c in text if c.isalpha()]),
            'has_accents': has_accents,
            'words': words
        }
