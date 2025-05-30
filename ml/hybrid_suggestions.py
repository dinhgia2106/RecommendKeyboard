#!/usr/bin/env python3
"""
Hybrid Vietnamese Suggestion System
Combines model predictions with dictionary-based fallbacks
Ensures always-available suggestions like Chinese IME
"""

import json
import os
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import re
from difflib import SequenceMatcher

from .word_segmentation import VietnameseWordSegmenter


class VietnameseHybridSuggestions:
    """
    Hybrid suggestion system that combines:
    1. Model predictions (primary)
    2. Dictionary-based matching (fallback)
    3. Fuzzy matching (backup)
    4. Character-level suggestions (last resort)
    """

    def __init__(self, data_dir: str = "ml/data"):
        self.data_dir = data_dir
        self.segmenter = VietnameseWordSegmenter(data_dir)

        # Dictionary-based suggestions
        self.frequency_dict = {}
        self.phrase_dict = defaultdict(list)
        self.char_to_words = defaultdict(set)

        # Load additional dictionaries
        self._load_frequency_dictionary()
        self._build_phrase_dictionary()
        self._build_character_index()

        print("✅ Hybrid suggestion system initialized")

    def _load_frequency_dictionary(self):
        """Load word frequency dictionary for fallback suggestions"""
        try:
            # Load from existing mappings and build frequency
            mapping_file = os.path.join(
                self.data_dir, "non_accented_to_words.json")
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            # Build frequency based on number of variants and word length
            for non_accented, variants in mappings.items():
                for variant in variants:
                    # Estimate frequency based on word characteristics
                    base_freq = 100

                    # Common words get higher frequency
                    common_words = {
                        'là', 'có', 'một', 'này', 'đó', 'tôi', 'bạn', 'của', 'với', 'và',
                        'việt', 'nam', 'tiếng', 'người', 'nước', 'thế', 'giới', 'học',
                        'sinh', 'trường', 'đại', 'cao', 'đẳng', 'công', 'ty', 'làm',
                        'việc', 'gia', 'đình', 'nhà', 'phố', 'đường', 'thành', 'phố'
                    }

                    if variant.lower() in common_words:
                        base_freq += 1000

                    # Shorter words are generally more frequent
                    if len(variant) <= 3:
                        base_freq += 500
                    elif len(variant) <= 5:
                        base_freq += 200

                    # Multiple variants indicate common words
                    base_freq += len(variants) * 10

                    self.frequency_dict[variant] = base_freq

            print(
                f"📊 Loaded frequency data for {len(self.frequency_dict):,} words")

        except Exception as e:
            print(f"⚠️ Could not load frequency data: {e}")
            self.frequency_dict = {}

    def _build_phrase_dictionary(self):
        """Build phrase dictionary for multi-word suggestions"""
        # Common Vietnamese phrases and patterns
        common_phrases = [
            # Greetings
            ("xin chào", "chào anh", "chào chị", "chào em"),
            ("cảm ơn", "cám ơn nhiều", "cảm ơn bạn"),
            ("xin lỗi", "xin lỗi nhé", "xin lỗi anh", "xin lỗi chị"),

            # Country/Language
            ("tiếng việt", "tiếng anh", "tiếng trung", "tiếng nhật"),
            ("việt nam", "trung quốc", "nhật bản", "hàn quốc"),
            ("hoa kỳ", "mỹ quốc", "thái lan", "singapore"),

            # Education
            ("học sinh", "sinh viên", "giáo viên", "thầy cô"),
            ("trường học", "đại học", "cao đẳng", "trung học"),
            ("bài học", "môn học", "học tập", "giáo dục"),

            # Work
            ("làm việc", "công việc", "việc làm", "công ty"),
            ("văn phòng", "nhân viên", "quản lý", "giám đốc"),

            # Family
            ("gia đình", "ba mẹ", "anh chị", "ông bà"),
            ("con cái", "vợ chồng", "họ hàng", "bạn bè"),

            # Places
            ("thành phố", "nông thôn", "thị trấn", "quận huyện"),
            ("đường phố", "con đường", "ngã tư", "công viên"),

            # Time
            ("hôm nay", "ngày mai", "hôm qua", "tuần này"),
            ("tháng này", "năm nay", "mùa hè", "mùa đông"),

            # Common actions
            ("đi học", "về nhà", "đi làm", "đi chơi"),
            ("ăn cơm", "uống nước", "ngủ nghỉ", "thể thao")
        ]

        # Build phrase index
        for phrase_group in common_phrases:
            for phrase in phrase_group:
                words = phrase.split()
                for i in range(len(words)):
                    prefix = " ".join(words[:i+1])
                    if i < len(words) - 1:
                        next_word = words[i+1]
                        self.phrase_dict[prefix.lower()].append(next_word)

        print(
            f"📚 Built phrase dictionary with {len(self.phrase_dict)} entries")

    def _build_character_index(self):
        """Build character-based index for fuzzy matching"""
        for word in self.frequency_dict.keys():
            for char in word:
                if char.isalpha():
                    self.char_to_words[char.lower()].add(word)

        print(
            f"🔤 Built character index for {len(self.char_to_words)} characters")

    def get_suggestions(self, input_text: str, max_suggestions: int = 10, model_predictions: List[Tuple[str, float]] = None) -> List[Dict]:
        """
        Get hybrid suggestions combining model + dictionary + fuzzy matching

        Args:
            input_text: User input (can be segmented or raw)
            max_suggestions: Maximum number of suggestions
            model_predictions: Optional model predictions [(word, confidence)]

        Returns:
            List of suggestion dictionaries with metadata
        """
        suggestions = []
        used_words = set()
        input_lower = input_text.lower().strip()

        # 0. FIRST - Try segmentation for longer inputs (IME-style)
        if len(input_lower) >= 4:  # Only for longer inputs
            segmented = self.segmenter.segment_text(input_text)
            if segmented != input_text and ' ' in segmented:  # Successfully segmented
                suggestions.append({
                    'word': segmented,
                    'confidence': 0.98,
                    'source': 'segmentation',
                    'score': 2000  # Highest priority
                })
                used_words.add(segmented)

        # 1. Check direct mappings from word segmentation
        direct_mappings = self._get_direct_mappings(input_text)
        for suggestion in direct_mappings[:max_suggestions//3]:
            word = suggestion['word']
            if word not in used_words:
                suggestions.append(suggestion)
                used_words.add(word)

        # 2. Model predictions (high priority)
        if model_predictions:
            for word, confidence in model_predictions[:max_suggestions//2]:
                if word not in used_words and word.strip():
                    suggestions.append({
                        'word': word,
                        'confidence': confidence,
                        'source': 'model',
                        'score': confidence * 100
                    })
                    used_words.add(word)

        # 3. Dictionary-based exact matches
        dict_suggestions = self._get_dictionary_suggestions(input_text)
        for suggestion in dict_suggestions[:max_suggestions//3]:
            word = suggestion['word']
            if word not in used_words:
                suggestions.append(suggestion)
                used_words.add(word)

        # 4. Phrase-based suggestions
        phrase_suggestions = self._get_phrase_suggestions(input_text)
        for suggestion in phrase_suggestions[:max_suggestions//4]:
            word = suggestion['word']
            if word not in used_words:
                suggestions.append(suggestion)
                used_words.add(word)

        # 5. Fuzzy matching (fallback) - only if we have some suggestions already
        if len(suggestions) < max_suggestions and len(suggestions) > 0:
            fuzzy_suggestions = self._get_fuzzy_suggestions(
                input_text, max_suggestions - len(suggestions))
            for suggestion in fuzzy_suggestions:
                word = suggestion['word']
                if word not in used_words:
                    suggestions.append(suggestion)
                    used_words.add(word)

        # 6. Character-based suggestions (last resort)
        if len(suggestions) < max_suggestions:
            char_suggestions = self._get_character_suggestions(
                input_text, max_suggestions - len(suggestions))
            for suggestion in char_suggestions:
                word = suggestion['word']
                if word not in used_words:
                    suggestions.append(suggestion)
                    used_words.add(word)

        # Sort by score and return
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:max_suggestions]

    def _get_direct_mappings(self, input_text: str) -> List[Dict]:
        """Get direct mappings from word segmentation data"""
        suggestions = []
        input_lower = input_text.lower().strip()

        # Basic single word mappings that might be missing
        basic_mappings = {
            'toi': ['tôi'],
            'ban': ['bạn'],
            'anh': ['anh'],
            'em': ['em'],
            'chi': ['chị'],
            'la': ['là'],
            'co': ['có'],
            'mot': ['một'],
            'nay': ['này', 'nay'],
            'do': ['đó'],
            'voi': ['với'],
            'va': ['và'],
            'cua': ['của'],
            'den': ['đến'],
            'tu': ['từ'],
            'trong': ['trong'],
            'ngoai': ['ngoài'],
            'tren': ['trên'],
            'duoi': ['dưới'],
            'sau': ['sau'],
            'truoc': ['trước'],
            'giua': ['giữa'],
            'ben': ['bên'],
            'gan': ['gần'],
            'xa': ['xa'],
            'lon': ['lớn'],
            'nho': ['nhỏ'],
            'dep': ['đẹp'],
            'xau': ['xấu'],
            'tot': ['tốt'],
            'xin': ['xin'],
            'cam': ['cảm'],
            'on': ['ơn'],
            'loi': ['lỗi'],
            'yeu': ['yêu'],
            'thuong': ['thương'],
            'hoc': ['học'],
            'sinh': ['sinh'],
            'lam': ['làm'],
            'viec': ['việc']
        }

        # Check basic mappings first
        if input_lower in basic_mappings:
            for word in basic_mappings[input_lower]:
                suggestions.append({
                    'word': word,
                    'confidence': 0.90,
                    'source': 'basic_mapping',
                    'score': 1200
                })

        # Check for partial matches in basic mappings
        for key, words in basic_mappings.items():
            if key.startswith(input_lower) and len(input_lower) >= 2 and key != input_lower:
                for word in words:
                    suggestions.append({
                        'word': word,
                        'confidence': 0.75,
                        'source': 'basic_partial',
                        'score': 900
                    })

        # Load the mappings from file
        try:
            mapping_file = os.path.join(
                self.data_dir, "non_accented_to_words.json")
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            # Check for exact matches and prefix matches
            for non_accented, variants in mappings.items():
                # Exact match
                if non_accented == input_lower:
                    for variant in variants:
                        suggestions.append({
                            'word': variant,
                            'confidence': 0.95,
                            'source': 'mapping_exact',
                            'score': 1000 + len(variant)
                        })

                # Prefix match (input is prefix of mapping)
                elif non_accented.startswith(input_lower) and len(input_lower) >= 2:
                    for variant in variants:
                        suggestions.append({
                            'word': variant,
                            'confidence': 0.85,
                            'source': 'mapping_prefix',
                            'score': 800 + len(variant)
                        })

        except Exception as e:
            print(f"Could not load direct mappings: {e}")

        # Remove duplicates while keeping highest scored
        seen = set()
        unique_suggestions = []
        for suggestion in sorted(suggestions, key=lambda x: x['score'], reverse=True):
            if suggestion['word'] not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion['word'])

        return unique_suggestions[:10]

    def _get_dictionary_suggestions(self, input_text: str) -> List[Dict]:
        """Get suggestions from frequency dictionary"""
        suggestions = []
        input_lower = input_text.lower().strip()

        if not input_lower:
            return []

        # Direct prefix matches (highest priority)
        for word, freq in self.frequency_dict.items():
            word_lower = word.lower()

            # Remove accents from word for comparison
            word_no_accent = self._remove_accents(word_lower)

            # IMPORTANT: Only suggest words that actually start with the user input
            # or have non-accented version that starts with user input
            match_found = False

            # Check if non-accented word starts with input
            if word_no_accent.startswith(input_lower):
                match_found = True
                # Higher score for non-accented matches
                score = freq + (300 - len(word))
                confidence = min(0.95, freq / 500)

            # Check if original word starts with input
            elif word_lower.startswith(input_lower):
                match_found = True
                score = freq + (250 - len(word))
                confidence = min(0.9, freq / 600)

            # Only add if we found a valid match AND the word is longer than input
            if match_found and len(word) > len(input_lower):
                suggestions.append({
                    'word': word,
                    'confidence': confidence,
                    'source': 'dictionary',
                    'score': score
                })

        # Sort by score and return only relevant matches
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:15]

    def _remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents for matching"""
        accent_map = {
            'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }

        result = ""
        for char in text:
            result += accent_map.get(char, char)
        return result

    def _get_phrase_suggestions(self, input_text: str) -> List[Dict]:
        """Get phrase-based suggestions"""
        suggestions = []
        input_lower = input_text.lower().strip()

        # Check for phrase matches
        for phrase_prefix, next_words in self.phrase_dict.items():
            if phrase_prefix.endswith(input_lower) or input_lower in phrase_prefix:
                for word in next_words:
                    score = 800 + len(phrase_prefix) * \
                        10  # High score for phrases
                    suggestions.append({
                        'word': word,
                        'confidence': 0.85,
                        'source': 'phrase',
                        'score': score
                    })

        return suggestions[:10]

    def _get_fuzzy_suggestions(self, input_text: str, max_count: int) -> List[Dict]:
        """Get fuzzy matching suggestions"""
        suggestions = []
        input_lower = input_text.lower().strip()

        if len(input_lower) < 2:
            return []

        # Find words with similar characters, but prioritize starting matches
        candidates = set()

        # First, get words that start with same characters (higher priority)
        for word, freq in self.frequency_dict.items():
            word_lower = word.lower()
            word_no_accent = self._remove_accents(word_lower)

            # Check if word contains significant portion of input characters
            if len(input_lower) >= 2:
                # Must start with at least first character
                if (word_lower.startswith(input_lower[0]) or
                        word_no_accent.startswith(input_lower[0])):

                    # Calculate how many input characters are in the word (in order)
                    chars_matched = 0
                    word_chars = word_no_accent
                    input_chars = input_lower

                    i = 0
                    for char in input_chars:
                        pos = word_chars.find(char, i)
                        if pos != -1:
                            chars_matched += 1
                            i = pos + 1

                    # Only consider if significant portion matches
                    if chars_matched >= min(3, len(input_lower) - 1):
                        candidates.add(word)

        # Calculate similarity for candidates
        for word in candidates:
            word_lower = word.lower()
            word_no_accent = self._remove_accents(word_lower)

            # Calculate similarity with both original and non-accented
            similarity1 = SequenceMatcher(
                None, input_lower, word_lower).ratio()
            similarity2 = SequenceMatcher(
                None, input_lower, word_no_accent).ratio()
            similarity = max(similarity1, similarity2)

            # Higher threshold to avoid irrelevant suggestions
            if similarity > 0.4:  # Increased threshold
                # Boost score if word starts with input
                start_bonus = 200 if (word_lower.startswith(input_lower) or
                                      word_no_accent.startswith(input_lower)) else 0

                score = similarity * 400 + \
                    self.frequency_dict.get(word, 50) + start_bonus
                suggestions.append({
                    'word': word,
                    'confidence': similarity * 0.6,
                    'source': 'fuzzy',
                    'score': score
                })

        # Sort and return
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:max_count]

    def _get_character_suggestions(self, input_text: str, max_count: int) -> List[Dict]:
        """Get character-based suggestions as last resort"""
        suggestions = []
        input_lower = input_text.lower().strip()

        if not input_lower:
            return []

        # Get words starting with first character(s) of input
        first_char = input_lower[0]

        # If input is longer, try to match more characters
        target_chars = input_lower[:min(2, len(input_lower))]

        # Find words from frequency dict that start with target characters
        candidates = []
        for word, freq in self.frequency_dict.items():
            word_lower = word.lower()
            word_no_accent = self._remove_accents(word_lower)

            # Check if word starts with input characters
            if (word_lower.startswith(target_chars) or
                    word_no_accent.startswith(target_chars)):
                candidates.append((word, freq))

        # Sort by frequency and get top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        for word, freq in candidates[:max_count * 2]:
            # Lower confidence since these are last resort
            score = freq + 30
            suggestions.append({
                'word': word,
                'confidence': 0.3,
                'source': 'character',
                'score': score
            })

        # Sort and return
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:max_count]

    def suggest_with_segmentation(self, long_input: str, max_suggestions: int = 10, model_predictions: List[Tuple[str, float]] = None) -> Dict:
        """
        Get suggestions for long input with automatic segmentation

        Returns:
            {
                'original': original input,
                'segmented': segmented input,
                'suggestions': list of suggestions,
                'segmentation_confidence': confidence of segmentation
            }
        """
        # Segment the input
        segmentation_result = self.segmenter.segment_with_details(long_input)
        segmented_text = segmentation_result['result']

        # Get suggestions for the segmented text
        suggestions = self.get_suggestions(
            segmented_text, max_suggestions, model_predictions)

        return {
            'original': long_input,
            'segmented': segmented_text,
            'suggestions': suggestions,
            'segmentation_confidence': segmentation_result['confidence'],
            'method': 'hybrid_with_segmentation'
        }


def main():
    """Test the hybrid suggestion system"""
    print("🚀 Testing Hybrid Vietnamese Suggestion System")
    print("=" * 70)

    # Initialize system
    hybrid = VietnameseHybridSuggestions()

    # Test cases including words not in training data
    test_cases = [
        "tieng",      # Should suggest "tiếng" (from dictionary)
        "viet",       # Should suggest "việt"
        "tiengviet",  # Should segment and suggest
        "homnay",     # Should suggest phrase continuations
        "xin",        # Should suggest "chào"
        "cam",        # Should suggest "ơn"
        "hello",      # Should fallback to fuzzy/character matching
        "abc",        # Should provide fallback suggestions
    ]

    print("\n🧪 Testing Suggestions:")

    for input_text in test_cases:
        print(f"\n📝 Input: '{input_text}'")

        # Test regular suggestions
        suggestions = hybrid.get_suggestions(input_text, max_suggestions=5)

        print(f"💡 Suggestions ({len(suggestions)}):")
        for i, sugg in enumerate(suggestions[:5], 1):
            print(
                f"   {i}. {sugg['word']} ({sugg['source']}, {sugg['confidence']:.2%})")

        # Test with segmentation for longer inputs
        if len(input_text) > 6:
            seg_result = hybrid.suggest_with_segmentation(input_text)
            print(f"🔄 Segmented: '{seg_result['segmented']}'")


if __name__ == "__main__":
    main()
