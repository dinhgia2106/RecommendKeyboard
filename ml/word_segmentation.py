#!/usr/bin/env python3
"""
Vietnamese Word Segmentation for Non-accented Text
Handles long input strings like "toimangdenchocacbanbogoamoi"
"""

import json
import os
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import re


class VietnameseWordSegmenter:
    """Vietnamese word segmentation for keyboard input"""

    def __init__(self, data_dir: str = "ml/data"):
        self.data_dir = data_dir
        self.word_mappings = {}
        self.vocabulary = set()
        self.max_word_length = 0

        # Basic word mappings for common Vietnamese words
        self.basic_words = {
            'toi': 'tôi',
            'la': 'là',
            'ban': 'bạn',
            'anh': 'anh',
            'em': 'em',
            'chi': 'chị',
            'co': 'có',
            'mot': 'một',
            'nay': 'này',
            'do': 'đó',
            'voi': 'với',
            'va': 'và',
            'cua': 'của',
            'den': 'đến',
            'tu': 'từ',
            'trong': 'trong',
            'ngoai': 'ngoài',
            'tren': 'trên',
            'duoi': 'dưới',
            'xin': 'xin',
            'cam': 'cảm',
            'on': 'ơn',
            'chao': 'chào',
            'yeu': 'yêu',
            'hoc': 'học',
            'sinh': 'sinh',
            'lam': 'làm',
            'viec': 'việc',
            'dep': 'đẹp',
            'tot': 'tốt',
            'to': 'tớ',  # Added 'to' -> 'tớ'
            'viet': 'việt',  # Added
            'nam': 'nam',    # Added
            'tieng': 'tiếng',  # Added
            'di': 'đi',      # Added
            've': 'về',      # Added
            'nha': 'nhà',    # Added
            'hoc': 'học',    # Added
            'truong': 'trường',  # Added
            'moi': 'mới',    # Added
            'cu': 'cũ',      # Added
            'lon': 'lớn',    # Added
            'nho': 'nhỏ',    # Added
        }

        # Load data
        self._load_mappings()
        self._build_vocabulary()

    def _load_mappings(self):
        """Load word mappings from preprocessed data"""
        mapping_file = os.path.join(
            self.data_dir, "non_accented_to_words.json")

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.word_mappings = json.load(f)
            print(f"✅ Loaded {len(self.word_mappings):,} word mappings")
        except Exception as e:
            print(f"❌ Error loading mappings: {e}")
            self.word_mappings = {}

    def _build_vocabulary(self):
        """Build vocabulary and find max word length"""
        # Build clean vocabulary without spaces
        clean_vocab = set()

        # Add basic words first
        for word in self.basic_words.keys():
            clean_vocab.add(word.lower())
            self.max_word_length = max(self.max_word_length, len(word))

        for non_accented_word in self.word_mappings.keys():
            # Remove spaces to get actual input format
            clean_word = non_accented_word.replace(' ', '').lower()
            if clean_word and len(clean_word) >= 1:  # Allow single chars too
                clean_vocab.add(clean_word)
                self.max_word_length = max(
                    self.max_word_length, len(clean_word))

        # Also add spaced versions for better mapping
        for non_accented_word in self.word_mappings.keys():
            if ' ' not in non_accented_word and len(non_accented_word) >= 1:
                clean_vocab.add(non_accented_word.lower())

        self.vocabulary = clean_vocab

        print(
            f"📊 Vocabulary: {len(self.vocabulary):,} words, max length: {self.max_word_length}")

    def _get_word_score(self, word: str) -> float:
        """Calculate score for a word based on length and validity"""
        base_score = len(word) * 2  # Reduced base multiplier

        # VERY HIGH bonus for basic words (IME-style)
        if word in self.basic_words:
            base_score += 100  # Massive bonus for basic words

        # Very strong bonus for complete words in priority list
        priority_words = {
            'toi', 'la', 'chao', 'hoc', 'sinh', 'ban', 'di', 'cam', 'on', 'yeu',
            'mang', 'den', 'cho', 'cac', 'bo', 'go', 'moi', 'nhieu', 'ten',
            'co', 'hanh', 'phuc', 've', 'muon', 'mon', 'to', 'anh', 'em', 'chi',
            'xinchao', 'camon', 'dihoc', 'vemuon', 'hocsinh', 'toiyeu'
        }

        if word in priority_words:
            base_score += 50  # Very strong bonus for priority words

        # Check if word exists in mappings (both spaced and non-spaced)
        if word in self.word_mappings:
            base_score += 30  # Strong bonus for direct mapping

        # Check if there's a spaced version
        for key in self.word_mappings.keys():
            if key.replace(' ', '') == word:
                base_score += 25  # Good bonus for compound words
                break

        # MUCH HIGHER penalty for very short words (avoid over-segmentation)
        if len(word) == 1:
            base_score -= 50  # VERY strong penalty for single characters
        elif len(word) == 2:
            base_score += 10  # Actually reward 2-char words since many Vietnamese words are 2 chars

        # Bonus for common word lengths in Vietnamese
        if 3 <= len(word) <= 6:
            base_score += 15  # Sweet spot for Vietnamese words
        elif 7 <= len(word) <= 10:
            base_score += 8  # Good for compound words

        return base_score

    def segment_dynamic(self, text: str) -> List[Tuple[str, str]]:
        """
        Improved Dynamic Programming segmentation with IME-style logic
        """
        text = text.lower().strip()
        if not text:
            return []

        n = len(text)

        # dp[i] = (best_score, best_segmentation_ending_at_i)
        dp = [(-float('inf'), [])] * (n + 1)
        dp[0] = (0, [])

        for i in range(1, n + 1):
            # Try all possible words ending at position i
            max_len = min(self.max_word_length, i, 10)  # Reasonable max length

            for j in range(max(0, i - max_len), i):
                word = text[j:i]

                # Check if this word exists and get its accented version
                is_valid_word = False
                accented_version = word

                # FIRST: Check basic words (highest priority)
                if word in self.basic_words:
                    is_valid_word = True
                    accented_version = self.basic_words[word]

                # SECOND: Direct check in mappings
                elif word in self.word_mappings:
                    is_valid_word = True
                    accented_version = self._get_best_accented(word)

                # THIRD: Check spaced version in mappings
                elif not is_valid_word:
                    for key in self.word_mappings.keys():
                        if key.replace(' ', '') == word and len(word) > 1:
                            is_valid_word = True
                            accented_version = self.word_mappings[key][0]
                            break

                # FOURTH: Allow single characters only as fallback
                elif len(word) == 1:
                    is_valid_word = True
                    accented_version = word

                if is_valid_word:
                    # Calculate score
                    score = self._get_word_score(word)

                    total_score = dp[j][0] + score

                    if total_score > dp[i][0]:
                        new_segmentation = dp[j][1] + \
                            [(word, accented_version)]
                        dp[i] = (total_score, new_segmentation)

            # Ensure we always have a valid segmentation
            if dp[i][0] == -float('inf'):
                char = text[i-1]
                # High penalty for fallback
                dp[i] = (dp[i-1][0] - 100, dp[i-1][1] + [(char, char)])

        return dp[n][1]

    def _get_best_accented(self, word: str) -> str:
        """Get the best accented version of a word"""
        # Try direct mapping first
        if word in self.word_mappings:
            options = self.word_mappings[word]
            # Choose the most common/appropriate word based on context
            return self._choose_best_option(word, options)

        # Try to find spaced version
        for key, values in self.word_mappings.items():
            if key.replace(' ', '') == word:
                return self._choose_best_option(word, values)

        # No mapping found, return as is
        return word

    def _choose_best_option(self, word: str, options: List[str]) -> str:
        """Choose the best option from multiple accented versions"""
        if not options:
            return word

        # Priority mapping for common words
        priority_map = {
            'toi': 'tôi',    # I/me
            'chao': 'chào',  # hello
            'hoc': 'học',    # study/learn
            'sinh': 'sinh',  # birth/student
            'ban': 'bạn',    # friend
            'di': 'đi',      # go
            'cam': 'cảm',    # feel
            'on': 'ơn',      # grace/favor
            'yeu': 'yêu',    # love
            'mang': 'mang',  # bring/carry
            'den': 'đến',    # to/arrive
            'cho': 'cho',    # give/for
            'cac': 'các',    # the/plural
            'bo': 'bộ',      # set/ministry
            'go': 'gõ',      # type/knock
            'moi': 'mới',    # new
            'nhieu': 'nhiều',  # many
            'ten': 'tên',    # name
            'la': 'là',      # is/be
            'gi': 'gì',      # what
            'dem': 'đêm',    # night
            'nay': 'này',    # this
            'co': 'có',      # have
            'hanh': 'hạnh',  # happiness
            'phuc': 'phúc',  # fortune
            've': 'về',      # return/about
            'muon': 'muộn',  # late
            'mon': 'món',    # dish
        }

        # Use priority if available
        if word in priority_map and priority_map[word] in options:
            return priority_map[word]

        # Otherwise, prefer options that appear more "natural"
        # Common endings and patterns
        preferred_endings = ['ết', 'ại', 'ồn', 'ận', 'ình', 'ươi', 'ậu']

        for option in options:
            for ending in preferred_endings:
                if option.endswith(ending):
                    return option

        # Default to first option
        return options[0]

    def segment_greedy(self, text: str) -> List[Tuple[str, str]]:
        """
        Greedy segmentation: Always pick the longest valid word first
        Returns: List of (non_accented, accented) tuples
        """
        text = text.lower().strip()
        if not text:
            return []

        result = []
        i = 0

        while i < len(text):
            # Try to find the longest valid word starting at position i
            best_word = None
            best_length = 0

            # Check all possible word lengths (longest first)
            for length in range(min(self.max_word_length, len(text) - i), 0, -1):
                candidate = text[i:i + length]

                if candidate in self.vocabulary:
                    best_word = candidate
                    best_length = length
                    break

            if best_word:
                # Get the accented version
                accented_options = self.word_mappings.get(
                    best_word, [best_word])
                best_accented = accented_options[0] if accented_options else best_word

                result.append((best_word, best_accented))
                i += best_length
            else:
                # No valid word found, take single character
                char = text[i]
                result.append((char, char))
                i += 1

        return result

    def segment_text(self, text: str, method: str = "dynamic") -> str:
        """
        Segment text and return accented version

        Args:
            text: Non-accented Vietnamese text (e.g., "toimangdenchocacban")
            method: "greedy" or "dynamic"

        Returns:
            Accented Vietnamese text (e.g., "tôi mang đến cho các bạn")
        """
        if method == "greedy":
            segments = self.segment_greedy(text)
        else:
            segments = self.segment_dynamic(text)

        # Join accented words with spaces
        accented_words = [accented for _, accented in segments]
        return ' '.join(accented_words)

    def segment_with_details(self, text: str, method: str = "dynamic") -> Dict:
        """
        Segment text and return detailed information

        Returns:
            {
                'original': original text,
                'segments': list of (non_accented, accented) tuples,
                'result': final accented text,
                'confidence': confidence score,
                'method': segmentation method used
            }
        """
        if method == "greedy":
            segments = self.segment_greedy(text)
        else:
            segments = self.segment_dynamic(text)

        # Calculate confidence based on known words ratio
        total_chars = len(text)
        known_chars = sum(len(non_acc) for non_acc, acc in segments
                          if non_acc in self.vocabulary)
        confidence = known_chars / total_chars if total_chars > 0 else 0

        accented_words = [accented for _, accented in segments]
        result_text = ' '.join(accented_words)

        return {
            'original': text,
            'segments': segments,
            'result': result_text,
            'confidence': confidence,
            'method': method,
            'word_count': len(segments)
        }

    def suggest_alternatives(self, text: str, max_alternatives: int = 3) -> List[Dict]:
        """
        Generate multiple segmentation alternatives
        """
        alternatives = []

        # Try both methods
        for method in ["greedy", "dynamic"]:
            result = self.segment_with_details(text, method)

            # Avoid duplicates
            if not any(alt['result'] == result['result'] for alt in alternatives):
                alternatives.append(result)

        # Sort by confidence
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)

        return alternatives[:max_alternatives]


def main():
    """Test the word segmenter"""
    print("🚀 Vietnamese Word Segmentation Test")
    print("=" * 60)

    # Initialize segmenter
    segmenter = VietnameseWordSegmenter()

    # Test cases
    test_cases = [
        "toimangdenchocacbanbogoamoi",
        "chaoban",
        "xinchao",
        "hocsinh",
        "dihoc",
        "vemuon",
        "camon",
        "camonnhieu",
        "bantenlagi",
        "hemnaydepanh",
        "demnaycohanhhpuc",
        "toiyeuban"
    ]

    print("\n🧪 Testing Word Segmentation:")

    for text in test_cases:
        print(f"\n📝 Input: '{text}'")

        # Get segmentation details
        result = segmenter.segment_with_details(text)

        print(f"✅ Output: '{result['result']}'")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"🔧 Method: {result['method']}")
        print(f"📋 Segments: {result['segments'][:5]}")  # Show first 5

        # Get alternatives
        alternatives = segmenter.suggest_alternatives(text, max_alternatives=2)
        if len(alternatives) > 1:
            print(f"🔄 Alternative: '{alternatives[1]['result']}'")


if __name__ == "__main__":
    main()
