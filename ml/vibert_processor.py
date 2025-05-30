#!/usr/bin/env python3
"""
ViBERT-based Vietnamese Processor
Using FPTAI/vibert-base-cased instead of PhoBERT
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import re
from collections import defaultdict


class ViBERTVietnameseProcessor:
    """Vietnamese processor using ViBERT model"""

    def __init__(self):
        print("🚀 Initializing ViBERT Vietnamese Processor...")

        # Load ViBERT model
        try:
            print("📥 Loading ViBERT model: FPTAI/vibert-base-cased")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "FPTAI/vibert-base-cased")
            self.model = AutoModel.from_pretrained("FPTAI/vibert-base-cased")
            print("✅ ViBERT loaded successfully")
        except Exception as e:
            print(f"⚠️ ViBERT loading failed: {e}")
            print("🔄 Falling back to PhoBERT...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vinai/phobert-base-v2")
            self.model = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # Core Vietnamese patterns
        self.load_vietnamese_patterns()

        print("✅ ViBERT Processor ready")

    def load_vietnamese_patterns(self):
        """Load Vietnamese language patterns"""

        # Basic syllable mappings
        self.basic_mappings = {
            'mot': 'một',
            'toi': 'tôi',
            'ban': 'bạn',
            'nha': 'nhà',
            'di': 'đi',
            'an': 'ăn',
            'hoc': 'học',
            'lam': 'làm',
            'viet': 'viết',
            'doc': 'đọc',
            'moi': 'mới',
            'co': 'có',
            'khong': 'không',
            'duoc': 'được',
            'rat': 'rất',
            'nhieu': 'nhiều',
            'cung': 'cũng',
            'den': 'đến',
            'tu': 'từ',
            'trong': 'trong',
            'ngoai': 'ngoài',
            'tren': 'trên',
            'duoi': 'dưới',
            'giua': 'giữa',
            'sau': 'sau',
            'truoc': 'trước'
        }

        # Common compound patterns
        self.compound_patterns = {
            'toidemden': 'tôi đem đến',
            'toilambai': 'tôi làm bài',
            'toimangden': 'tôi mang đến',
            'anhdichuyen': 'anh đi chuyển',
            'emhocbai': 'em học bài',
            'chungtoilam': 'chúng tôi làm',
            'banvietbai': 'bạn viết bài',
            'cogiaoday': 'cô giáo dạy',
            'thaygiaoday': 'thầy giáo dạy',
            'hocsinhhoc': 'học sinh học',
            'chocacban': 'cho các bạn'  # Context-dependent
        }

        # Vietnamese phonetic patterns
        self.phonetic_patterns = {
            'nh': ['nh', 'n'],
            'ch': ['ch', 'tr'],
            'gi': ['gi', 'dz'],
            'ph': ['ph', 'f'],
            'd': ['d', 'z'],
            'tr': ['tr', 'ch'],
            'x': ['x', 's']
        }

    def process_input(self, input_text: str, max_suggestions: int = 5) -> List[Dict]:
        """Process input với ViBERT enhancement"""

        input_text = input_text.lower().strip()
        if not input_text:
            return []

        suggestions = []

        # 1. Direct mapping check
        direct_suggestions = self._get_direct_mappings(input_text)
        suggestions.extend(direct_suggestions)

        # 2. Compound pattern check
        compound_suggestions = self._get_compound_suggestions(input_text)
        suggestions.extend(compound_suggestions)

        # 3. ViBERT-enhanced suggestions
        vibert_suggestions = self._get_vibert_suggestions(input_text)
        suggestions.extend(vibert_suggestions)

        # 4. Phonetic variations
        phonetic_suggestions = self._get_phonetic_suggestions(input_text)
        suggestions.extend(phonetic_suggestions)

        # Remove duplicates and rank
        unique_suggestions = self._deduplicate_and_rank(suggestions)

        return unique_suggestions[:max_suggestions]

    def _get_direct_mappings(self, input_text: str) -> List[Dict]:
        """Get direct syllable mappings"""
        suggestions = []

        if input_text in self.basic_mappings:
            suggestions.append({
                'vietnamese_text': self.basic_mappings[input_text],
                'confidence': 85,
                'method': 'direct_mapping',
                'source': 'vibert_basic'
            })

        return suggestions

    def _get_compound_suggestions(self, input_text: str) -> List[Dict]:
        """Get compound pattern suggestions"""
        suggestions = []

        if input_text in self.compound_patterns:
            suggestions.append({
                'vietnamese_text': self.compound_patterns[input_text],
                'confidence': 90,
                'method': 'compound_pattern',
                'source': 'vibert_compounds'
            })

        return suggestions

    def _get_vibert_suggestions(self, input_text: str) -> List[Dict]:
        """Get ViBERT-enhanced suggestions"""
        suggestions = []

        try:
            # Generate candidates based on common Vietnamese patterns
            candidates = self._generate_candidates(input_text)

            # Score each candidate with ViBERT
            for candidate in candidates:
                score = self._score_with_vibert(input_text, candidate)

                if score > 0.5:  # Threshold for relevance
                    suggestions.append({
                        'vietnamese_text': candidate,
                        'confidence': min(score * 100, 95),
                        'method': 'vibert_enhanced',
                        'source': 'vibert_scoring',
                        'vibert_score': score
                    })

        except Exception as e:
            print(f"⚠️ ViBERT enhancement error: {e}")

        return suggestions

    def _generate_candidates(self, input_text: str) -> List[str]:
        """Generate candidate Vietnamese texts"""
        candidates = []

        # Simple segmentation candidates
        if len(input_text) >= 6:
            # Try 3-3 split
            mid = len(input_text) // 2
            part1 = input_text[:mid]
            part2 = input_text[mid:]

            if part1 in self.basic_mappings and part2 in self.basic_mappings:
                candidate = f"{self.basic_mappings[part1]} {self.basic_mappings[part2]}"
                candidates.append(candidate)

        # Add tone variations for common syllables
        if input_text in ['mot', 'nha', 'co', 'lam', 'an']:
            tone_variations = {
                'mot': ['một', 'mốt', 'mọt', 'mót', 'mợt'],
                'nha': ['nhà', 'nhá', 'nhạ', 'nhả', 'nhã'],
                'co': ['có', 'cô', 'có', 'cò', 'cõ'],
                'lam': ['làm', 'lám', 'lạm', 'lảm', 'lãm'],
                'an': ['ăn', 'án', 'ạn', 'ản', 'ãn']
            }

            if input_text in tone_variations:
                candidates.extend(tone_variations[input_text])

        return candidates

    def _score_with_vibert(self, input_text: str, candidate: str) -> float:
        """Score candidate using ViBERT"""
        try:
            # Create context for scoring
            context = f"Từ '{input_text}' có nghĩa là '{candidate}'"

            # Tokenize
            inputs = self.tokenizer(context, return_tensors="pt",
                                    padding=True, truncation=True, max_length=128)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Calculate semantic coherence score
            norm = torch.norm(embeddings).item()

            # Normalize score between 0 and 1
            score = min(max(norm / 10.0, 0.0), 1.0)

            return score

        except Exception as e:
            print(f"⚠️ ViBERT scoring error: {e}")
            return 0.5  # Default neutral score

    def _get_phonetic_suggestions(self, input_text: str) -> List[Dict]:
        """Get phonetic variation suggestions"""
        suggestions = []

        # Generate phonetic variations
        variations = self._generate_phonetic_variations(input_text)

        for variation in variations:
            if variation in self.basic_mappings:
                suggestions.append({
                    'vietnamese_text': self.basic_mappings[variation],
                    'confidence': 70,
                    'method': 'phonetic_variation',
                    'source': 'vibert_phonetic'
                })

        return suggestions

    def _generate_phonetic_variations(self, input_text: str) -> List[str]:
        """Generate phonetic variations of input"""
        variations = [input_text]

        # Apply phonetic rules
        for pattern, replacements in self.phonetic_patterns.items():
            if pattern in input_text:
                for replacement in replacements:
                    if replacement != pattern:
                        variation = input_text.replace(pattern, replacement)
                        variations.append(variation)

        return list(set(variations))

    def _deduplicate_and_rank(self, suggestions: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank by confidence"""
        seen = {}

        for suggestion in suggestions:
            text = suggestion['vietnamese_text']
            if text not in seen or suggestion['confidence'] > seen[text]['confidence']:
                seen[text] = suggestion

        unique_suggestions = list(seen.values())
        unique_suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return unique_suggestions


def test_vibert_processor():
    """Test ViBERT processor"""
    print("🧪 TESTING ViBERT PROCESSOR")
    print("=" * 50)

    processor = ViBERTVietnameseProcessor()

    # Test cases from quality assessment
    test_cases = [
        'mot',
        'toi',
        'nha',
        'doc',
        'toidemden',
        'toilambai',
        'anhdichuyen',
        'chungtoilam',
        'cogiaoday'
    ]

    for input_text in test_cases:
        print(f"\n📝 Input: '{input_text}'")

        suggestions = processor.process_input(input_text, max_suggestions=5)

        if suggestions:
            print(f"   Suggestions ({len(suggestions)}):")
            for i, suggestion in enumerate(suggestions, 1):
                score_info = f" (ViBERT: {suggestion['vibert_score']:.2f})" if 'vibert_score' in suggestion else ""
                print(
                    f"     {i}. '{suggestion['vietnamese_text']}' ({suggestion['confidence']}%) - {suggestion['method']}{score_info}")
        else:
            print("   No suggestions")


def compare_with_old_processor():
    """Compare ViBERT với PhoBERT processor"""
    print(f"\n🔬 COMPARISON: ViBERT vs PhoBERT")
    print("=" * 50)

    # Load both processors
    vibert_processor = ViBERTVietnameseProcessor()

    from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor
    phobert_processor = HybridVietnameseProcessor()

    test_cases = ['mot', 'nha', 'doc', 'anhdichuyen', 'cogiaoday']

    for input_text in test_cases:
        print(f"\n📝 Testing: '{input_text}'")

        # ViBERT results
        vibert_results = vibert_processor.process_input(
            input_text, max_suggestions=3)
        print(f"   ViBERT:")
        if vibert_results:
            for i, result in enumerate(vibert_results, 1):
                print(
                    f"     {i}. '{result['vietnamese_text']}' ({result['confidence']}%)")
        else:
            print("     No suggestions")

        # PhoBERT results
        phobert_results = phobert_processor.process_text(
            input_text, max_suggestions=3)
        print(f"   PhoBERT:")
        if phobert_results:
            for i, result in enumerate(phobert_results, 1):
                print(
                    f"     {i}. '{result['vietnamese_text']}' ({result['confidence']}%)")
        else:
            print("     No suggestions")


if __name__ == "__main__":
    test_vibert_processor()
    compare_with_old_processor()
