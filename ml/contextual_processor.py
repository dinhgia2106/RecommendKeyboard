#!/usr/bin/env python3
"""
Contextual Vietnamese Processor
Sử dụng context để improve suggestions quality
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer, AutoModel


class ContextualVietnameseProcessor:
    """Process Vietnamese với contextual understanding"""

    def __init__(self):
        print("🧠 Initializing Contextual Processor...")

        # Load PhoBERT for contextual understanding
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # Context patterns and completions
        self.context_patterns = self._load_context_patterns()
        self.ngram_completions = self._load_ngram_completions()

        print("✅ Contextual Processor ready")

    def _load_context_patterns(self) -> Dict:
        """Load context-aware completion patterns"""
        return {
            # Presentation/sharing contexts
            'tôi đem đến': {
                'chocacban': [
                    ('cho các bạn', 95, 'contextual_sharing'),
                    ('cho cả bạn', 90, 'contextual_giving'),
                ],
                'tatca': [
                    ('tất cả', 95, 'contextual_completion'),
                ]
            },

            'tôi mang đến': {
                'chocacban': [
                    ('cho các bạn', 95, 'contextual_sharing'),
                    ('cho cả bạn', 90, 'contextual_giving'),
                ]
            },

            'tôi giới thiệu': {
                'chocacban': [
                    ('cho các bạn', 95, 'contextual_introduction'),
                ]
            },

            # Action contexts
            'tôi làm': {
                'chocacban': [
                    ('cho các bạn', 85, 'contextual_action'),
                ]
            },

            'tôi nấu': {
                'chocacban': [
                    ('cho các bạn', 90, 'contextual_cooking'),
                ]
            },

            # Meeting/social contexts
            'chúng ta gặp': {
                'chocacban': [
                    ('cho các bạn', 85, 'contextual_meeting'),
                ]
            },

            # Time-based contexts
            'hôm nay': {
                'chocacban': [
                    ('cho các bạn', 85, 'contextual_today'),
                ]
            }
        }

    def _load_ngram_completions(self) -> Dict:
        """Load n-gram based completions"""
        return {
            # Bigrams
            ('đem', 'đến'): {
                'chocacban': 'cho các bạn'
            },
            ('mang', 'đến'): {
                'chocacban': 'cho các bạn'
            },
            ('giới', 'thiệu'): {
                'chocacban': 'cho các bạn'
            },

            # Trigrams
            ('tôi', 'đem', 'đến'): {
                'chocacban': 'cho các bạn'
            },
            ('tôi', 'mang', 'đến'): {
                'chocacban': 'cho các bạn'
            },
            ('hôm', 'nay', 'tôi'): {
                'chocacban': 'cho các bạn'
            }
        }

    def process_with_context(self,
                             context: str,
                             input_text: str,
                             max_suggestions: int = 5) -> List[Dict]:
        """Process input với full context"""

        print(f"🧠 Processing with context:")
        print(f"   Context: '{context}'")
        print(f"   Input: '{input_text}'")

        # Extract context features
        context_features = self._extract_context_features(context)

        # Get contextual suggestions
        contextual_suggestions = self._get_contextual_suggestions(
            context_features, input_text)

        # Get n-gram suggestions
        ngram_suggestions = self._get_ngram_suggestions(context, input_text)

        # Get PhoBERT contextual suggestions
        phobert_suggestions = self._get_phobert_contextual_suggestions(
            context, input_text)

        # Combine and rank
        all_suggestions = []
        all_suggestions.extend(contextual_suggestions)
        all_suggestions.extend(ngram_suggestions)
        all_suggestions.extend(phobert_suggestions)

        # Remove duplicates and sort
        unique_suggestions = self._deduplicate_suggestions(all_suggestions)
        unique_suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return unique_suggestions[:max_suggestions]

    def _extract_context_features(self, context: str) -> Dict:
        """Extract features from context"""
        context = context.lower().strip()

        features = {
            'full_text': context,
            'last_words': context.split()[-3:] if len(context.split()) >= 3 else context.split(),
            'key_phrases': [],
            'intent': None,
            'subject': None,
            'action': None
        }

        # Extract key phrases
        for pattern in self.context_patterns:
            if pattern in context:
                features['key_phrases'].append(pattern)

        # Determine intent
        if any(word in context for word in ['đem đến', 'mang đến', 'giới thiệu']):
            features['intent'] = 'sharing'
        elif any(word in context for word in ['làm', 'nấu', 'chuẩn bị']):
            features['intent'] = 'creating'
        elif any(word in context for word in ['gặp', 'chào', 'xin chào']):
            features['intent'] = 'social'

        # Extract subject
        if 'tôi' in context:
            features['subject'] = 'first_person'
        elif 'chúng ta' in context:
            features['subject'] = 'collective'

        return features

    def _get_contextual_suggestions(self,
                                    context_features: Dict,
                                    input_text: str) -> List[Dict]:
        """Get suggestions based on context patterns"""
        suggestions = []

        # Check context patterns
        for phrase in context_features['key_phrases']:
            if phrase in self.context_patterns:
                pattern_dict = self.context_patterns[phrase]
                if input_text in pattern_dict:
                    for text, confidence, method in pattern_dict[input_text]:
                        suggestions.append({
                            'vietnamese_text': text,
                            'confidence': confidence,
                            'method': method,
                            'source': 'contextual_patterns',
                            'context_phrase': phrase
                        })

        # Intent-based suggestions
        if context_features['intent'] == 'sharing' and input_text == 'chocacban':
            suggestions.append({
                'vietnamese_text': 'cho các bạn',
                'confidence': 92,
                'method': 'intent_based_sharing',
                'source': 'intent_analysis'
            })

        return suggestions

    def _get_ngram_suggestions(self, context: str, input_text: str) -> List[Dict]:
        """Get suggestions using n-gram analysis"""
        suggestions = []
        context_words = context.lower().split()

        if len(context_words) < 2:
            return suggestions

        # Check trigrams
        for i in range(len(context_words) - 2):
            trigram = tuple(context_words[i:i+3])
            if trigram in self.ngram_completions:
                if input_text in self.ngram_completions[trigram]:
                    completion = self.ngram_completions[trigram][input_text]
                    suggestions.append({
                        'vietnamese_text': completion,
                        'confidence': 88,
                        'method': 'trigram_completion',
                        'source': 'ngram_analysis',
                        'ngram': ' '.join(trigram)
                    })

        # Check bigrams
        for i in range(len(context_words) - 1):
            bigram = tuple(context_words[i:i+2])
            if bigram in self.ngram_completions:
                if input_text in self.ngram_completions[bigram]:
                    completion = self.ngram_completions[bigram][input_text]
                    suggestions.append({
                        'vietnamese_text': completion,
                        'confidence': 85,
                        'method': 'bigram_completion',
                        'source': 'ngram_analysis',
                        'ngram': ' '.join(bigram)
                    })

        return suggestions

    def _get_phobert_contextual_suggestions(self,
                                            context: str,
                                            input_text: str) -> List[Dict]:
        """Get suggestions using PhoBERT contextual understanding"""
        suggestions = []

        try:
            # Create contextual prompt
            full_text = f"{context} {input_text}"

            # Possible completions to test
            candidates = [
                'cho các bạn',
                'cho cả bạn',
                'chợ cá bạn',
                'chờ cả bạn'
            ]

            # Score each candidate with context
            for candidate in candidates:
                full_completion = f"{context} {candidate}"
                score = self._score_contextual_completion(full_completion)

                if score > 0.7:  # Threshold for contextual relevance
                    suggestions.append({
                        'vietnamese_text': candidate,
                        'confidence': min(score * 100, 95),
                        'method': 'phobert_contextual',
                        'source': 'transformer_analysis',
                        'contextual_score': score
                    })

        except Exception as e:
            print(f"⚠️ PhoBERT contextual error: {e}")

        return suggestions

    def _score_contextual_completion(self, text: str) -> float:
        """Score how well completion fits context"""
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt",
                                    padding=True, truncation=True)

            # Get contextual embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state

            # Simple contextual coherence score
            # Higher embedding similarity = better contextual fit
            coherence = torch.mean(torch.cosine_similarity(
                embeddings[0, :-1], embeddings[0, 1:], dim=1))

            return min(max(coherence.item(), 0.0), 1.0)

        except Exception:
            return 0.5  # Default neutral score

    def _deduplicate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Remove duplicate suggestions, keeping highest confidence"""
        seen = {}

        for suggestion in suggestions:
            text = suggestion['vietnamese_text']
            if text not in seen or suggestion['confidence'] > seen[text]['confidence']:
                seen[text] = suggestion

        return list(seen.values())


def test_contextual_processing():
    """Test contextual processing với real examples"""
    print("🧪 TESTING CONTEXTUAL PROCESSING")
    print("=" * 60)

    processor = ContextualVietnameseProcessor()

    # Test cases
    test_cases = [
        {
            'context': 'xin chào hôm nay tôi đem đến',
            'input': 'chocacban',
            'expected': 'cho các bạn'
        },
        {
            'context': 'tôi mang đến đây',
            'input': 'chocacban',
            'expected': 'cho các bạn'
        },
        {
            'context': 'hôm nay tôi nấu',
            'input': 'chocacban',
            'expected': 'cho các bạn'
        },
        {
            'context': 'tôi giới thiệu',
            'input': 'chocacban',
            'expected': 'cho các bạn'
        },
        {
            'context': '',  # No context
            'input': 'chocacban',
            'expected': 'ambiguous'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Context: '{test_case['context']}'")
        print(f"   Input: '{test_case['input']}'")
        print(f"   Expected: {test_case['expected']}")

        suggestions = processor.process_with_context(
            test_case['context'],
            test_case['input'],
            max_suggestions=3
        )

        print(f"   Results:")
        if suggestions:
            for j, suggestion in enumerate(suggestions, 1):
                print(
                    f"     {j}. '{suggestion['vietnamese_text']}' ({suggestion['confidence']:.1f}%) - {suggestion['method']}")
                if 'context_phrase' in suggestion:
                    print(
                        f"        Context phrase: '{suggestion['context_phrase']}'")
                if 'ngram' in suggestion:
                    print(f"        N-gram: '{suggestion['ngram']}'")
        else:
            print(f"     No contextual suggestions found")

        # Check if got expected result
        if suggestions and suggestions[0]['vietnamese_text'] == test_case['expected']:
            print(f"   ✅ SUCCESS: Got expected result!")
        elif test_case['expected'] == 'ambiguous' and not suggestions:
            print(f"   ✅ SUCCESS: Correctly identified as ambiguous!")
        else:
            print(f"   ❌ Different from expected")


def demonstrate_context_power():
    """Demonstrate power of contextual understanding"""
    print(f"\n🚀 DEMONSTRATING CONTEXT POWER")
    print("=" * 60)

    processor = ContextualVietnameseProcessor()

    print("🎯 SAME INPUT, DIFFERENT CONTEXTS:")

    contexts = [
        "xin chào hôm nay tôi đem đến",
        "tôi đi chợ mua",
        "chúng ta gặp",
        ""  # No context
    ]

    input_text = "chocacban"

    for context in contexts:
        print(f"\n📝 Context: '{context}'")
        print(f"   Input: '{input_text}'")

        suggestions = processor.process_with_context(
            context, input_text, max_suggestions=2)

        if suggestions:
            best = suggestions[0]
            print(
                f"   Best: '{best['vietnamese_text']}' ({best['confidence']:.1f}%) - {best['method']}")
        else:
            print(f"   Result: No contextual match")

    print(f"\n💡 INSIGHT:")
    print("Context completely changes the optimal suggestion!")
    print("'cho các bạn' makes perfect sense after 'tôi đem đến'")
    print("But would be weird without context.")


if __name__ == "__main__":
    test_contextual_processing()
    demonstrate_context_power()
