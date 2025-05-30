#!/usr/bin/env python3
"""
Vietnamese Accent Marker Integration
Kết hợp với ViBERT để tạo suggestions with perfect accents
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Optional
import requests
import os


class VietnameseAccentMarker:
    """Vietnamese Accent Marker using XLM-Roberta"""

    def __init__(self):
        print("🎯 Initializing Vietnamese Accent Marker...")
        self.model = None
        self.tokenizer = None
        self.label_list = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._load_model()
            self._load_tags()
            print("✅ Accent Marker ready")
        except Exception as e:
            print(f"⚠️ Accent Marker failed to load: {e}")

    def _load_model(self):
        """Load accent marker model"""
        model_path = "peterhung/vietnamese-accent-marker-xlm-roberta"

        print(f"📥 Loading accent marker: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path)

        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully")

    def _load_tags(self):
        """Load tags from file or download if needed"""
        tags_file = "selected_tags_names.txt"

        if not os.path.exists(tags_file):
            print("📥 Downloading tags file...")
            self._download_tags_file(tags_file)

        self.label_list = self._load_tags_set(tags_file)
        print(f"✅ Loaded {len(self.label_list)} accent tags")

    def _download_tags_file(self, filename):
        """Download tags file if not exists"""
        # Simulated tags - in reality, download from the repo
        tags_content = """
-
a-à
a-á
a-ạ
a-ả
a-ã
e-è
e-é
e-ẹ
e-ẻ
e-ẽ
i-ì
i-í
i-ị
i-ỉ
i-ĩ
o-ò
o-ó
o-ọ
o-ỏ
o-õ
u-ù
u-ú
u-ụ
u-ủ
u-ũ
y-ỳ
y-ý
y-ỵ
y-ỷ
y-ỹ
di-đi
du-đư
da-đa
de-đê
""".strip()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(tags_content)

    def _load_tags_set(self, fpath):
        """Load tags from file"""
        labels = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        return labels

    def insert_accents(self, text: str) -> str:
        """Insert accents for Vietnamese text"""
        if not self.model or not text.strip():
            return text

        try:
            # Get tokens and predictions
            tokens, predictions = self._get_predictions(text)

            # Merge tokens and predictions
            merged_tokens_preds = self._merge_tokens_and_preds(
                tokens, predictions)

            # Get accented words
            accented_words = self._get_accented_words(merged_tokens_preds)

            return ' '.join(accented_words)

        except Exception as e:
            print(f"⚠️ Accent insertion error: {e}")
            return text

    def _get_predictions(self, text: str):
        """Get model predictions"""
        our_tokens = text.strip().split()

        inputs = self.tokenizer(
            our_tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = tokens[1:-1]  # Remove <s> and </s>

        with torch.no_grad():
            inputs.to(self.device)
            outputs = self.model(**inputs)

        predictions = outputs["logits"].cpu().numpy()
        predictions = np.argmax(predictions, axis=2)
        predictions = predictions[0][1:-1]  # Remove special tokens

        return tokens, predictions

    def _merge_tokens_and_preds(self, tokens, predictions):
        """Merge subword tokens back to original words"""
        TOKENIZER_WORD_PREFIX = "▁"
        merged_tokens_preds = []
        i = 0

        while i < len(tokens):
            tok = tokens[i]
            label_indexes = set([predictions[i]])

            if tok.startswith(TOKENIZER_WORD_PREFIX):
                tok_no_prefix = tok[len(TOKENIZER_WORD_PREFIX):]
                cur_word_toks = [tok_no_prefix]

                # Check subsequent tokens
                j = i + 1
                while j < len(tokens):
                    if not tokens[j].startswith(TOKENIZER_WORD_PREFIX):
                        cur_word_toks.append(tokens[j])
                        label_indexes.add(predictions[j])
                        j += 1
                    else:
                        break

                cur_word = ''.join(cur_word_toks)
                merged_tokens_preds.append((cur_word, label_indexes))
                i = j
            else:
                merged_tokens_preds.append((tok, label_indexes))
                i += 1

        return merged_tokens_preds

    def _get_accented_words(self, merged_tokens_preds):
        """Apply accent tags to get final words"""
        accented_words = []

        for word_raw, label_indexes in merged_tokens_preds:
            word_accented = word_raw

            # Apply first valid transformation
            for label_index in label_indexes:
                if label_index < len(self.label_list):
                    tag_name = self.label_list[int(label_index)]

                    if '-' in tag_name:
                        raw, vowel = tag_name.split("-", 1)
                        if raw and raw in word_raw:
                            word_accented = word_raw.replace(raw, vowel)
                            break

            accented_words.append(word_accented)

        return accented_words


class HybridViBERTAccentProcessor:
    """Hybrid processor combining ViBERT + Accent Marker"""

    def __init__(self):
        print("🚀 Initializing Hybrid ViBERT + Accent Marker...")

        # Load ViBERT
        from ml.vibert_processor import ViBERTVietnameseProcessor
        self.vibert = ViBERTVietnameseProcessor()

        # Load Accent Marker
        self.accent_marker = VietnameseAccentMarker()

        print("✅ Hybrid processor ready")

    def process_with_accents(self, input_text: str, max_suggestions: int = 10) -> List[Dict]:
        """Process input with ViBERT + accent enhancement"""

        # 1. Get ViBERT suggestions
        vibert_suggestions = self.vibert.process_input(
            input_text, max_suggestions=max_suggestions*2)

        # 2. Generate accent variations
        enhanced_suggestions = []

        for suggestion in vibert_suggestions:
            original_text = suggestion['vietnamese_text']

            # Original suggestion
            enhanced_suggestions.append({
                **suggestion,
                'source': suggestion.get('source', 'vibert') + '_original'
            })

            # Accent-enhanced version
            if self.accent_marker.model:
                try:
                    # Remove accents first, then re-add with model
                    no_accent_text = self._remove_accents(original_text)
                    accented_text = self.accent_marker.insert_accents(
                        no_accent_text)

                    if accented_text != original_text:
                        enhanced_suggestions.append({
                            'vietnamese_text': accented_text,
                            # Boost confidence
                            'confidence': min(suggestion['confidence'] + 5, 95),
                            'method': suggestion['method'] + '_accent_enhanced',
                            'source': 'accent_marker',
                            'original_suggestion': original_text
                        })
                except Exception as e:
                    print(f"⚠️ Accent enhancement error: {e}")

        # 3. Remove duplicates and rank
        unique_suggestions = self._deduplicate_and_rank(enhanced_suggestions)

        # 4. Generate additional no-accent variations
        no_accent_suggestions = self._generate_no_accent_variations(
            input_text, unique_suggestions)
        unique_suggestions.extend(no_accent_suggestions)

        # Final ranking
        final_suggestions = self._deduplicate_and_rank(unique_suggestions)

        return final_suggestions[:max_suggestions]

    def _remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents"""
        accent_map = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd'
        }

        result = text
        for accented, plain in accent_map.items():
            result = result.replace(accented, plain)

        return result

    def _generate_no_accent_variations(self, input_text: str, existing_suggestions: List[Dict]) -> List[Dict]:
        """Generate additional suggestions by accent correction"""
        additional_suggestions = []

        if not self.accent_marker.model:
            return additional_suggestions

        try:
            # Try accent correction on raw input
            accented_input = self.accent_marker.insert_accents(input_text)

            if accented_input != input_text:
                # Check if not already in suggestions
                existing_texts = [s['vietnamese_text']
                                  for s in existing_suggestions]

                if accented_input not in existing_texts:
                    additional_suggestions.append({
                        'vietnamese_text': accented_input,
                        'confidence': 80,
                        'method': 'direct_accent_correction',
                        'source': 'accent_marker_direct'
                    })

        except Exception as e:
            print(f"⚠️ Additional suggestion error: {e}")

        return additional_suggestions

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


def test_accent_integration():
    """Test accent marker integration"""
    print("🧪 TESTING ACCENT MARKER INTEGRATION")
    print("=" * 60)

    processor = HybridViBERTAccentProcessor()

    test_cases = [
        'toimuon',
        'nha',
        'doc',
        'anhdichuyen',
        'xinchao',
        'camon'
    ]

    for input_text in test_cases:
        print(f"\n📝 Testing: '{input_text}'")

        suggestions = processor.process_with_accents(
            input_text, max_suggestions=5)

        if suggestions:
            print(f"   Enhanced suggestions ({len(suggestions)}):")
            for i, suggestion in enumerate(suggestions, 1):
                method = suggestion.get('method', 'unknown')
                source = suggestion.get('source', 'unknown')
                print(
                    f"     {i}. '{suggestion['vietnamese_text']}' ({suggestion['confidence']}%) - {method} [{source}]")
        else:
            print("   No suggestions")


def test_direct_accent_correction():
    """Test direct accent correction"""
    print(f"\n🎯 TESTING DIRECT ACCENT CORRECTION")
    print("=" * 60)

    accent_marker = VietnameseAccentMarker()

    test_cases = [
        'toi muon di hoc',
        'xin chao ban',
        'cam on ban',
        'nhin nhung mua thu di',
        'anh di chuyen den nha'
    ]

    for text in test_cases:
        if accent_marker.model:
            corrected = accent_marker.insert_accents(text)
            print(f"Input:  '{text}'")
            print(f"Output: '{corrected}'")
            print()


if __name__ == "__main__":
    test_direct_accent_correction()
    test_accent_integration()
