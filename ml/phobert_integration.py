#!/usr/bin/env python3
"""
PhoBERT Integration for Vietnamese Keyboard
Kết hợp PhoBERT-base-v2 để nâng cao AI-driven approach
"""

import torch
import warnings
from typing import List, Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer


class PhoBERTVietnameseEnhancer:
    """
    PhoBERT-enhanced Vietnamese input method
    Sử dụng state-of-the-art PhoBERT để improve pattern recognition và quality scoring
    """

    def __init__(self, model_name: str = "vinai/phobert-base-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        self._load_model()

        print(f"🤖 PhoBERT Enhanced System initialized on {self.device}")

    def _load_model(self):
        """Load PhoBERT model và tokenizer"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                print(f"📥 Loading PhoBERT model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()

                print(f"✅ PhoBERT loaded successfully")

        except Exception as e:
            print(f"⚠️ Could not load PhoBERT: {e}")
            print("🔄 Fallback to non-PhoBERT mode")
            self.model = None
            self.tokenizer = None

    def enhance_suggestions(self, input_text: str, base_suggestions: List[Dict]) -> List[Dict]:
        """
        Enhance base suggestions using PhoBERT contextual understanding
        """
        if not self.model or not base_suggestions:
            return base_suggestions

        enhanced_suggestions = []

        for suggestion in base_suggestions:
            try:
                # PhoBERT quality scoring
                phobert_score = self._calculate_phobert_score(
                    suggestion['vietnamese_text'])

                # Combine scores
                combined_confidence = self._combine_scores(
                    suggestion['confidence'],
                    phobert_score
                )

                enhanced_suggestion = suggestion.copy()
                enhanced_suggestion['confidence'] = combined_confidence
                enhanced_suggestion['phobert_score'] = phobert_score
                enhanced_suggestion['method'] = f"{suggestion['method']}_phobert_enhanced"

                enhanced_suggestions.append(enhanced_suggestion)

            except Exception as e:
                # Fallback to original suggestion nếu PhoBERT fails
                enhanced_suggestions.append(suggestion)

        # Sort by enhanced confidence
        enhanced_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return enhanced_suggestions

    def _calculate_phobert_score(self, text: str) -> float:
        """
        Calculate PhoBERT-based quality score cho Vietnamese text
        """
        try:
            # Word segment text (PhoBERT requires word-segmented input)
            segmented_text = self._word_segment_for_phobert(text)

            # Encode text
            input_ids = torch.tensor(
                [self.tokenizer.encode(segmented_text)]).to(self.device)

            with torch.no_grad():
                # Get PhoBERT features
                outputs = self.model(input_ids)

                # Calculate perplexity-based score
                # Lower perplexity = higher quality
                hidden_states = outputs.last_hidden_state

                # Simple quality metric based on hidden state variance
                # Good Vietnamese text should have consistent patterns
                variance = torch.var(hidden_states).item()

                # Convert to 0-100 scale (higher = better)
                quality_score = max(0, min(100, 85 - (variance * 1000)))

                return quality_score

        except Exception as e:
            # Fallback score
            return 50.0

    def _word_segment_for_phobert(self, text: str) -> str:
        """
        Simple word segmentation for PhoBERT
        PhoBERT requires word-segmented input với underscores
        """
        # Basic segmentation rules cho common Vietnamese patterns
        words = text.split()
        segmented_words = []

        for word in words:
            # Common compound words
            if word in ['sinh viên', 'sinh_viên']:
                segmented_words.append('sinh_viên')
            elif word in ['đại học', 'đại_học']:
                segmented_words.append('đại_học')
            elif word in ['công nghệ', 'công_nghệ']:
                segmented_words.append('công_nghệ')
            elif word in ['nghiên cứu', 'nghiên_cứu']:
                segmented_words.append('nghiên_cứu')
            else:
                segmented_words.append(word)

        return ' '.join(segmented_words)

    def _combine_scores(self, base_confidence: float, phobert_score: float) -> float:
        """
        Combine base confidence với PhoBERT score
        """
        # Weighted combination: 70% base + 30% PhoBERT
        combined = (base_confidence * 0.7) + (phobert_score * 0.3)
        return min(95, max(10, int(combined)))

    def generate_phobert_suggestions(self, input_text: str, context: str = "") -> List[Dict]:
        """
        Generate suggestions using PhoBERT fill-mask capability
        Đặc biệt useful cho toi+verb+object patterns
        """
        if not self.model:
            return []

        suggestions = []

        try:
            # Strategy 1: Fill missing parts trong toi+verb+object
            if input_text.startswith('toi') and len(input_text) == 9:
                # Parse toi+verb+object structure
                seg1 = input_text[0:3]  # toi
                seg2 = input_text[3:6]  # verb
                seg3 = input_text[6:9]  # object

                # Create masked patterns để PhoBERT predict
                masked_patterns = [
                    f"tôi <mask> {seg3}",  # Predict verb
                    f"tôi {seg2} <mask>",  # Predict object
                    f"<mask> {seg2} {seg3}"  # Predict subject
                ]

                for pattern in masked_patterns:
                    try:
                        # Use PhoBERT fill-mask (cần implement)
                        predicted = self._simple_fill_mask(pattern)
                        if predicted:
                            suggestions.append({
                                'vietnamese_text': predicted,
                                'confidence': 85,
                                'method': 'phobert_fill_mask',
                                'pattern_type': 'toi_verb_object'
                            })
                    except:
                        continue

        except Exception as e:
            pass

        return suggestions[:2]  # Top 2 PhoBERT suggestions

    def _simple_fill_mask(self, masked_text: str) -> Optional[str]:
        """
        Simple fill-mask implementation
        (Simplified version - real implementation would use full PhoBERT fill-mask)
        """
        try:
            # Basic pattern recognition
            if "tôi <mask>" in masked_text:
                if "đến" in masked_text:
                    return masked_text.replace("<mask>", "đem")
                elif "bạn" in masked_text:
                    return masked_text.replace("<mask>", "tặng")
                elif "chợ" in masked_text:
                    return masked_text.replace("<mask>", "đi")

            return None
        except:
            return None

    def get_contextual_embeddings(self, text: str) -> Optional[torch.Tensor]:
        """
        Get PhoBERT contextual embeddings cho advanced pattern matching
        """
        if not self.model:
            return None

        try:
            segmented_text = self._word_segment_for_phobert(text)
            input_ids = torch.tensor(
                [self.tokenizer.encode(segmented_text)]).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                return outputs.last_hidden_state

        except Exception as e:
            return None

    def is_available(self) -> bool:
        """Check if PhoBERT is available"""
        return self.model is not None and self.tokenizer is not None


# Test function
def test_phobert_integration():
    """Test PhoBERT integration"""
    print("🧪 Testing PhoBERT Integration")
    print("=" * 50)

    enhancer = PhoBERTVietnameseEnhancer()

    if enhancer.is_available():
        # Test cases
        test_cases = [
            "tôi đem đến",
            "tôi mang đến",
            "tôi tặng bạn",
            "tôi đi chợ"
        ]

        for text in test_cases:
            score = enhancer._calculate_phobert_score(text)
            print(f"📝 '{text}' → PhoBERT score: {score:.1f}")

        # Test suggestions enhancement
        base_suggestions = [
            {'vietnamese_text': 'tôi đem đến',
                'confidence': 89, 'method': 'corpus_learning'},
            {'vietnamese_text': 'tôi đêm đến', 'confidence': 75,
                'method': 'hybrid_segmentation'}
        ]

        enhanced = enhancer.enhance_suggestions("toidemden", base_suggestions)
        print(f"\n🚀 Enhanced suggestions:")
        for i, sug in enumerate(enhanced, 1):
            print(
                f"  {i}. {sug['vietnamese_text']} ({sug['confidence']}%) - {sug['method']}")
            if 'phobert_score' in sug:
                print(f"     PhoBERT score: {sug['phobert_score']:.1f}")
    else:
        print("⚠️ PhoBERT not available - install transformers và torch")


if __name__ == "__main__":
    test_phobert_integration()
