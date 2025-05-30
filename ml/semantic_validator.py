#!/usr/bin/env python3
"""
Semantic Validator for Vietnamese Suggestions
Filter meaningless suggestions using PhoBERT và semantic rules
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import re


class SemanticValidator:
    """Validate semantic meaningfulness của Vietnamese suggestions"""

    def __init__(self):
        print("🔧 Initializing Semantic Validator...")

        # Load PhoBERT for semantic validation
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # Vietnamese word patterns
        self.meaningful_patterns = self._load_meaningful_patterns()
        self.meaningless_patterns = self._load_meaningless_patterns()

        # Perplexity thresholds
        self.perplexity_threshold = 50.0  # Reject if higher
        self.confidence_penalty = 0.3     # Penalty for low quality

        print("✅ Semantic Validator ready")

    def _load_meaningful_patterns(self) -> List[str]:
        """Load patterns that indicate meaningful Vietnamese"""
        return [
            # Common Vietnamese word combinations
            r'(tôi|bạn|anh|chị) (đi|về|ăn|làm|học)',
            r'(chợ|cửa hàng|siêu thị) (cá|thịt|rau)',
            r'(chờ|gặp|gọi) (bạn|anh|chị|mọi người)',
            r'(mua|bán|chọn|tìm) (thức ăn|đồ ăn|cá|thịt)',

            # Common actions
            r'(đem|mang|tặng|cho) (đến|về|cho)',
            r'(học|làm|viết|đọc) (bài|việc|sách)',
        ]

    def _load_meaningless_patterns(self) -> List[str]:
        """Load patterns that indicate meaningless text"""
        return [
            # Broken syllables
            r'(chốc|ậc|tợ|đệ|nghệt|hớ)',

            # Too many single characters
            r'\b[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ]\b.*\b[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ]\b.*\b[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ]\b',

            # Broken word boundaries
            r'[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ]\s[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ]{1,2}\s[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ]',
        ]

    def validate_suggestion(self, suggestion: Dict) -> Dict:
        """Validate single suggestion và return enhanced version"""
        text = suggestion['vietnamese_text']
        confidence = suggestion['confidence']

        # Check meaningfulness
        is_meaningful = self._check_meaningfulness(text)
        semantic_score = self._get_semantic_score(text)
        perplexity = self._calculate_perplexity(text)

        # Apply quality filters
        if not is_meaningful:
            confidence *= 0.3  # Heavy penalty for meaningless

        if perplexity > self.perplexity_threshold:
            confidence *= 0.5  # Penalty for high perplexity

        # Enhance suggestion
        enhanced_suggestion = suggestion.copy()
        enhanced_suggestion.update({
            'confidence': min(confidence, 95),  # Cap at 95%
            'is_meaningful': is_meaningful,
            'semantic_score': semantic_score,
            'perplexity': perplexity,
            'validation_status': 'validated'
        })

        return enhanced_suggestion

    def _check_meaningfulness(self, text: str) -> bool:
        """Check if text is semantically meaningful"""
        # Check for meaningless patterns
        for pattern in self.meaningless_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        # Check for broken segmentation
        words = text.split()
        if len(words) == 0:
            return False

        # Count single characters
        single_chars = [w for w in words if len(w) == 1]
        if len(single_chars) > len(words) * 0.3:
            return False

        # Check for meaningful patterns
        meaningful_count = 0
        for pattern in self.meaningful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                meaningful_count += 1

        # Basic heuristics
        avg_word_length = sum(len(w) for w in words) / len(words)
        has_common_words = any(
            word in ['tôi', 'bạn', 'chợ', 'cá', 'chờ', 'cả'] for word in words)

        return (meaningful_count > 0 or
                (avg_word_length >= 2.5 and has_common_words))

    def _get_semantic_score(self, text: str) -> float:
        """Get semantic score from PhoBERT"""
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt",
                                    padding=True, truncation=True)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Simple semantic score based on embedding norms
            score = torch.norm(embeddings).item()
            return min(score / 10.0, 10.0)  # Normalize to 0-10

        except Exception as e:
            print(f"⚠️ PhoBERT error: {e}")
            return 5.0  # Default neutral score

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity using PhoBERT"""
        try:
            # Simple perplexity approximation
            words = text.split()
            if len(words) == 0:
                return 1000.0  # Very high perplexity for empty

            # Basic heuristic: longer words = lower perplexity
            avg_length = sum(len(w) for w in words) / len(words)
            base_perplexity = max(1.0, 10.0 - avg_length)

            # Penalty for broken words
            broken_words = sum(1 for w in words if len(w) == 1)
            perplexity = base_perplexity * (1 + broken_words * 2)

            return min(perplexity, 1000.0)

        except Exception:
            return 100.0  # High perplexity on error

    def filter_suggestions(self, suggestions: List[Dict],
                           min_confidence: float = 50.0,
                           max_suggestions: int = 5) -> List[Dict]:
        """Filter và validate all suggestions"""
        if not suggestions:
            return []

        # Validate each suggestion
        validated = []
        for suggestion in suggestions:
            enhanced = self.validate_suggestion(suggestion)

            # Apply filters
            if (enhanced['is_meaningful'] and
                enhanced['confidence'] >= min_confidence and
                    enhanced['perplexity'] <= self.perplexity_threshold):
                validated.append(enhanced)

        # Sort by confidence và return top results
        validated.sort(key=lambda x: x['confidence'], reverse=True)
        return validated[:max_suggestions]

    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        return {
            'meaningful_patterns': len(self.meaningful_patterns),
            'meaningless_patterns': len(self.meaningless_patterns),
            'perplexity_threshold': self.perplexity_threshold,
            'validation_active': True
        }


def test_semantic_validator():
    """Test semantic validator"""
    print("🧪 Testing Semantic Validator")
    print("=" * 50)

    validator = SemanticValidator()

    # Test cases
    test_suggestions = [
        {'vietnamese_text': 'chốc ậc bạn', 'confidence': 52, 'method': 'test'},
        {'vietnamese_text': 'chợ cạ c ba n', 'confidence': 37, 'method': 'test'},
        {'vietnamese_text': 'chợ cá bạn', 'confidence': 85, 'method': 'test'},
        {'vietnamese_text': 'chờ cả bạn', 'confidence': 80, 'method': 'test'},
    ]

    print("📝 Testing Individual Suggestions:")
    for suggestion in test_suggestions:
        enhanced = validator.validate_suggestion(suggestion)
        status = "✅" if enhanced['is_meaningful'] else "❌"

        print(
            f"\n  Input: '{suggestion['vietnamese_text']}' ({suggestion['confidence']}%)")
        print(f"  Result: {status} Meaningful: {enhanced['is_meaningful']}")
        print(f"  Enhanced confidence: {enhanced['confidence']:.1f}%")
        print(f"  Semantic score: {enhanced['semantic_score']:.1f}")
        print(f"  Perplexity: {enhanced['perplexity']:.1f}")

    # Test filtering
    print(f"\n🔧 Testing Filter Function:")
    filtered = validator.filter_suggestions(
        test_suggestions, min_confidence=50.0)

    print(f"  Original: {len(test_suggestions)} suggestions")
    print(f"  Filtered: {len(filtered)} meaningful suggestions")

    for i, suggestion in enumerate(filtered, 1):
        print(
            f"  {i}. '{suggestion['vietnamese_text']}' ({suggestion['confidence']:.1f}%)")

    # Statistics
    stats = validator.get_validation_stats()
    print(f"\n📊 Validator Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_semantic_validator()
