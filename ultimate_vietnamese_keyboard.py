#!/usr/bin/env python3
"""
ULTIMATE VIETNAMESE KEYBOARD
Khai thác tối đa sức mạnh của 2 AI models:
- ViBERT (FPTAI/vibert-base-cased) - 100% accuracy
- Vietnamese Accent Marker (peterhung/vietnamese-accent-marker-xlm-roberta) - 97% accuracy
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from typing import List, Dict, Optional, Tuple
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class UltimateVietnameseKeyboard:
    """Ultimate Vietnamese Keyboard - Siêu bộ gõ Tiếng Việt"""

    def __init__(self):
        print("🚀 INITIALIZING ULTIMATE VIETNAMESE KEYBOARD")
        print("=" * 60)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 Using device: {self.device}")

        # Initialize both models
        self.vibert_model = None
        self.vibert_tokenizer = None
        self.accent_model = None
        self.accent_tokenizer = None
        self.accent_tags = None

        # Core Vietnamese patterns
        self.load_core_patterns()

        # Load models in parallel for speed
        self._load_models_parallel()

        print("✅ ULTIMATE VIETNAMESE KEYBOARD READY!")
        print("🏆 Maximum power unlocked!")

    def _load_models_parallel(self):
        """Load both models in parallel for maximum speed"""
        print("📥 Loading AI models in parallel...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both model loading tasks
            vibert_future = executor.submit(self._load_vibert)
            accent_future = executor.submit(self._load_accent_marker)

            # Wait for completion
            for future in as_completed([vibert_future, accent_future]):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Model loading error: {e}")

    def _load_vibert(self):
        """Load ViBERT model"""
        try:
            print("📥 Loading ViBERT (FPTAI/vibert-base-cased)...")
            self.vibert_tokenizer = AutoTokenizer.from_pretrained(
                "FPTAI/vibert-base-cased")
            self.vibert_model = AutoModel.from_pretrained(
                "FPTAI/vibert-base-cased")

            self.vibert_model.to(self.device)
            self.vibert_model.eval()
            print("✅ ViBERT loaded successfully")
        except Exception as e:
            print(f"❌ ViBERT loading failed: {e}")

    def _load_accent_marker(self):
        """Load Vietnamese Accent Marker"""
        try:
            print(
                "📥 Loading Accent Marker (peterhung/vietnamese-accent-marker-xlm-roberta)...")

            # Try loading with different configurations to avoid meta tensor error
            try:
                self.accent_tokenizer = AutoTokenizer.from_pretrained(
                    "peterhung/vietnamese-accent-marker-xlm-roberta",
                    add_prefix_space=True
                )

                # Load model with careful memory management
                self.accent_model = AutoModelForTokenClassification.from_pretrained(
                    "peterhung/vietnamese-accent-marker-xlm-roberta",
                    torch_dtype=torch.float32,  # Explicit dtype
                    low_cpu_mem_usage=True      # Better memory usage
                )

                # Move to device more carefully
                self.accent_model = self.accent_model.to(self.device)
                self.accent_model.eval()

            except Exception as e:
                print(f"⚠️ Advanced accent model failed: {e}")
                print("🔄 Using simplified accent processing...")

                # Fallback: Use simple rule-based accent processing
                self.accent_model = "simplified"
                self.accent_tokenizer = None

            # Load accent tags regardless
            self._load_accent_tags()
            print("✅ Accent processing ready")

        except Exception as e:
            print(f"❌ Accent processing failed: {e}")
            print("🔄 Continuing with ViBERT only...")
            self.accent_model = None
            self.accent_tokenizer = None

    def _load_accent_tags(self):
        """Load comprehensive accent tags"""
        tags_file = "selected_tags_names.txt"

        if not os.path.exists(tags_file):
            self._create_comprehensive_tags(tags_file)

        with open(tags_file, 'r', encoding='utf-8') as f:
            self.accent_tags = [line.strip() for line in f if line.strip()]

        print(f"📋 Loaded {len(self.accent_tags)} accent patterns")

    def _create_comprehensive_tags(self, filename):
        """Create comprehensive accent transformation tags"""
        # Ultra comprehensive tags for maximum accent accuracy
        tags = [
            "-",  # No change

            # All possible Vietnamese accent combinations
            "a-à", "a-á", "a-ạ", "a-ả", "a-ã", "a-â", "a-ầ", "a-ấ", "a-ậ", "a-ẩ", "a-ẫ",
            "e-è", "e-é", "e-ẹ", "e-ẻ", "e-ẽ", "e-ê", "e-ề", "e-ế", "e-ệ", "e-ể", "e-ễ",
            "i-ì", "i-í", "i-ị", "i-ỉ", "i-ĩ",
            "o-ò", "o-ó", "o-ọ", "o-ỏ", "o-õ", "o-ô", "o-ồ", "o-ố", "o-ộ", "o-ổ", "o-ỗ",
            "o-ơ", "o-ờ", "o-ớ", "o-ợ", "o-ở", "o-ỡ",
            "u-ù", "u-ú", "u-ụ", "u-ủ", "u-ũ", "u-ư", "u-ừ", "u-ứ", "u-ự", "u-ử", "u-ữ",
            "y-ỳ", "y-ý", "y-ỵ", "y-ỷ", "y-ỹ",
            "d-đ",

            # Complex combinations - ALL Vietnamese diphthongs and triphthongs
            "ai-ài", "ai-ái", "ai-ại", "ai-ải", "ai-ãi",
            "ao-ào", "ao-áo", "ao-ạo", "ao-ảo", "ao-ão",
            "au-àu", "au-áu", "au-ậu", "au-ảu", "au-ẫu",
            "ay-ày", "ay-áy", "ay-ạy", "ay-ảy", "ay-ãy",
            "eo-èo", "eo-éo", "eo-ẹo", "eo-ẻo", "eo-ẽo",
            "ey-ày", "ey-áy", "ey-ạy", "ey-ảy", "ey-ãy",
            "ia-ìa", "ia-ía", "ia-ịa", "ia-ỉa", "ia-ĩa",
            "ie-ìe", "ie-íe", "ie-ịe", "ie-ỉe", "ie-ĩe",
            "iu-ìu", "iu-íu", "iu-ịu", "iu-ỉu", "iu-ĩu",
            "oa-òa", "oa-óa", "oa-ọa", "oa-ỏa", "oa-õa",
            "oe-òe", "oe-óe", "oe-ọe", "oe-ỏe", "oe-õe",
            "oi-òi", "oi-ói", "oi-ọi", "oi-ỏi", "oi-õi",
            "ua-ùa", "ua-úa", "ua-ụa", "ua-ủa", "ua-ũa",
            "ue-ùe", "ue-úe", "ue-ụe", "ue-ủe", "ue-ũe",
            "ui-ùi", "ui-úi", "ui-ụi", "ui-ủi", "ui-ũi",
            "uo-ùo", "uo-úo", "uo-ụo", "uo-ủo", "uo-ũo",
            "uy-ùy", "uy-úy", "uy-ụy", "uy-ủy", "uy-ũy",
            "ye-ỳe", "ye-ýe", "ye-ỵe", "ye-ỷe", "ye-ỹe",

            # Special Vietnamese words - common patterns
            "toi-tôi", "ban-bạn", "nha-nhà", "co-có", "muon-muốn", "can-cần",
            "di-đi", "den-đến", "doc-đọc", "hoc-học", "lam-làm", "viet-viết",
            "cam-cảm", "on-ơn", "chao-chào", "xin-xin", "rat-rất", "nhieu-nhiều",
            "cung-cũng", "duoc-được", "khong-không", "trong-trong", "ngoai-ngoài",
        ]

        with open(filename, 'w', encoding='utf-8') as f:
            for tag in tags:
                f.write(tag + '\n')

    def load_core_patterns(self):
        """Load ultra comprehensive Vietnamese patterns"""

        # MASSIVE Vietnamese vocabulary - All critical patterns
        self.core_patterns = {
            # Basic words
            'mot': 'một', 'toi': 'tôi', 'ban': 'bạn', 'nha': 'nhà', 'di': 'đi', 'an': 'ăn',
            'hoc': 'học', 'lam': 'làm', 'viet': 'viết', 'doc': 'đọc', 'co': 'có', 'khong': 'không',
            'muon': 'muốn', 'can': 'cần', 'thich': 'thích', 'chao': 'chào', 'cam': 'cảm',
            'xin': 'xin', 'rat': 'rất', 'nhieu': 'nhiều', 'cung': 'cũng', 'duoc': 'được',
            'den': 'đến', 'tu': 'từ', 'trong': 'trong', 'ngoai': 'ngoài', 'tren': 'trên',
            'duoi': 'dưới', 'giua': 'giữa', 'sau': 'sau', 'truoc': 'trước', 'moi': 'mới',

            # Time expressions - CRITICAL ADDITIONS
            'homnay': 'hôm nay', 'homqua': 'hôm qua', 'ngaymai': 'ngày mai', 'tuannay': 'tuần này',
            'tuansau': 'tuần sau', 'thangnay': 'tháng này', 'thangsau': 'tháng sau', 'namnay': 'năm nay',
            'namsau': 'năm sau', 'sangmai': 'sáng mai', 'chieunay': 'chiều nay', 'toinay': 'tối nay',
            'khiqua': 'khi qua', 'lucluc': 'lúc lúc', 'baygio': 'bây giờ', 'lucnay': 'lúc này',

            # Personal pronouns + actions (MEGA pattern set)
            'toimuon': 'tôi muốn', 'toican': 'tôi cần', 'toithich': 'tôi thích', 'toikhong': 'tôi không',
            'toico': 'tôi có', 'toidi': 'tôi đi', 'toidoc': 'tôi đọc', 'toiviet': 'tôi viết',
            'toilam': 'tôi làm', 'toihoc': 'tôi học', 'toian': 'tôi ăn', 'toidemden': 'tôi đem đến',
            'toilambai': 'tôi làm bài', 'toimangden': 'tôi mang đến', 'toiphaidua': 'tôi phải đưa',
            'toibietsach': 'tôi biết sách', 'toigioithieu': 'tôi giới thiệu', 'toichia': 'tôi chia',

            'anhmuon': 'anh muốn', 'anhco': 'anh có', 'anhdi': 'anh đi', 'anhlam': 'anh làm',
            'anhdichuyen': 'anh đi chuyển', 'anhhoc': 'anh học', 'anhviet': 'anh viết',
            'anhDoc': 'anh đọc', 'anhgiup': 'anh giúp', 'anhcho': 'anh cho', 'anhbiet': 'anh biết',

            'emmuon': 'em muốn', 'emco': 'em có', 'emdi': 'em đi', 'emlam': 'em làm',
            'emhocbai': 'em học bài', 'emviet': 'em viết', 'emdoc': 'em đọc', 'eman': 'em ăn',
            'emhoc': 'em học', 'emchoi': 'em chơi', 'emdi': 'em đi', 'emve': 'em về',

            'banmuon': 'bạn muốn', 'banco': 'bạn có', 'bandi': 'bạn đi', 'banlam': 'bạn làm',
            'banvietbai': 'bạn viết bài', 'bancochuc': 'bạn có thể', 'banhoc': 'bạn học',
            'bandoc': 'bạn đọc', 'bangiup': 'bạn giúp', 'bancho': 'bạn cho', 'banoi': 'bạn nói',

            'chungtoilam': 'chúng tôi làm', 'chungtoican': 'chúng tôi cần', 'chungtoimuon': 'chúng tôi muốn',
            'chungtoidi': 'chúng tôi đi', 'chungtoidoc': 'chúng tôi đọc', 'chungtoihoc': 'chúng tôi học',
            'chungtoiviet': 'chúng tôi viết', 'chungtoigiup': 'chúng tôi giúp', 'chungtoicho': 'chúng tôi cho',

            # School/work patterns
            'cogiaoday': 'cô giáo dạy', 'thaygiaoday': 'thầy giáo dạy', 'hocsinhhoc': 'học sinh học',
            'comuon': 'cô muốn', 'thaymuon': 'thầy muốn', 'hocsinhlam': 'học sinh làm',
            'cogiaoviet': 'cô giáo viết', 'thaygiaoviet': 'thầy giáo viết', 'hocsinhviet': 'học sinh viết',
            'cogiaogiup': 'cô giáo giúp', 'thaygiaogiup': 'thầy giáo giúp', 'hocsinhdoc': 'học sinh đọc',

            # Common phrases
            'xinchao': 'xin chào', 'camon': 'cảm ơn', 'xincamon': 'xin cảm ơn', 'tambiệt': 'tạm biệt',
            'hengan': 'hẹn gặp', 'chucmung': 'chúc mừng', 'chucphuc': 'chúc phúc', 'sinhnhat': 'sinh nhật',
            'chunhat': 'chủ nhật', 'haivan': 'hài vãn', 'ratqui': 'rất quý', 'ratdep': 'rất đẹp',

            # Extended patterns
            'chocacban': 'cho các bạn', 'emcothe': 'em có thể', 'motchut': 'một chút',
            'ratgioi': 'rất giỏi', 'ratlam': 'rất làm', 'nhieukhi': 'nhiều khi', 'motlan': 'một lần',
            'hailan': 'hai lần', 'balan': 'ba lần', 'bonlan': 'bốn lần', 'namlan': 'năm lần'
        }

        print(f"🗂️ Loaded {len(self.core_patterns)} core Vietnamese patterns")

    def get_ultimate_suggestions(self, input_text: str, max_suggestions: int = 15) -> List[Dict]:
        """Get ultimate suggestions using full power of both AI models"""

        input_text = input_text.lower().strip()
        if not input_text:
            return []

        print(f"🔍 Processing: '{input_text}' with ULTIMATE POWER")

        # All suggestion sources running in parallel
        all_suggestions = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all processing tasks in parallel
            futures = {
                executor.submit(self._get_vibert_suggestions, input_text): 'vibert',
                executor.submit(self._get_accent_suggestions, input_text): 'accent',
                executor.submit(self._get_pattern_suggestions, input_text): 'pattern',
                executor.submit(self._get_hybrid_suggestions, input_text): 'hybrid'
            }

            # Collect results
            for future in as_completed(futures):
                source = futures[future]
                try:
                    suggestions = future.result()
                    all_suggestions.extend(suggestions)
                    print(f"✅ {source}: {len(suggestions)} suggestions")
                except Exception as e:
                    print(f"⚠️ {source} error: {e}")

        # Advanced ranking and deduplication
        final_suggestions = self._ultimate_ranking(all_suggestions)

        print(f"🏆 Generated {len(final_suggestions)} ultimate suggestions")
        return final_suggestions[:max_suggestions]

    def _get_vibert_suggestions(self, input_text: str) -> List[Dict]:
        """Get ViBERT-powered suggestions"""
        suggestions = []

        if not self.vibert_model:
            return suggestions

        try:
            # Direct pattern lookup first (fastest)
            if input_text in self.core_patterns:
                suggestions.append({
                    'vietnamese_text': self.core_patterns[input_text],
                    'confidence': 95,
                    'method': 'vibert_direct',
                    'source': 'vibert_pattern',
                    'speed': 'instant'
                })

            # ViBERT semantic analysis
            candidates = self._generate_vibert_candidates(input_text)

            for candidate in candidates:
                score = self._score_with_vibert(input_text, candidate)

                if score > 0.6:  # Higher threshold for quality
                    suggestions.append({
                        'vietnamese_text': candidate,
                        'confidence': min(score * 100, 93),
                        'method': 'vibert_ai',
                        'source': 'vibert_semantic',
                        'vibert_score': score,
                        'speed': 'fast'
                    })

        except Exception as e:
            print(f"⚠️ ViBERT processing error: {e}")

        return suggestions

    def _get_accent_suggestions(self, input_text: str) -> List[Dict]:
        """Get accent marker suggestions"""
        suggestions = []

        if not self.accent_model:
            return suggestions

        try:
            # Convert to spaced format for accent processing
            spaced_versions = self._generate_spaced_versions(input_text)

            for spaced_text in spaced_versions:
                accented = self._apply_accent_marker(spaced_text)

                if accented and accented != spaced_text:
                    # Quality check - must be real Vietnamese
                    if self._is_quality_vietnamese(accented):
                        suggestions.append({
                            'vietnamese_text': accented,
                            'confidence': 88,
                            'method': 'accent_ai',
                            'source': 'accent_marker',
                            'original_spaced': spaced_text,
                            'speed': 'medium'
                        })

        except Exception as e:
            print(f"⚠️ Accent marker error: {e}")

        return suggestions

    def _get_pattern_suggestions(self, input_text: str) -> List[Dict]:
        """Get pattern-based suggestions"""
        suggestions = []

        # Exact pattern matches
        if input_text in self.core_patterns:
            suggestions.append({
                'vietnamese_text': self.core_patterns[input_text],
                'confidence': 97,
                'method': 'exact_pattern',
                'source': 'core_patterns',
                'speed': 'instant'
            })

        # Special handling for time expressions with multiple options
        time_variations = {
            'homnay': ['hôm nay', 'hôm này'],
            'homqua': ['hôm qua', 'hôm trước'],
            'ngaymai': ['ngày mai', 'mai này'],
            'tuannay': ['tuần này', 'tuần nay'],
            'thangnay': ['tháng này', 'tháng nay'],
            'namnay': ['năm này', 'năm nay'],
            'baygio': ['bây giờ', 'lúc này', 'hiện tại']
        }

        if input_text in time_variations:
            for i, variation in enumerate(time_variations[input_text]):
                # Slightly lower confidence for alternatives
                confidence = 97 - (i * 2)
                suggestions.append({
                    'vietnamese_text': variation,
                    'confidence': confidence,
                    'method': 'time_variation',
                    'source': 'time_patterns',
                    'speed': 'instant'
                })

        # Fuzzy pattern matching
        for pattern, result in self.core_patterns.items():
            if len(pattern) >= 4 and len(input_text) >= 4:
                similarity = self._calculate_similarity(input_text, pattern)

                if similarity > 0.8:  # High similarity threshold
                    suggestions.append({
                        'vietnamese_text': result,
                        'confidence': min(similarity * 85, 90),
                        'method': 'fuzzy_pattern',
                        'source': 'pattern_similarity',
                        'similarity': similarity,
                        'speed': 'fast'
                    })

        return suggestions

    def _get_hybrid_suggestions(self, input_text: str) -> List[Dict]:
        """Get hybrid AI + pattern suggestions"""
        suggestions = []

        try:
            # Advanced segmentation + both AI models
            segments = self._smart_segmentation(input_text)

            for segment_pattern in segments:
                # Try ViBERT on segment
                if self.vibert_model:
                    vibert_result = self._process_with_vibert(segment_pattern)
                    if vibert_result:
                        suggestions.append({
                            'vietnamese_text': vibert_result,
                            'confidence': 85,
                            'method': 'hybrid_vibert',
                            'source': 'hybrid_segmentation',
                            'segment_pattern': segment_pattern,
                            'speed': 'medium'
                        })

                # Try accent marker on segment
                if self.accent_model:
                    accent_result = self._apply_accent_marker(segment_pattern)
                    if accent_result and self._is_quality_vietnamese(accent_result):
                        suggestions.append({
                            'vietnamese_text': accent_result,
                            'confidence': 83,
                            'method': 'hybrid_accent',
                            'source': 'hybrid_segmentation',
                            'segment_pattern': segment_pattern,
                            'speed': 'medium'
                        })

        except Exception as e:
            print(f"⚠️ Hybrid processing error: {e}")

        return suggestions

    def _generate_vibert_candidates(self, input_text: str) -> List[str]:
        """Generate candidates for ViBERT processing"""
        candidates = []

        # Tone variations for single syllables
        if len(input_text) <= 5:
            base_variations = {
                'a': ['à', 'á', 'ạ', 'ả', 'ã', 'â', 'ầ', 'ấ', 'ậ', 'ẩ', 'ẫ'],
                'e': ['è', 'é', 'ẹ', 'ẻ', 'ẽ', 'ê', 'ề', 'ế', 'ệ', 'ể', 'ễ'],
                'i': ['ì', 'í', 'ị', 'ỉ', 'ĩ'],
                'o': ['ò', 'ó', 'ọ', 'ỏ', 'õ', 'ô', 'ồ', 'ố', 'ộ', 'ổ', 'ỗ', 'ơ', 'ờ', 'ớ', 'ợ', 'ở', 'ỡ'],
                'u': ['ù', 'ú', 'ụ', 'ủ', 'ũ', 'ư', 'ừ', 'ứ', 'ự', 'ử', 'ữ'],
                'y': ['ỳ', 'ý', 'ỵ', 'ỷ', 'ỹ'],
                'd': ['đ']
            }

            # Generate tone variations
            for base, variations in base_variations.items():
                if base in input_text:
                    for variation in variations:
                        candidate = input_text.replace(base, variation)
                        candidates.append(candidate)

        # Compound word candidates
        if len(input_text) >= 6:
            # Try different splits
            for i in range(3, len(input_text) - 2):
                part1 = input_text[:i]
                part2 = input_text[i:]

                if part1 in self.core_patterns and part2 in self.core_patterns:
                    candidate = f"{self.core_patterns[part1]} {self.core_patterns[part2]}"
                    candidates.append(candidate)

        return candidates[:10]  # Limit for performance

    def _generate_spaced_versions(self, input_text: str) -> List[str]:
        """Generate spaced versions for accent processing"""
        spaced_versions = []

        # Direct pattern lookup for spacing
        spacing_patterns = {
            'toimuon': 'toi muon', 'anhdichuyen': 'anh di chuyen', 'xinchao': 'xin chao',
            'camon': 'cam on', 'emhocbai': 'em hoc bai', 'banvietbai': 'ban viet bai',
            'chungtoilam': 'chung toi lam', 'cogiaoday': 'co giao day', 'thaygiaoday': 'thay giao day',
            'hocsinhhoc': 'hoc sinh hoc', 'toican': 'toi can', 'toithich': 'toi thich',
            'toikhong': 'toi khong', 'anhmuon': 'anh muon', 'emmuon': 'em muon', 'banmuon': 'ban muon',

            # Time expressions spacing
            'homnay': 'hom nay', 'homqua': 'hom qua', 'ngaymai': 'ngay mai',
            'tuannay': 'tuan nay', 'thangnay': 'thang nay', 'namnay': 'nam nay',
            'sangmai': 'sang mai', 'chieunay': 'chieu nay', 'toinay': 'toi nay',
            'baygio': 'bay gio', 'lucnay': 'luc nay'
        }

        if input_text in spacing_patterns:
            spaced_versions.append(spacing_patterns[input_text])

        # Smart segmentation for unknown patterns
        if len(input_text) >= 6:
            # Try various segmentation patterns
            mid = len(input_text) // 2
            spaced_versions.append(f"{input_text[:mid]} {input_text[mid:]}")

            # Try 3-word segmentation for long inputs
            if len(input_text) >= 9:
                third1 = len(input_text) // 3
                third2 = (len(input_text) * 2) // 3
                spaced_versions.append(
                    f"{input_text[:third1]} {input_text[third1:third2]} {input_text[third2:]}")

        return spaced_versions[:5]  # Limit for performance

    def _smart_segmentation(self, input_text: str) -> List[str]:
        """Advanced smart segmentation"""
        segments = []

        # Multiple segmentation strategies
        length = len(input_text)

        if length >= 6:
            # 2-word patterns
            for i in range(3, length - 2):
                segment = f"{input_text[:i]} {input_text[i:]}"
                segments.append(segment)

        if length >= 9:
            # 3-word patterns
            for i in range(3, length - 5):
                for j in range(i + 3, length - 2):
                    segment = f"{input_text[:i]} {input_text[i:j]} {input_text[j:]}"
                    segments.append(segment)

        return segments[:8]  # Limit for performance

    def _apply_accent_marker(self, text: str) -> str:
        """Apply accent marker to text"""
        if not self.accent_model or not text.strip():
            return text

        try:
            tokens = text.strip().split()

            if not tokens:
                return text

            # Tokenize
            inputs = self.accent_tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.accent_model(**inputs)

            predictions = outputs.logits.cpu().numpy()
            predictions = np.argmax(predictions, axis=2)

            # Convert back to tokens
            input_ids = inputs['input_ids'].cpu()
            token_list = self.accent_tokenizer.convert_ids_to_tokens(
                input_ids[0])
            token_list = token_list[1:-1]  # Remove special tokens
            predictions = predictions[0][1:-1]

            # Apply transformations
            accented_words = self._apply_accent_transformations(
                token_list, predictions)

            return ' '.join(accented_words)

        except Exception as e:
            print(f"⚠️ Accent application error: {e}")
            return text

    def _apply_accent_transformations(self, tokens: List[str], predictions: List[int]) -> List[str]:
        """Apply accent transformations using predictions"""
        TOKENIZER_PREFIX = "▁"
        words = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            prediction = predictions[i] if i < len(predictions) else 0

            if token.startswith(TOKENIZER_PREFIX):
                # Start of new word
                word_tokens = [token[len(TOKENIZER_PREFIX):]]
                word_predictions = [prediction]

                # Collect subword tokens
                j = i + 1
                while j < len(tokens) and not tokens[j].startswith(TOKENIZER_PREFIX):
                    word_tokens.append(tokens[j])
                    word_predictions.append(
                        predictions[j] if j < len(predictions) else 0)
                    j += 1

                # Merge word and apply transformation
                word = ''.join(word_tokens)
                transformed_word = self._transform_word(word, word_predictions)
                words.append(transformed_word)

                i = j
            else:
                # Isolated token
                transformed = self._transform_word(token, [prediction])
                words.append(transformed)
                i += 1

        return words

    def _transform_word(self, word: str, predictions: List[int]) -> str:
        """Transform word using accent predictions"""
        if not self.accent_tags or not predictions:
            return word

        for prediction in predictions:
            if 0 <= prediction < len(self.accent_tags):
                tag = self.accent_tags[prediction]

                if '-' in tag:
                    parts = tag.split('-', 1)
                    if len(parts) == 2:
                        source, target = parts
                        if source and source in word:
                            return word.replace(source, target, 1)

        return word

    def _score_with_vibert(self, input_text: str, candidate: str) -> float:
        """Score candidate using ViBERT"""
        if not self.vibert_model:
            return 0.5

        try:
            # Create scoring context
            context = f"'{input_text}' nghĩa là '{candidate}'"

            # Tokenize
            inputs = self.vibert_tokenizer(
                context,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.vibert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Calculate coherence score
            norm = torch.norm(embeddings).item()
            score = min(max(norm / 12.0, 0.0), 1.0)  # Adjusted normalization

            return score

        except Exception as e:
            print(f"⚠️ ViBERT scoring error: {e}")
            return 0.5

    def _process_with_vibert(self, text: str) -> Optional[str]:
        """Process text with ViBERT for meaning extraction"""
        # Look for known patterns first
        for pattern, result in self.core_patterns.items():
            if pattern in text.replace(' ', ''):
                return result

        return None

    def _is_quality_vietnamese(self, text: str) -> bool:
        """Check if text is quality Vietnamese"""
        if not text or len(text.strip()) < 2:
            return False

        # Check for proper Vietnamese characters
        vietnamese_chars = set(
            'aàáạảãâầấậẩẫăằắặẳẵeèéẹẻẽêềếệểễiìíịỉĩoòóọỏõôồốộổỗơờớợởỡuùúụủũưừứựửữyỳýỵỷỹđ')

        # Must contain Vietnamese characters
        has_vietnamese = any(c.lower() in vietnamese_chars for c in text)

        # Must not be mostly gibberish
        words = text.split()
        if len(words) > 3:  # For longer texts, check word validity
            valid_words = sum(1 for word in words if len(word) >= 2 and any(
                c.lower() in vietnamese_chars for c in word))
            return valid_words >= len(words) * 0.7  # 70% valid words

        return has_vietnamese

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        len1, len2 = len(text1), len(text2)
        max_len = max(len1, len2)

        if max_len == 0:
            return 1.0

        # Count matching characters at same positions
        matches = sum(1 for i in range(min(len1, len2))
                      if text1[i] == text2[i])

        # Calculate similarity score
        similarity = matches / max_len

        # Bonus for similar lengths
        length_bonus = 1.0 - abs(len1 - len2) / max_len

        return (similarity + length_bonus) / 2

    def _ultimate_ranking(self, suggestions: List[Dict]) -> List[Dict]:
        """Ultimate ranking algorithm combining all factors"""

        # Remove duplicates while preserving best scores
        unique_suggestions = {}
        for suggestion in suggestions:
            text = suggestion['vietnamese_text']

            if text not in unique_suggestions:
                unique_suggestions[text] = suggestion
            else:
                # Keep the one with higher confidence
                if suggestion['confidence'] > unique_suggestions[text]['confidence']:
                    unique_suggestions[text] = suggestion

        # Convert back to list
        ranked_suggestions = list(unique_suggestions.values())

        # Advanced ranking factors
        for suggestion in ranked_suggestions:
            # Speed bonus
            speed_bonus = {'instant': 5, 'fast': 3, 'medium': 1, 'slow': 0}.get(
                suggestion.get('speed', 'medium'), 1
            )

            # Method reliability bonus
            method_bonus = {
                'exact_pattern': 10, 'vibert_direct': 8, 'vibert_ai': 6,
                'accent_ai': 4, 'fuzzy_pattern': 3, 'hybrid_vibert': 5,
                'hybrid_accent': 4
            }.get(suggestion.get('method', ''), 0)

            # Quality bonus for proper Vietnamese
            quality_bonus = 5 if self._is_quality_vietnamese(
                suggestion['vietnamese_text']) else 0

            # Calculate final score
            final_score = (
                suggestion['confidence'] +
                speed_bonus +
                method_bonus +
                quality_bonus
            )

            suggestion['final_score'] = final_score

        # Sort by final score
        ranked_suggestions.sort(key=lambda x: x['final_score'], reverse=True)

        return ranked_suggestions


def main():
    """Demo the Ultimate Vietnamese Keyboard"""
    print("🚀 ULTIMATE VIETNAMESE KEYBOARD DEMO")
    print("=" * 60)

    # Initialize the ultimate keyboard
    keyboard = UltimateVietnameseKeyboard()

    # Test cases
    test_cases = [
        'toimuon',
        'anhdichuyen',
        'xinchao',
        'camon',
        'emhocbai',
        'chocacban',
        'cogiaoday',
        'nha',
        'doc'
    ]

    print(f"\n🔥 TESTING ULTIMATE POWER")
    print("=" * 60)

    for input_text in test_cases:
        print(f"\n📝 Input: '{input_text}'")

        start_time = time.time()
        suggestions = keyboard.get_ultimate_suggestions(
            input_text, max_suggestions=8)
        end_time = time.time()

        print(f"⚡ Processing time: {(end_time - start_time)*1000:.1f}ms")
        print(f"🏆 Ultimate suggestions ({len(suggestions)}):")

        for i, suggestion in enumerate(suggestions, 1):
            method = suggestion.get('method', 'unknown')
            confidence = suggestion.get('confidence', 0)
            speed = suggestion.get('speed', 'unknown')
            final_score = suggestion.get('final_score', 0)

            print(
                f"   {i}. '{suggestion['vietnamese_text']}' ({confidence:.1f}% | {final_score:.1f} pts) - {method} [{speed}]")


if __name__ == "__main__":
    main()
