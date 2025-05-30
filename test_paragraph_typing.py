#!/usr/bin/env python3
"""
Test Gõ Đoạn Văn Tiếng Việt
Comprehensive testing for paragraph typing with Vietnamese AI Keyboard
"""

from typing import List, Dict, Tuple
from ml.hybrid_suggestions import VietnameseHybridSuggestions
from ml.word_segmentation import VietnameseWordSegmenter
import sys
import os
sys.path.append('.')


class ParagraphTypingTest:
    """Test system for Vietnamese paragraph typing"""

    def __init__(self):
        print("🚀 Initializing Vietnamese AI Keyboard Test System...")
        self.segmenter = VietnameseWordSegmenter()
        self.hybrid = VietnameseHybridSuggestions()
        print("✅ System ready!")

        # Đoạn văn test
        self.target_paragraph = """Buổi sáng sớm, mặt trời dần nhô lên khỏi đường chân trời, nhuộm vàng cả bầu trời và mặt nước. Những cơn gió nhẹ mang theo hương hoa thoang thoảng, khiến lòng người trở nên nhẹ nhàng và thư thái. Ai nấy đều bắt đầu một ngày mới với hy vọng và niềm vui nhỏ bé trong tim."""

        # Break down thành các level test
        self.individual_words = [
            "buoi", "sang", "som", "mat", "troi", "dan", "nho", "len", "khoi",
            "duong", "chan", "troi", "nhuom", "vang", "ca", "bau", "troi", "va",
            "mat", "nuoc", "nhung", "con", "gio", "nhe", "mang", "theo", "huong",
            "hoa", "thoang", "thoang", "khien", "long", "nguoi", "tro", "nen",
            "nhe", "nhang", "va", "thu", "thai", "ai", "nay", "deu", "bat", "dau",
            "mot", "ngay", "moi", "voi", "hy", "vong", "va", "niem", "vui", "nho",
            "be", "trong", "tim"
        ]

        self.word_phrases = [
            "buoisang", "sangsom", "mattroi", "dannho", "nholen", "khoihuong",
            "duongchan", "chantroi", "nhuomvang", "bautroi", "matnuoc",
            "nhungcon", "congio", "gionhe", "mangtheo", "huonghoa",
            "thoangfloang", "khienlong", "longnguoi", "tronen", "nhenhang",
            "thuthai", "ainay", "naydeu", "batdau", "motngay", "ngaymoi",
            "hyvong", "niemvui", "nhobe", "trongtin"
        ]

        self.full_phrases = [
            "buoisangsom", "mattroidannho", "khoihuongchantroi",
            "nhuomvangcabautroi", "nhungcongionghe", "mangtheohuonghoa",
            "khienlongnguoi", "nhenhangvathuthai", "ainaydeuvatdau",
            "motngaymoivoihyvong", "niemvuinhobetrongtim"
        ]

        self.sentences = [
            "buoisangsom mattroi dannho len khoi duong chan troi",
            "nhuom vang ca bau troi va mat nuoc",
            "nhung con gio nhe mang theo huong hoa thoang thoang",
            "khien long nguoi tro nen nhe nhang va thu thai",
            "ai nay deu bat dau mot ngay moi voi hy vong",
            "va niem vui nho be trong tim"
        ]

    def remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents for testing input"""
        accent_map = {
            'àáạảãâầấậẩẫăằắặẳẵ': 'a',
            'èéẹẻẽêềếệểễ': 'e',
            'ìíịỉĩ': 'i',
            'òóọỏõôồốộổỗơờớợởỡ': 'o',
            'ùúụủũưừứựửữ': 'u',
            'ỳýỵỷỹ': 'y',
            'đ': 'd'
        }

        result = text.lower()
        for accented_chars, non_accented in accent_map.items():
            for char in accented_chars:
                result = result.replace(char, non_accented)
        return result

    def test_individual_words(self) -> Dict:
        """Test gõ từng từ đơn lẻ"""
        print("\n🔤 TEST 1: INDIVIDUAL WORDS")
        print("=" * 50)

        results = []
        success_count = 0

        for word in self.individual_words[:15]:  # Test 15 từ đầu
            # Get suggestions
            suggestions = self.hybrid.get_suggestions(word, max_suggestions=3)

            if suggestions and suggestions[0]['confidence'] > 0.1:
                top_suggestion = suggestions[0]['word']
                confidence = suggestions[0]['confidence']
                success = True
                success_count += 1
                status = "✅"
            else:
                top_suggestion = word
                confidence = 0.0
                success = False
                status = "❌"

            print(f"{status} '{word}' → '{top_suggestion}' ({confidence:.1%})")
            results.append({
                'input': word,
                'output': top_suggestion,
                'confidence': confidence,
                'success': success
            })

        success_rate = success_count / len(results) * 100
        print(f"\n📊 Individual Words Success Rate: {success_rate:.1f}%")

        return {'results': results, 'success_rate': success_rate}

    def test_word_phrases(self) -> Dict:
        """Test gõ cụm từ"""
        print("\n🔗 TEST 2: WORD PHRASES")
        print("=" * 50)

        results = []
        success_count = 0

        for phrase in self.word_phrases[:10]:  # Test 10 phrase đầu
            # Try word segmentation first
            segmented = self.segmenter.segment_text(phrase)

            if segmented != phrase:
                # Segmentation worked
                output = segmented
                success = True
                success_count += 1
                status = "✅"
                method = "segmentation"
            else:
                # Try suggestions
                suggestions = self.hybrid.get_suggestions(
                    phrase, max_suggestions=1)
                if suggestions and suggestions[0]['confidence'] > 0.2:
                    output = suggestions[0]['word']
                    success = True
                    success_count += 1
                    status = "✅"
                    method = "suggestion"
                else:
                    output = phrase
                    success = False
                    status = "❌"
                    method = "none"

            print(f"{status} '{phrase}' → '{output}' ({method})")
            results.append({
                'input': phrase,
                'output': output,
                'success': success,
                'method': method
            })

        success_rate = success_count / len(results) * 100
        print(f"\n📊 Word Phrases Success Rate: {success_rate:.1f}%")

        return {'results': results, 'success_rate': success_rate}

    def test_full_phrases(self) -> Dict:
        """Test gõ cụm từ dài"""
        print("\n📝 TEST 3: FULL PHRASES")
        print("=" * 50)

        results = []
        success_count = 0

        for phrase in self.full_phrases[:8]:  # Test 8 phrase đầu
            # Try word segmentation
            segmented = self.segmenter.segment_text(phrase)

            if segmented != phrase and ' ' in segmented:
                # Good segmentation
                output = segmented
                success = True
                success_count += 1
                status = "✅"
                method = "segmentation"
            else:
                # Try as smaller chunks
                chunks = []
                remaining = phrase
                while remaining and len(remaining) > 2:
                    # Try to segment progressively
                    best_chunk = remaining[:5] if len(
                        remaining) >= 5 else remaining
                    chunk_segmented = self.segmenter.segment_text(best_chunk)
                    if chunk_segmented != best_chunk:
                        chunks.append(chunk_segmented)
                        remaining = remaining[len(best_chunk):]
                    else:
                        chunks.append(best_chunk)
                        remaining = remaining[len(best_chunk):]

                if chunks:
                    output = ' '.join(chunks)
                    success = len(chunks) > 1
                    if success:
                        success_count += 1
                    status = "✅" if success else "🟡"
                    method = "chunked"
                else:
                    output = phrase
                    success = False
                    status = "❌"
                    method = "failed"

            print(f"{status} '{phrase}' → '{output}' ({method})")
            results.append({
                'input': phrase,
                'output': output,
                'success': success,
                'method': method
            })

        success_rate = success_count / len(results) * 100
        print(f"\n📊 Full Phrases Success Rate: {success_rate:.1f}%")

        return {'results': results, 'success_rate': success_rate}

    def test_sentences(self) -> Dict:
        """Test gõ câu hoàn chỉnh"""
        print("\n📖 TEST 4: COMPLETE SENTENCES")
        print("=" * 50)

        results = []
        success_count = 0

        for sentence in self.sentences:
            print(f"\n🔤 Input: '{sentence}'")

            # Split into words and process each
            words = sentence.split()
            processed_words = []

            for word in words:
                # Try segmentation first
                segmented = self.segmenter.segment_text(word)
                if segmented != word:
                    processed_words.append(segmented)
                else:
                    # Try suggestions
                    suggestions = self.hybrid.get_suggestions(
                        word, max_suggestions=1)
                    if suggestions and suggestions[0]['confidence'] > 0.1:
                        processed_words.append(suggestions[0]['word'])
                    else:
                        processed_words.append(word)

            output = ' '.join(processed_words)

            # Check success (has some improvements)
            improvements = sum(1 for orig, proc in zip(
                words, processed_words) if orig != proc)
            # At least 30% words improved
            success = improvements > len(words) * 0.3

            if success:
                success_count += 1
            status = "✅" if success else "🟡"

            print(f"{status} '{output}'")
            print(f"   Improvements: {improvements}/{len(words)} words")

            results.append({
                'input': sentence,
                'output': output,
                'improvements': improvements,
                'total_words': len(words),
                'success': success
            })

        success_rate = success_count / len(results) * 100
        print(f"\n📊 Sentences Success Rate: {success_rate:.1f}%")

        return {'results': results, 'success_rate': success_rate}

    def run_complete_test(self):
        """Chạy test toàn bộ"""
        print("🎯 VIETNAMESE PARAGRAPH TYPING TEST")
        print("=" * 60)
        print(f"Target text: {self.target_paragraph[:100]}...")
        print()

        # Run all tests
        test1 = self.test_individual_words()
        test2 = self.test_word_phrases()
        test3 = self.test_full_phrases()
        test4 = self.test_sentences()

        # Overall results
        print("\n🏆 OVERALL RESULTS")
        print("=" * 50)

        overall_score = (test1['success_rate'] + test2['success_rate'] +
                         test3['success_rate'] + test4['success_rate']) / 4

        print(f"Individual Words: {test1['success_rate']:.1f}%")
        print(f"Word Phrases: {test2['success_rate']:.1f}%")
        print(f"Full Phrases: {test3['success_rate']:.1f}%")
        print(f"Complete Sentences: {test4['success_rate']:.1f}%")
        print(f"OVERALL SCORE: {overall_score:.1f}%")

        # Assessment
        if overall_score >= 80:
            assessment = "🟢 EXCELLENT - Ready for complex text typing"
        elif overall_score >= 60:
            assessment = "🟡 GOOD - Suitable for basic paragraph typing"
        elif overall_score >= 40:
            assessment = "🟠 FAIR - Needs improvement for fluent typing"
        else:
            assessment = "🔴 POOR - Major improvements needed"

        print(f"\nAssessment: {assessment}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if test1['success_rate'] < 70:
            print("   - Expand vocabulary for individual words")
        if test2['success_rate'] < 70:
            print("   - Add more word combination mappings")
        if test3['success_rate'] < 70:
            print("   - Improve segmentation for long phrases")
        if test4['success_rate'] < 70:
            print("   - Enhance contextual processing")

        return {
            'individual_words': test1,
            'word_phrases': test2,
            'full_phrases': test3,
            'sentences': test4,
            'overall_score': overall_score
        }


def main():
    """Main function"""
    try:
        tester = ParagraphTypingTest()
        results = tester.run_complete_test()

        print(f"\n✅ Test completed successfully!")
        print(f"Final Score: {results['overall_score']:.1f}%")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
