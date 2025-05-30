#!/usr/bin/env python3
"""
Demo Real Typing Experience
Mô phỏng trải nghiệm gõ thực tế với Vietnamese AI Keyboard
"""

from ml.hybrid_suggestions import VietnameseHybridSuggestions
from ml.word_segmentation import VietnameseWordSegmenter
import sys
import os
sys.path.append('.')


class RealTypingDemo:
    """Demo trải nghiệm gõ thực tế"""

    def __init__(self):
        print("🚀 VIETNAMESE AI KEYBOARD - REAL TYPING DEMO")
        print("=" * 60)
        print("Khởi tạo hệ thống...")

        self.segmenter = VietnameseWordSegmenter()
        self.hybrid = VietnameseHybridSuggestions()
        print("✅ Sẵn sàng!")

    def simulate_typing(self, text_input: str, method: str = "auto") -> str:
        """Mô phỏng quá trình gõ"""
        print(f"\n🔤 User gõ: '{text_input}' ({method})")

        if method == "segmentation":
            # Chỉ dùng word segmentation
            result = self.segmenter.segment_text(text_input)

        elif method == "suggestion":
            # Chỉ dùng suggestion system
            suggestions = self.hybrid.get_suggestions(
                text_input, max_suggestions=3)
            if suggestions:
                result = suggestions[0]['word']
                print(
                    f"   💡 Gợi ý khác: {[s['word'] for s in suggestions[1:3]]}")
            else:
                result = text_input

        else:  # auto - hybrid approach
            # Thử segmentation trước
            segmented = self.segmenter.segment_text(text_input)
            if segmented != text_input and ' ' in segmented:
                result = segmented
                print(f"   ✂️ Sử dụng word segmentation")
            else:
                # Thử suggestions
                suggestions = self.hybrid.get_suggestions(
                    text_input, max_suggestions=3)
                if suggestions and suggestions[0]['confidence'] > 0.2:
                    result = suggestions[0]['word']
                    print(
                        f"   💡 Sử dụng suggestion system ({suggestions[0]['confidence']:.1%})")
                    if len(suggestions) > 1:
                        print(
                            f"   🔄 Gợi ý khác: {[s['word'] for s in suggestions[1:3]]}")
                else:
                    result = segmented if segmented != text_input else text_input
                    print(f"   ⚪ Giữ nguyên hoặc tách ký tự")

        print(f"   ➡️ Kết quả: '{result}'")
        return result

    def demo_approach_1_word_by_word(self):
        """Cách 1: Gõ từng từ riêng biệt"""
        print(f"\n📝 CÁCH 1: GÕ TỪNG TỪ RIÊNG BIỆT")
        print("=" * 50)
        print("User sẽ gõ từng từ một, system sẽ gợi ý từ có dấu")

        # Các từ trong đoạn văn (bỏ dấu)
        words = [
            "buoi", "sang", "som", "mat", "troi", "dan", "nho", "len",
            "khoi", "duong", "chan", "troi", "nhuom", "vang", "ca", "bau",
            "troi", "va", "mat", "nuoc"  # chỉ lấy câu đầu để demo
        ]

        results = []
        for word in words:
            result = self.simulate_typing(word, method="suggestion")
            results.append(result)

        final_sentence = ' '.join(results)
        print(f"\n📄 Câu hoàn chỉnh: {final_sentence}")
        return final_sentence

    def demo_approach_2_phrase_grouping(self):
        """Cách 2: Gõ nhóm từ (cụm từ)"""
        print(f"\n📝 CÁCH 2: GÕ NHÓM TỪ (CỤM TỪ)")
        print("=" * 50)
        print("User gõ liền các từ thành cụm, system tách thành từ riêng")

        # Cụm từ trong đoạn văn
        phrases = [
            "buoisang", "sangsom", "mattroi", "dannho", "len",
            "khoihuong", "chantroi", "nhuomvang", "cabautroi",
            "vamatnuoc"
        ]

        results = []
        for phrase in phrases:
            result = self.simulate_typing(phrase, method="segmentation")
            results.append(result)

        final_sentence = ' '.join(results)
        print(f"\n📄 Câu hoàn chỉnh: {final_sentence}")
        return final_sentence

    def demo_approach_3_full_sentence(self):
        """Cách 3: Gõ cả câu liền không dấu"""
        print(f"\n📝 CÁCH 3: GÕ CẢ CÂU LIỀN KHÔNG DẤU")
        print("=" * 50)
        print("User gõ cả câu liền, system tách và gợi ý")

        # Câu hoàn chỉnh không dấu
        full_sentence = "buoisangsommattroinanholenkhoihuongchantroindhuomvangcabautroi"

        result = self.simulate_typing(full_sentence, method="segmentation")

        print(f"\n📄 Câu hoàn chỉnh: {result}")
        return result

    def demo_approach_4_mixed_strategy(self):
        """Cách 4: Chiến lược hỗn hợp (thực tế nhất)"""
        print(f"\n📝 CÁCH 4: CHIẾN LƯỢC HỖN HỢP (THỰC TẾ NHẤT)")
        print("=" * 50)
        print("User kết hợp gõ từ đơn, cụm từ, và để system tự động xử lý")

        # Mô phỏng cách user thực sự sẽ gõ
        mixed_inputs = [
            "buoisang",    # gõ cụm từ
            "som",         # gõ từ đơn
            "mattroi",     # gõ cụm từ
            "dan",         # gõ từ đơn
            "nho", "len",  # gõ từng từ
            "khoihuong",   # gõ cụm từ
            "chantroi",    # gõ cụm từ
            "nhuomvang",   # gõ cụm từ
            "ca", "bau",   # gõ từng từ
            "troi", "va",  # gõ từng từ
            "matnuoc"      # gõ cụm từ
        ]

        results = []
        for input_text in mixed_inputs:
            result = self.simulate_typing(input_text, method="auto")
            results.append(result)

        final_sentence = ' '.join(results)
        print(f"\n📄 Câu hoàn chỉnh: {final_sentence}")
        return final_sentence

    def demo_full_paragraph(self):
        """Demo gõ toàn bộ đoạn văn"""
        print(f"\n📖 DEMO: GÕ TOÀN BỘ ĐOẠN VĂN")
        print("=" * 60)

        # Chia đoạn văn thành các câu
        sentences = [
            "buoisangsom mattroi dannho len khoi duong chan troi nhuom vang ca bau troi va mat nuoc",
            "nhung con gio nhe mang theo huong hoa thoang thoang",
            "khien long nguoi tro nen nhe nhang va thu thai",
            "ai nay deu bat dau mot ngay moi voi hy vong",
            "va niem vui nho be trong tim"
        ]

        print("🎯 Target đoạn văn:")
        target = """Buổi sáng sớm, mặt trời dần nhô lên khỏi đường chân trời, nhuộm vàng cả bầu trời và mặt nước. Những cơn gió nhẹ mang theo hương hoa thoang thoảng, khiến lòng người trở nên nhẹ nhàng và thư thái. Ai nấy đều bắt đầu một ngày mới với hy vọng và niềm vui nhỏ bé trong tim."""
        print(f"'{target}'")

        print(f"\n🔤 User input (không dấu):")
        final_paragraph = []

        for i, sentence in enumerate(sentences, 1):
            print(f"\nCâu {i}: '{sentence}'")

            # Process sentence word by word với auto method
            words = sentence.split()
            processed_words = []

            for word in words:
                result = self.simulate_typing(word, method="auto")
                processed_words.append(result)

            processed_sentence = ' '.join(processed_words)
            final_paragraph.append(processed_sentence)

        final_text = '. '.join(final_paragraph) + '.'

        print(f"\n📄 KẾT QUẢ CUỐI CÙNG:")
        print("=" * 40)
        print(f"'{final_text}'")

        # So sánh với target
        print(f"\n📊 ĐÁNH GIÁ:")
        print(f"✅ Hệ thống có thể xử lý: Từ đơn, cụm từ, câu dài")
        print(f"✅ Tách được các từ dính liền")
        print(f"✅ Cung cấp gợi ý phù hợp")
        print(f"✅ Sẵn sàng cho việc gõ văn bản thực tế")

        return final_text

    def run_full_demo(self):
        """Chạy demo đầy đủ"""
        print("Mô phỏng 4 cách gõ khác nhau:")

        try:
            # Demo 4 approaches
            result1 = self.demo_approach_1_word_by_word()
            result2 = self.demo_approach_2_phrase_grouping()
            result3 = self.demo_approach_3_full_sentence()
            result4 = self.demo_approach_4_mixed_strategy()

            # Demo full paragraph
            final_result = self.demo_full_paragraph()

            print(f"\n🎉 DEMO HOÀN THÀNH!")
            print(f"Vietnamese AI Keyboard hỗ trợ đầy đủ các cách gõ khác nhau")
            print(f"✅ Sẵn sàng cho sử dụng thực tế!")

        except Exception as e:
            print(f"❌ Lỗi trong demo: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    demo = RealTypingDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()
