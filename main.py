"""
Main entry point cho bàn phím recommend tiếng Việt
"""

import sys
import os
from core import TextProcessor, Dictionary, Recommender


def test_phase_1():
    """
    Test Phase 1: Core functionality
    """
    print("=" * 60)
    print("PHASE 1 TEST - CORE FUNCTIONALITY")
    print("=" * 60)
    
    # Test TextProcessor
    print("\n1. Testing TextProcessor...")
    processor = TextProcessor()
    
    test_texts = [
        "xin chào mọi người",
        "tôi học tiếng việt",
        "hôm nay trời đẹp"
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        print(f"  No accents: {processor.remove_accents(text)}")
        print(f"  Tokenized: {processor.tokenize(text)}")
    
    # Test Dictionary
    print("\n\n2. Testing Dictionary...")
    dictionary = Dictionary()
    print(f"Dictionary stats: {dictionary.get_stats()}")
    
    test_queries = ["xin", "chao", "xinchao", "moinguoi"]
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = dictionary.search_comprehensive(query, max_results=3)
        for result, confidence, match_type in results:
            print(f"  {result} (confidence: {confidence:.2f}, type: {match_type})")
    
    # Test Recommender
    print("\n\n3. Testing Recommender...")
    recommender = Recommender()
    
    test_inputs = [
        "xinchao",
        "xinchaomoinguoi", 
        "toihoc",
        "chucmung"
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        recommendations = recommender.recommend_smart(test_input, max_suggestions=3)
        for i, (text, confidence, rec_type) in enumerate(recommendations, 1):
            print(f"  {i}. {text} (confidence: {confidence:.3f}, type: {rec_type})")


def interactive_demo():
    """
    Demo tương tác với user
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    print("Nhập text không dấu để xem gợi ý (nhập 'quit' để thoát)")
    print("Ví dụ: xinchao, toihoc, moinguoi, etc.")
    
    recommender = Recommender()
    context = []
    
    while True:
        try:
            user_input = input("\n📝 Nhập: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Tạm biệt! 👋")
                break
            
            if not user_input:
                continue
            
            # Lấy gợi ý
            recommendations = recommender.recommend_smart(user_input, context, max_suggestions=5)
            
            if recommendations:
                print("💡 Gợi ý:")
                for i, (text, confidence, rec_type) in enumerate(recommendations, 1):
                    confidence_bar = "█" * int(confidence * 10)
                    print(f"  {i}. {text} [{confidence_bar:10}] ({confidence:.2f})")
                
                # Cho user chọn
                try:
                    choice = input("\n🔘 Chọn số (hoặc Enter để bỏ qua): ").strip()
                    if choice.isdigit():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(recommendations):
                            chosen_text = recommendations[choice_idx][0]
                            print(f"✅ Bạn chọn: '{chosen_text}'")
                            
                            # Cập nhật context và learning
                            context.extend(chosen_text.split())
                            recommender.update_user_choice(chosen_text, context)
                            
                            # Giới hạn context length
                            if len(context) > 10:
                                context = context[-10:]
                except ValueError:
                    pass
            else:
                print("❌ Không tìm thấy gợi ý phù hợp")
                
        except KeyboardInterrupt:
            print("\n\nTạm biệt! 👋")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


def run_ui():
    """
    Chạy giao diện đồ họa
    """
    try:
        from ui import KeyboardUI
        print("🚀 Khởi động giao diện đồ họa...")
        app = KeyboardUI()
        app.run()
    except ImportError as e:
        print(f"❌ Lỗi import UI: {e}")
        print("Hãy chắc chắn rằng tkinter đã được cài đặt")
    except Exception as e:
        print(f"❌ Lỗi UI: {e}")


def main():
    """
    Main function
    """
    print("🚀 BÀN PHÍM RECOMMEND TIẾNG VIỆT")
    print("Version: 1.0.0 - Phase 2")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_phase_1()
        elif sys.argv[1] == "demo":
            interactive_demo()
        elif sys.argv[1] == "ui":
            run_ui()
        else:
            print("Usage: python main.py [test|demo|ui]")
            print("  test - Chạy tests cơ bản")
            print("  demo - Demo tương tác dòng lệnh")
            print("  ui   - Chạy giao diện đồ họa")
    else:
        # Mặc định chạy UI
        print("\n🎯 Chọn chế độ:")
        print("1. Test cơ bản")
        print("2. Demo dòng lệnh")
        print("3. Giao diện đồ họa (khuyến nghị)")
        
        try:
            choice = input("\nChọn (1/2/3): ").strip()
            if choice == "1":
                test_phase_1()
            elif choice == "2":
                interactive_demo()
            elif choice == "3" or choice == "":
                run_ui()
            else:
                print("Lựa chọn không hợp lệ. Chạy giao diện đồ họa...")
                run_ui()
        except KeyboardInterrupt:
            print("\nTạm biệt! 👋")


if __name__ == "__main__":
    main() 