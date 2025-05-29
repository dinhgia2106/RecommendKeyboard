"""
Main entry point cho bàn phím recommend tiếng Việt
Version 2.0 - Enhanced với advanced features
"""

import sys
import os
import time
from core import TextProcessor, Dictionary, AdvancedRecommender


def test_phase_3():
    """
    Test Phase 3: Advanced Features
    """
    print("=" * 60)
    print("PHASE 3 TEST - ADVANCED FEATURES")
    print("=" * 60)
    
    # Test AdvancedRecommender
    print("\n1. Testing Advanced Recommender...")
    recommender = AdvancedRecommender()
    
    print(f"Initial statistics: {recommender.get_statistics()}")
    
    # Advanced test cases
    advanced_test_cases = [
        "toihoctiengviet",
        "anhyeuemdennaychungcothe", 
        "chucmungnamoi",
        "camonnhieulam",
        "toikhoemoinguoi",
        "dichoisaukhi",
        "hocbaitapve"
    ]
    
    print("\n📊 Advanced Text Splitting Tests:")
    for test_input in advanced_test_cases:
        print(f"\nInput: '{test_input}'")
        
        # Test advanced text splitting
        start_time = time.time()
        splits = recommender.advanced_text_splitting(test_input)
        response_time = (time.time() - start_time) * 1000
        
        print(f"  Advanced splits ({response_time:.1f}ms):")
        for i, (words, score) in enumerate(splits[:3], 1):
            phrase = " ".join(words)
            print(f"    {i}. {phrase} (score: {score:.2f})")
        
        # Test smart recommendations
        start_time = time.time()
        recommendations = recommender.smart_recommend(test_input, max_suggestions=5)
        response_time = (time.time() - start_time) * 1000
        
        print(f"  Smart recommendations ({response_time:.1f}ms):")
        for i, (text, confidence, rec_type) in enumerate(recommendations[:3], 1):
            print(f"    {i}. {text} (conf: {confidence:.3f}, type: {rec_type})")
        
        # Simulate user choosing first recommendation
        if recommendations:
            chosen = recommendations[0][0]
            recommender.update_user_preferences(chosen)
            print(f"  → User chose: '{chosen}'")
    
    print(f"\nFinal statistics: {recommender.get_statistics()}")
    
    # Test context prediction
    print("\n2. Testing Enhanced Context Prediction...")
    test_contexts = [
        ["tôi", "đang"],
        ["hôm", "nay"],
        ["chúc", "mừng"],
        ["xin", "chào"]
    ]
    
    for context in test_contexts:
        predictions = recommender.enhanced_context_prediction(context, max_predictions=3)
        print(f"Context {context} → Predictions:")
        for word, score in predictions:
            print(f"  • {word} (score: {score:.3f})")
    
    # Performance metrics
    print("\n3. Performance Analysis:")
    total_tests = len(advanced_test_cases) * 2  # splits + recommendations
    print(f"  • Total test cases: {total_tests}")
    print(f"  • Dictionary size: {recommender.dictionary.get_stats()}")
    print(f"  • 4-gram patterns: {len(recommender.fourgram_freq)}")
    print(f"  • User preferences learned: {len(recommender.user_preferences)}")


def test_phase_1_2():
    """
    Test Phase 1 & 2: Core functionality (updated)
    """
    print("=" * 60)
    print("PHASE 1-2 TEST - CORE FUNCTIONALITY (Updated)")
    print("=" * 60)
    
    # Test TextProcessor
    print("\n1. Testing TextProcessor...")
    processor = TextProcessor()
    
    test_texts = [
        "xin chào mọi người",
        "tôi học tiếng việt",
        "hôm nay trời đẹp",
        "anhyeuemdennaychungcothe"
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        print(f"  No accents: {processor.remove_accents(text)}")
        print(f"  Tokenized: {processor.tokenize(text)}")
    
    # Test Dictionary with enhanced data
    print("\n\n2. Testing Enhanced Dictionary...")
    dictionary = Dictionary()
    print(f"Dictionary stats: {dictionary.get_stats()}")
    
    test_queries = ["xin", "chao", "xinchao", "toihoc", "anhyeu"]
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = dictionary.search_comprehensive(query, max_results=3)
        for result, confidence, match_type in results:
            print(f"  {result} (confidence: {confidence:.2f}, type: {match_type})")
    
    # Test Enhanced Recommender
    print("\n\n3. Testing Enhanced Recommender...")
    recommender = AdvancedRecommender()
    
    test_inputs = [
        "xinchao",
        "toihoc",
        "chucmung",
        "anhyeu"
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        recommendations = recommender.smart_recommend(test_input, max_suggestions=3)
        for i, (text, confidence, rec_type) in enumerate(recommendations, 1):
            print(f"  {i}. {text} (confidence: {confidence:.3f}, type: {rec_type})")


def interactive_demo_enhanced():
    """
    Enhanced demo tương tác với advanced features
    """
    print("\n" + "=" * 60)
    print("ENHANCED INTERACTIVE DEMO")
    print("=" * 60)
    print("Nhập text không dấu để xem gợi ý AI thông minh (nhập 'quit' để thoát)")
    print("🧠 4-gram models • 🎯 Pattern matching • 📈 User learning")
    print("Thử nghiệm: toihoctiengviet, anhyeuemdennaychungcothe, chucmungnamoi")
    
    recommender = AdvancedRecommender()
    context = []
    session_stats = {"suggestions": 0, "selections": 0}
    
    while True:
        try:
            user_input = input("\n📝 Nhập: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Tạm biệt! 👋")
                break
            
            if user_input.lower() in ['stats', 'thongke']:
                stats = recommender.get_statistics()
                print(f"\n📊 Thống kê:")
                print(f"  • Từ vựng: {stats['word_count']} words")
                print(f"  • 4-grams: {stats['fourgram_count']} patterns")
                print(f"  • User preferences: {stats['user_preferences']}")
                print(f"  • Session: {session_stats['selections']}/{session_stats['suggestions']} selections")
                continue
            
            if not user_input:
                continue
            
            # Lấy enhanced recommendations
            start_time = time.time()
            recommendations = recommender.smart_recommend(user_input, context, max_suggestions=8)
            response_time = (time.time() - start_time) * 1000
            
            if recommendations:
                session_stats["suggestions"] += len(recommendations)
                print(f"💡 Gợi ý AI ({response_time:.1f}ms):")
                
                for i, (text, confidence, rec_type) in enumerate(recommendations, 1):
                    confidence_bar = "█" * int(confidence * 10)
                    algo_desc = {
                        "dict_exact": "🎯",
                        "dict_prefix": "🔍", 
                        "dict_fuzzy": "🧩",
                        "advanced_split": "🧠",
                        "pattern_match": "🎨",
                        "context_extend": "📈"
                    }.get(rec_type.split('_')[0] + '_' + rec_type.split('_')[1] if '_' in rec_type else rec_type, "❓")
                    
                    print(f"  {i}. {text} [{confidence_bar:10}] {algo_desc} ({confidence:.3f})")
                
                # Enhanced context prediction nếu có context
                if context:
                    context_preds = recommender.enhanced_context_prediction(context, max_predictions=3)
                    if context_preds:
                        print(f"🔮 Context predictions:")
                        for word, score in context_preds[:3]:
                            print(f"     + {word} ({score:.3f})")
                
                # Cho user chọn
                try:
                    choice = input("\n🔘 Chọn số (hoặc Enter để bỏ qua): ").strip()
                    if choice.isdigit():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(recommendations):
                            chosen_text, confidence, rec_type = recommendations[choice_idx]
                            session_stats["selections"] += 1
                            
                            print(f"✅ Bạn chọn: '{chosen_text}' | {rec_type} | {confidence:.3f}")
                            
                            # Enhanced context và learning
                            context.extend(chosen_text.split())
                            recommender.update_user_preferences(chosen_text, context)
                            
                            # Giới hạn context length
                            if len(context) > 15:
                                context = context[-15:]
                                
                            # Show learning progress
                            stats = recommender.get_statistics()
                            print(f"📈 Learned: {stats['user_preferences']} preferences")
                except ValueError:
                    pass
            else:
                print("❌ Không tìm thấy gợi ý phù hợp")
                print("💡 Thử: 'toihoc', 'xinchao', 'chucmung'")
                
        except KeyboardInterrupt:
            print("\n\nTạm biệt! 👋")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    # Final session stats
    accuracy = (session_stats["selections"] / max(session_stats["suggestions"], 1)) * 100
    print(f"\n📊 Session Summary:")
    print(f"  • Suggestions shown: {session_stats['suggestions']}")
    print(f"  • Selections made: {session_stats['selections']}")
    print(f"  • Selection rate: {accuracy:.1f}%")


def run_ui():
    """
    Chạy giao diện đồ họa enhanced
    """
    try:
        from ui import AdvancedKeyboardUI
        print("🚀 Khởi động Enhanced UI với Advanced AI...")
        app = AdvancedKeyboardUI()
        app.run()
    except ImportError as e:
        print(f"❌ Lỗi import UI: {e}")
        print("Hãy chắc chắn rằng tkinter đã được cài đặt")
        
        # Fallback to backward compatibility
        try:
            from ui import KeyboardUI
            print("🔄 Fallback to basic UI...")
            app = KeyboardUI()
            app.run()
        except ImportError:
            print("❌ Không thể load UI. Chạy demo console thay thế.")
            interactive_demo_enhanced()
    except Exception as e:
        print(f"❌ Lỗi UI: {e}")


def main():
    """
    Main function
    """
    print("🚀 BÀN PHÍM RECOMMEND TIẾNG VIỆT")
    print("Version: 2.0.0 - Phase 3 Advanced Features")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_phase_1_2()
        elif sys.argv[1] == "test3":
            test_phase_3()
        elif sys.argv[1] == "demo":
            interactive_demo_enhanced()
        elif sys.argv[1] == "ui":
            run_ui()
        else:
            print("Usage: python main.py [test|test3|demo|ui]")
            print("  test  - Chạy tests Phase 1-2")
            print("  test3 - Chạy tests Phase 3 Advanced")
            print("  demo  - Enhanced demo tương tác")
            print("  ui    - Chạy Enhanced UI")
    else:
        # Enhanced menu
        print("\n🎯 Chọn chế độ:")
        print("1. Test Phase 1-2 (Core)")
        print("2. Test Phase 3 (Advanced)")
        print("3. Enhanced Demo")
        print("4. Enhanced UI (khuyến nghị)")
        
        try:
            choice = input("\nChọn (1/2/3/4): ").strip()
            if choice == "1":
                test_phase_1_2()
            elif choice == "2":
                test_phase_3()
            elif choice == "3":
                interactive_demo_enhanced()
            elif choice == "4" or choice == "":
                run_ui()
            else:
                print("Lựa chọn không hợp lệ. Chạy Enhanced UI...")
                run_ui()
        except KeyboardInterrupt:
            print("\nTạm biệt! 👋")


if __name__ == "__main__":
    main() 