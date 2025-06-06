#!/usr/bin/env python3
"""
Script test cho tính năng multiple suggestions
Test các trường hợp khác nhau để xem khả năng đưa ra nhiều gợi ý của model
"""

import os
import sys
sys.path.append('.')

from src.inference import CRFInference


def test_multiple_suggestions():
    """Test tính năng multiple suggestions với các ví dụ cụ thể"""
    
    print("🇻🇳 TEST TÍNH NĂNG MULTIPLE SUGGESTIONS")
    print("=" * 60)
    
    # Kiểm tra model có tồn tại không
    model_path = "models/crf/best_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model không tìm thấy tại {model_path}")
        print("   Vui lòng chạy training trước: python -m src.training")
        return
    
    # Khởi tạo inference
    try:
        inference = CRFInference(model_path)
        print("✅ Model đã được load thành công!")
        
        # Test cases từ user
        test_cases = [
            "demanoicacbaclanhdaocuckigioiphaicuckigioimoidandatmayduanguthenayphattrientheduoc",
            "xinchao",
            "toilasinhhvien",
            "moibandenquannuocvietnam",
            "chungtoicunglamviec",
            "homnaylaicuoituan",
            "toisethamsinhvienvietnam"
        ]
        
        print(f"\n📍 Test với {len(test_cases)} trường hợp:")
        print("-" * 50)
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\n{i}. Test case: '{test_text}'")
            print("   " + "=" * 50)
            
            # Lấy multiple suggestions
            result = inference.segment_multiple(test_text, n_suggestions=5)
            
            print(f"   Input: {result.input_text}")
            print(f"   Suggestions:")
            
            for j, (candidate, confidence) in enumerate(result.candidates, 1):
                mark = "👑" if j == 1 else "  "
                print(f"     {mark} {j}. '{candidate}' (confidence: {confidence:.3f})")
            
            print(f"   Processing time: {result.processing_time:.4f}s")
            
            # So sánh với single result
            single_result = inference.segment(test_text)
            print(f"   Single result: '{single_result.segmented_text}'")
            
            if result.segmented_text != single_result.segmented_text:
                print("   ⚠️  Multiple result khác với single result!")
        
        # Test case cụ thể từ user
        print(f"\n🎯 TEST CASE CỤ THỂ TỪ USER:")
        print("-" * 50)
        
        user_text = "demanoicacbaclanhdaocuckigioiphaicuckigioimoidandatmayduanguthenayphattrientheduoc"
        expected = "de ma noi cac bac lanh dao cuc ki gioi phai cuc ki gioi moi dan dat may dua ngu the nay phat trien the duoc"
        
        result = inference.segment_multiple(user_text, n_suggestions=10)
        
        print(f"Input: {user_text}")
        print(f"Expected: {expected}")
        print(f"Suggestions:")
        
        found_expected = False
        for j, (candidate, confidence) in enumerate(result.candidates, 1):
            mark = "🎯" if candidate == expected else "👑" if j == 1 else "  "
            if candidate == expected:
                found_expected = True
            print(f"  {mark} {j}. '{candidate}' (confidence: {confidence:.3f})")
        
        if found_expected:
            print("✅ Tìm thấy kết quả mong muốn trong suggestions!")
        else:
            print("❌ Không tìm thấy kết quả mong muốn trong suggestions")
            print(f"    Có thể cần tăng số lượng suggestions hoặc cải thiện model")
        
        print(f"\n📊 THỐNG KÊ:")
        print(f"   - Tổng test cases: {len(test_cases) + 1}")
        print(f"   - Model type: CRF với multiple suggestions")
        print(f"   - Số suggestions tối đa: 10")
        
    except Exception as e:
        print(f"❌ Lỗi khi test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multiple_suggestions() 