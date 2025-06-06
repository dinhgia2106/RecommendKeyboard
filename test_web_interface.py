#!/usr/bin/env python3
"""
Test script để kiểm tra web interface với real-time auto-update
"""

import requests
import json
import time


def test_api_endpoint():
    """Test API endpoint với multiple suggestions"""
    
    print("🌐 TEST WEB API với MULTIPLE SUGGESTIONS")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:7862"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API health check passed")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure web server is running on port 7862")
        return
    
    # Test cases
    test_cases = [
        "xinchao",
        "toilasinhhvien",
        "demanoicacbaclanhdaocuckigioiphaicuckigioimoidandatmayduanguthenayphattrientheduoc",
        "moibandenquannuocvietnam"
    ]
    
    print(f"\n📍 Testing {len(test_cases)} cases với multiple suggestions:")
    print("-" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Test với multiple suggestions
        payload = {
            "text": text,
            "multiple_suggestions": True,
            "n_suggestions": 5
        }
        
        try:
            response = requests.post(f"{base_url}/segment", 
                                   json=payload,
                                   headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: '{result['segmented_text']}'")
                print(f"   ⏱️  Time: {result['processing_time']:.4f}s")
                
                if result.get('candidates'):
                    print(f"   📍 Multiple suggestions:")
                    for j, candidate in enumerate(result['candidates'][:3], 1):
                        print(f"      {j}. '{candidate['text']}' ({candidate['confidence']:.3f})")
                else:
                    print("   ⚠️  No multiple suggestions returned")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        time.sleep(0.1)  # Small delay
    
    print(f"\n📊 SUMMARY:")
    print(f"   - API Endpoint: {base_url}")
    print(f"   - Multiple suggestions: Enabled")
    print(f"   - Real-time capability: Available via API")
    print(f"   - Web interface: http://127.0.0.1:7862")


def check_gradio_interface():
    """Check if Gradio interface is accessible"""
    
    print(f"\n🎨 GRADIO INTERFACE CHECK:")
    print("-" * 40)
    
    try:
        response = requests.get("http://127.0.0.1:7862/")
        if response.status_code == 200:
            print("✅ Gradio interface is accessible")
            print("💡 Features:")
            print("   - Real-time auto-update on text change")
            print("   - Multiple suggestions enabled by default")
            print("   - 8 suggestions per input")
            print("   - Processing time display")
            print("   - Confidence scores")
            
            print(f"\n🚀 INSTRUCTIONS:")
            print(f"   1. Mở browser: http://127.0.0.1:7862")
            print(f"   2. Nhập text vào ô 'Input Text'")
            print(f"   3. Kết quả sẽ hiện ngay khi bạn nhập (auto-update)")
            print(f"   4. Multiple suggestions hiện ở phần dưới")
            print(f"   5. Checkbox 'Show multiple suggestions' để bật/tắt")
        else:
            print("❌ Cannot access Gradio interface")
    except Exception as e:
        print(f"❌ Gradio interface error: {e}")


if __name__ == "__main__":
    print("🇻🇳 VIETNAMESE WORD SEGMENTATION WEB INTERFACE TEST")
    print("=" * 70)
    
    # Test API
    test_api_endpoint()
    
    # Check Gradio
    check_gradio_interface()
    
    print(f"\n🎯 REAL-TIME TESTING:")
    print(f"   Nếu auto-update chưa hoạt động, thử:")
    print(f"   - Nhập text và bấm Enter")
    print(f"   - Click ra ngoài textbox (blur event)")
    print(f"   - Toggle checkbox multiple suggestions")
    print(f"   - Reload trang web") 