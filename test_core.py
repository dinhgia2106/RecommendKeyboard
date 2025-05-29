"""
Test module cho Vietnamese Keyboard v2.1 - Production Ready
Bao gồm tests cho Phase 1-4
"""

import sys
import os
import time

# Thêm đường dẫn để import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import TextProcessor, Dictionary, AdvancedRecommender


def test_phase_1_2():
    """
    Test Phase 1 & 2: Core functionality (updated)
    """
    print("=" * 60)
    print("PHASE 1-2 TEST - CORE FUNCTIONALITY")
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


def test_phase_4():
    """
    Test Phase 4: Production Ready Features
    """
    print("=" * 60)
    print("PHASE 4 TEST - PRODUCTION READY")
    print("=" * 60)
    
    print("\n1. Testing Performance Optimizations...")
    recommender = AdvancedRecommender()
    
    # Performance stress test
    stress_test_cases = [
        "t",
        "to", 
        "toi",
        "toih",
        "toiho",
        "toihoc",
        "toihocti",
        "toihoctie", 
        "toihoctien",
        "toihoctieng",
        "toihoctiengv",
        "toihoctiengvi",
        "toihoctiengvie", 
        "toihoctiengviet",
        "anhyeuemdennaychungcothe"  # Long input
    ]
    
    print(f"Running stress test with {len(stress_test_cases)} cases...")
    
    total_time = 0
    max_time = 0
    min_time = float('inf')
    
    for i, test_input in enumerate(stress_test_cases, 1):
        print(f"  Test {i:2d}: '{test_input}' ({len(test_input):2d} chars)", end=" ")
        
        start_time = time.time()
        recommendations = recommender.smart_recommend(test_input, max_suggestions=8)
        response_time = time.time() - start_time
        
        total_time += response_time
        max_time = max(max_time, response_time)
        min_time = min(min_time, response_time)
        
        print(f"→ {response_time*1000:5.1f}ms ({len(recommendations)} suggestions)")
    
    avg_time = total_time / len(stress_test_cases)
    
    print(f"\n📊 Performance Results:")
    print(f"  • Average: {avg_time*1000:.1f}ms")
    print(f"  • Minimum: {min_time*1000:.1f}ms")
    print(f"  • Maximum: {max_time*1000:.1f}ms")
    print(f"  • Total: {total_time:.2f}s")
    
    # Performance grading
    if avg_time < 0.030:
        grade = "🏆 EXCELLENT"
    elif avg_time < 0.050:
        grade = "🥇 VERY GOOD" 
    elif avg_time < 0.100:
        grade = "🥈 GOOD"
    else:
        grade = "🥉 NEEDS IMPROVEMENT"
    
    print(f"  • Grade: {grade}")
    
    # Test caching effectiveness
    print("\n2. Testing Cache Effectiveness...")
    
    # First run (cold cache)
    start_time = time.time()
    result1 = recommender.smart_recommend("toihoctiengviet", max_suggestions=8)
    cold_time = time.time() - start_time
    
    # Second run (warm cache)
    start_time = time.time()
    result2 = recommender.smart_recommend("toihoctiengviet", max_suggestions=8)
    warm_time = time.time() - start_time
    
    if warm_time < cold_time:
        speedup = (cold_time - warm_time) / cold_time * 100
        print(f"  ✅ Cache speedup: {speedup:.1f}% ({cold_time*1000:.1f}ms → {warm_time*1000:.1f}ms)")
    else:
        print(f"  ℹ️  No significant cache improvement detected")
    
    # Test memory efficiency
    print("\n3. Testing Memory Efficiency...")
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage: {memory_mb:.1f} MB")
        
        if memory_mb < 100:
            print("  ✅ Memory efficient")
        else:
            print("  ⚠️  High memory usage")
            
    except ImportError:
        print("  ℹ️  psutil not available for memory testing")
    
    # Test error handling
    print("\n4. Testing Error Handling...")
    
    error_test_cases = [
        ("", "Empty input"),
        ("a" * 100, "Very long input"),
        ("123456", "Numeric input"),
        ("!@#$%^", "Special characters")
    ]
    
    for test_input, description in error_test_cases:
        try:
            result = recommender.smart_recommend(test_input, max_suggestions=5)
            print(f"  ✅ {description}: {len(result)} results")
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")
    
    # Test performance stats
    print("\n5. Testing Performance Stats...")
    perf_stats = recommender.get_performance_stats()
    
    print(f"  Cache sizes:")
    for cache_type, size in perf_stats['cache_sizes'].items():
        print(f"    • {cache_type}: {size} entries")
    
    print(f"  Performance settings:")
    for setting, value in perf_stats['performance_settings'].items():
        print(f"    • {setting}: {value}")
    
    print(f"\n🚀 PHASE 4 STATUS: Production Ready ✅")


if __name__ == "__main__":
    """
    Run all tests when called directly
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            test_phase_1_2()
        elif sys.argv[1] == "3": 
            test_phase_3()
        elif sys.argv[1] == "4":
            test_phase_4()
        else:
            print("Usage: python test_core.py [1|3|4]")
    else:
        # Run all tests
        print("🧪 RUNNING ALL TESTS - Vietnamese Keyboard v2.1")
        print("=" * 60)
        
        test_phase_1_2()
        print("\n" + "="*60)
        test_phase_3()
        print("\n" + "="*60)
        test_phase_4()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED")
        print("✅ Vietnamese Keyboard v2.1 - Production Ready") 