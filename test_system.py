#!/usr/bin/env python3
"""
Vietnamese AI Keyboard - System Test
Quick comprehensive test for all functionality
"""

from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor


def test_core_functionality():
    """Test core proven functionality"""
    print("🧪 Testing Core Functionality:")

    processor = HybridVietnameseProcessor()

    # Core test cases proven to work
    test_cases = [
        ("toihocbai", "tôi học bài"),
        ("toilasinhvien", "tôi là sinh viên"),
        ("homnaytoilam", "hôm nay tôi làm"),
        ("xemphimhomnay", "xem phim hôm nay"),
        ("dihochomnay", "đi học hôm nay"),
        ("ancomroidi", "ăn cơm rồi đi"),
        ("baitaptoingay", "bài tập tối ngày"),
        ("sinhviennamnhat", "sinh viên năm nhất"),
        ("xinchao", "xin chào"),
        ("camon", "cảm ơn"),
        ("maytinh", "máy tính"),
        ("dienthoai", "điện thoại")
    ]

    passed = 0
    total = len(test_cases)

    for input_text, expected in test_cases:
        results = processor.process_text(input_text, max_suggestions=1)

        if results and results[0]['vietnamese_text'] == expected:
            print(
                f"  ✅ {input_text} → {results[0]['vietnamese_text']} ({results[0]['confidence']}%)")
            passed += 1
        else:
            actual = results[0]['vietnamese_text'] if results else "No result"
            print(f"  ❌ {input_text} → Expected: {expected}, Got: {actual}")

    accuracy = (passed / total) * 100
    print(f"\n📊 Core Functionality Results:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")

    return accuracy >= 95.0


def test_system_performance():
    """Test system performance metrics"""
    print("\n⚡ Testing System Performance:")

    processor = HybridVietnameseProcessor()

    import time

    # Performance test
    test_inputs = ["toihocbai", "toilasinhvien", "homnaytoilam"] * 100

    start_time = time.time()
    for input_text in test_inputs:
        processor.process_text(input_text, max_suggestions=3)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = (total_time / len(test_inputs)) * 1000  # ms

    print(f"  Total processing time: {total_time:.3f}s")
    print(f"  Average per suggestion: {avg_time:.2f}ms")
    print(f"  Target: <3ms ({'✅ PASS' if avg_time < 3 else '❌ FAIL'})")

    return avg_time < 3.0


def test_system_statistics():
    """Test system statistics and coverage"""
    print("\n📊 Testing System Statistics:")

    processor = HybridVietnameseProcessor()
    stats = processor.get_statistics()

    print(f"  Core Words: {stats['core_count']:,}")
    print(f"  Extended Words: {stats['extended_count']:,}")
    print(f"  Total Coverage: {stats['total_dictionaries']:,}")

    expected_coverage = 44000  # At least 44K words
    actual_coverage = stats['total_dictionaries']

    print(
        f"  Coverage Target: {expected_coverage:,} ({'✅ PASS' if actual_coverage >= expected_coverage else '❌ FAIL'})")

    return actual_coverage >= expected_coverage


def run_comprehensive_test():
    """Run all tests and generate report"""
    print("🚀 Vietnamese AI Keyboard - Comprehensive System Test\n")

    # Run all tests
    core_pass = test_core_functionality()
    performance_pass = test_system_performance()
    statistics_pass = test_system_statistics()

    # Generate final report
    all_tests_pass = core_pass and performance_pass and statistics_pass

    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"  Core Functionality: {'✅ PASS' if core_pass else '❌ FAIL'}")
    print(f"  Performance: {'✅ PASS' if performance_pass else '❌ FAIL'}")
    print(f"  System Statistics: {'✅ PASS' if statistics_pass else '❌ FAIL'}")
    print(
        f"  Overall: {'🎉 ALL TESTS PASS' if all_tests_pass else '❌ SOME TESTS FAILED'}")

    if all_tests_pass:
        print(f"\n✨ System is ready for production use!")
    else:
        print(f"\n⚠️ System needs attention before production use.")

    return all_tests_pass


if __name__ == "__main__":
    run_comprehensive_test()
