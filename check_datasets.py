#!/usr/bin/env python3
"""
Script kiểm tra trạng thái và thống kê các dataset có sẵn
"""

import os
from datetime import datetime

def format_size(size_bytes):
    """Format file size"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names)-1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def count_lines(file_path):
    """Đếm số dòng trong file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0

def check_dataset_file(file_path, description):
    """Kiểm tra một file dataset"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        lines = count_lines(file_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        print(f"✅ {description}")
        print(f"   📁 {file_path}")
        print(f"   📊 Size: {format_size(size)}")
        print(f"   📄 Lines: {lines:,}")
        print(f"   🕐 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show sample if not too large
        if size < 100 * 1024 * 1024:  # < 100MB
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = f.readline().strip()
                    if '\t' in sample:
                        x_raw, y_gold = sample.split('\t', 1)
                        print(f"   📝 Sample: '{x_raw[:50]}...' -> '{y_gold[:50]}...'")
            except:
                pass
        print()
        
        return True, lines, size
    else:
        print(f"❌ {description}")
        print(f"   📁 {file_path} (không tồn tại)")
        print()
        return False, 0, 0

def main():
    print("🔍 KIỂM TRA TRẠNG THÁI DATASETS")
    print("=" * 60)
    
    datasets = [
        # Original datasets
        ("data/Viet74K_clean.txt", "Viet74K Clean Corpus"),
        ("data/train.txt", "Small Train Set"),
        ("data/dev.txt", "Small Dev Set"), 
        ("data/test.txt", "Small Test Set"),
        
        # Large corpus files
        ("data/corpus-full.txt", "Original Large Corpus"),
        ("data/corpus_full_processed.txt", "Processed Large Corpus"),
        ("data/corpus_full_processed_train.txt", "Large Train Set"),
        ("data/corpus_full_processed_dev.txt", "Large Dev Set"),
        ("data/corpus_full_processed_test.txt", "Large Test Set"),
        
        # Sample files
        ("data/corpus_sample.txt", "Sample Corpus"),
        ("data/corpus_sample_processed.txt", "Processed Sample"),
    ]
    
    total_files = 0
    total_lines = 0
    total_size = 0
    
    for file_path, description in datasets:
        exists, lines, size = check_dataset_file(file_path, description)
        if exists:
            total_files += 1
            total_lines += lines
            total_size += size
    
    print("=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    print(f"✅ Files có sẵn: {total_files}/{len(datasets)}")
    print(f"📄 Tổng lines: {total_lines:,}")
    print(f"💾 Tổng size: {format_size(total_size)}")
    
    # Recommendations
    print("\n🎯 KHUYẾN NGHỊ")
    print("=" * 60)
    
    large_train_exists = os.path.exists("data/corpus_full_processed_train.txt")
    large_dev_exists = os.path.exists("data/corpus_full_processed_dev.txt")
    large_test_exists = os.path.exists("data/corpus_full_processed_test.txt")
    
    if large_train_exists and large_dev_exists and large_test_exists:
        print("🚀 SẴN SÀNG TRAINING VỚI LARGE CORPUS!")
        print("\n📋 Các lệnh training khuyến nghị:")
        print("   🧪 Test nhanh (10K samples):")
        print("      python train_large_corpus.py --test-run")
        print()
        print("   📊 Training trung bình (100K samples):")
        print("      python train_large_corpus.py --samples 100000")
        print()
        print("   🚀 Training lớn (500K samples):")
        print("      python train_large_corpus.py --samples 500000")
        print()
        print("   🎯 Kết hợp tất cả datasets (Viet74K + corpus-full.txt):")
        print("      python train_large_corpus.py --combine-all --samples 1000000")
        print()
        print("   🏆 TRAINING TOÀN BỘ (tất cả dữ liệu có sẵn):")
        print("      python train_large_corpus.py --full-training")
        
    elif os.path.exists("data/corpus-full.txt"):
        print("📈 CÓ LARGE CORPUS NHƯNG CHƯA XỬ LÝ")
        print("\n💡 Chạy lệnh sau để xử lý:")
        print("   python src/process_full_corpus_background.py")
        
    else:
        print("📚 CHỈ CÓ SMALL CORPUS")
        print("\n💡 Có thể training với:")
        print("   python src/training.py")
    
    # Model status
    print("\n🤖 TRẠNG THÁI MODELS")
    print("=" * 60)
    
    model_dirs = [
        ("models/crf", "CRF Model (Small Corpus)"),
        ("models/crf_large", "CRF Model (Large Corpus)"),
        ("models/crf_test", "CRF Test Model"),
    ]
    
    for model_dir, description in model_dirs:
        model_path = os.path.join(model_dir, "best_model.pkl")
        metadata_path = os.path.join(model_dir, "best_model_metadata.json")
        
        if os.path.exists(model_path):
            print(f"✅ {description}")
            print(f"   📁 {model_path}")
            
            if os.path.exists(metadata_path):
                try:
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    print(f"   📊 F1-score: {metadata.get('test_f1', 'N/A')}")
                    print(f"   📚 Train size: {metadata.get('train_size', 'N/A'):,}")
                    print(f"   🔧 Data source: {metadata.get('data_source', 'N/A')}")
                except:
                    pass
        else:
            print(f"❌ {description} (chưa train)")
        print()

if __name__ == "__main__":
    main() 