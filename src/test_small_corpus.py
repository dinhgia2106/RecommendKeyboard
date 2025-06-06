"""
Test script để xử lý mẫu nhỏ từ corpus-full.txt trước
"""

import os
from data_preparation import VietnameseDataPreprocessor

def create_small_sample():
    """Tạo file mẫu nhỏ từ corpus-full.txt để test"""
    print("Tạo mẫu nhỏ từ corpus-full.txt...")
    
    sample_size = 10000  # 10K dòng đầu tiên
    sample_file = 'data/corpus_sample.txt'
    
    with open('data/corpus-full.txt', 'r', encoding='utf-8') as f_in, \
         open(sample_file, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            if i >= sample_size:
                break
            f_out.write(line)
            
            if i % 1000 == 0:
                print(f"Đã copy {i} dòng...", end='\r')
    
    print(f"\nĐã tạo file mẫu: {sample_file} ({sample_size} dòng)")
    return sample_file

def test_processing():
    """Test xử lý với mẫu nhỏ"""
    preprocessor = VietnameseDataPreprocessor()
    
    # Tạo mẫu nhỏ
    sample_file = create_small_sample()
    
    # Xử lý mẫu
    print("\nBắt đầu xử lý mẫu...")
    processed_file = 'data/corpus_sample_processed.txt'
    
    processed_count = preprocessor.process_large_corpus_streaming(
        sample_file, processed_file, max_samples=None, chunk_size=1000
    )
    
    print(f"Đã xử lý {processed_count} samples từ mẫu")
    
    # Chia thành train/dev/test
    print("Chia dataset mẫu...")
    train_file, dev_file, test_file = preprocessor.split_large_dataset_file(processed_file)
    
    print(f"Dataset mẫu đã chia:")
    print(f"- Train: {train_file}")
    print(f"- Dev: {dev_file}")
    print(f"- Test: {test_file}")
    
    # Hiển thị một vài ví dụ
    print("\nVí dụ từ train file:")
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            if '\t' in line:
                x_raw, y_gold = line.strip().split('\t', 1)
                print(f"{i+1}. X_raw: '{x_raw}' -> Y_gold: '{y_gold}'")
    
    return train_file, dev_file, test_file

if __name__ == "__main__":
    test_processing() 