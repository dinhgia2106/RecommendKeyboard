"""
Script xử lý toàn bộ corpus-full.txt với monitoring tiến độ
"""

import os
import time
from datetime import datetime, timedelta
from data_preparation import VietnameseDataPreprocessor

def process_full_corpus_with_monitoring():
    """Xử lý toàn bộ corpus-full.txt với monitoring"""
    preprocessor = VietnameseDataPreprocessor()
    
    corpus_file = 'data/corpus-full.txt'
    processed_file = 'data/corpus_full_processed.txt'
    
    print(f"Bắt đầu xử lý toàn bộ corpus-full.txt lúc: {datetime.now()}")
    print("Ước tính thời gian: 3-7 giờ tùy cấu hình máy")
    
    start_time = time.time()
    
    # Tạo wrapper để monitor tiến độ
    class MonitoringPreprocessor(VietnameseDataPreprocessor):
        def process_large_corpus_streaming(self, file_path, output_file, 
                                         max_samples=None, chunk_size=10000):
            print(f"Bắt đầu xử lý file lớn: {file_path}")
            processed_count = 0
            start_time = time.time()
            
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for chunk_num, chunk_lines in enumerate(self.load_corpus_streaming(file_path, chunk_size)):
                    pairs = self.create_training_pairs(chunk_lines)
                    
                    # Ghi pairs vào file
                    for x_raw, y_gold in pairs:
                        out_f.write(f"{x_raw}\t{y_gold}\n")
                        processed_count += 1
                        
                        # Dừng nếu đạt giới hạn
                        if max_samples and processed_count >= max_samples:
                            print(f"Đã xử lý {processed_count} samples, dừng theo giới hạn.")
                            return processed_count
                    
                    # Progress report mỗi 100 chunk
                    if chunk_num % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = "N/A"
                        if rate > 0 and max_samples:
                            remaining = max_samples - processed_count
                            eta_seconds = remaining / rate
                            eta = str(timedelta(seconds=int(eta_seconds)))
                        
                        print(f"Chunk {chunk_num}: {processed_count:,} samples | "
                              f"Rate: {rate:.0f} samples/s | ETA: {eta}")
            
            total_time = time.time() - start_time
            print(f"\nHoàn thành! Tổng cộng {processed_count:,} samples trong {timedelta(seconds=int(total_time))}")
            return processed_count
    
    monitor_preprocessor = MonitoringPreprocessor()
    
    # Xử lý toàn bộ file (có thể set max_samples=None để xử lý tất cả)
    # Hoặc set max_samples=5000000 để xử lý 5M samples trước
    max_samples = 5000000  # 5 triệu samples
    
    processed_count = monitor_preprocessor.process_large_corpus_streaming(
        corpus_file, processed_file, max_samples=max_samples, chunk_size=10000
    )
    
    print(f"Đã xử lý {processed_count:,} samples từ corpus-full.txt")
    
    # Chia dataset thành train/dev/test
    print("Chia dataset thành train/dev/test...")
    train_file, dev_file, test_file = monitor_preprocessor.split_large_dataset_file(processed_file)
    
    total_time = time.time() - start_time
    print(f"\nHOÀN THÀNH TOÀN BỘ QUÁ TRÌNH!")
    print(f"Thời gian xử lý: {timedelta(seconds=int(total_time))}")
    print(f"Dataset đã được chia:")
    print(f"- Train: {train_file}")
    print(f"- Dev: {dev_file}")
    print(f"- Test: {test_file}")
    
    return train_file, dev_file, test_file

if __name__ == "__main__":
    process_full_corpus_with_monitoring() 