"""
Bước 1: Chuẩn bị Dữ liệu
- Thu thập corpus Vietnamese đã annotate 
- Tiền xử lý: lowercase, loại bỏ dấu, xóa space → tạo X_raw
- Giữ bản chuẩn (có dấu + khoảng trắng) cho nhãn Y_gold
- Tạo bộ train/dev/test
- Hỗ trợ xử lý file lớn với streaming
"""

import re
import unicodedata
import random
from typing import List, Tuple, Dict, Iterator
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class VietnameseDataPreprocessor:
    def __init__(self):
        # Mapping các ký tự có dấu về không dấu
        self.diacritic_map = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd', 'Đ': 'D'
        }
    
    def remove_diacritics(self, text: str) -> str:
        """Loại bỏ dấu khỏi văn bản tiếng Việt"""
        result = ""
        for char in text:
            if char in self.diacritic_map:
                result += self.diacritic_map[char]
            else:
                result += char
        return result
    
    def remove_spaces(self, text: str) -> str:
        """Loại bỏ khoảng trắng"""
        return re.sub(r'\s+', '', text)
    
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa văn bản: lowercase, loại bỏ ký tự đặc biệt"""
        text = text.lower()
        # Giữ lại chữ cái và một số ký tự đặc biệt cần thiết
        text = re.sub(r'[^\w\s\-]', '', text, flags=re.UNICODE)
        return text
    
    def create_raw_input(self, text: str) -> str:
        """Tạo X_raw: lowercase, không dấu, không space"""
        text = self.normalize_text(text)
        text = self.remove_diacritics(text)
        text = self.remove_spaces(text)
        return text
    
    def load_corpus(self, file_path: str) -> List[str]:
        """Đọc corpus từ file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    
    def load_corpus_streaming(self, file_path: str, chunk_size: int = 10000) -> Iterator[List[str]]:
        """
        Đọc corpus lớn theo chunk để tiết kiệm memory
        Trả về iterator của các chunk
        """
        chunk = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Bỏ qua dòng trống
                        chunk.append(line)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                # Yield chunk cuối cùng nếu còn
                if chunk:
                    yield chunk
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return
    
    def create_training_pairs(self, corpus_lines: List[str]) -> List[Tuple[str, str]]:
        """
        Tạo cặp (X_raw, Y_gold) từ corpus
        X_raw: không dấu, không space
        Y_gold: có dấu, có space
        """
        pairs = []
        for line in corpus_lines:
            if len(line.strip()) > 0:
                y_gold = line.strip()  # Giữ nguyên (có dấu, có space)
                x_raw = self.create_raw_input(y_gold)  # Loại dấu, loại space
                if len(x_raw) > 0:
                    pairs.append((x_raw, y_gold))
        return pairs
    
    def process_large_corpus_streaming(self, file_path: str, output_file: str, 
                                     max_samples: int = None, chunk_size: int = 10000):
        """
        Xử lý file corpus lớn theo streaming và ghi trực tiếp ra file
        """
        print(f"Bắt đầu xử lý file lớn: {file_path}")
        processed_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for chunk_lines in self.load_corpus_streaming(file_path, chunk_size):
                pairs = self.create_training_pairs(chunk_lines)
                
                # Ghi pairs vào file
                for x_raw, y_gold in pairs:
                    out_f.write(f"{x_raw}\t{y_gold}\n")
                    processed_count += 1
                    
                    # Dừng nếu đạt giới hạn
                    if max_samples and processed_count >= max_samples:
                        print(f"Đã xử lý {processed_count} samples, dừng theo giới hạn.")
                        return processed_count
                
                print(f"Đã xử lý {processed_count} samples...", end='\r')
        
        print(f"\nHoàn thành! Tổng cộng {processed_count} samples được xử lý.")
        return processed_count
    
    def create_sequence_labels(self, text_with_spaces: str) -> List[str]:
        """
        Tạo nhãn sequence labeling theo schema BIES:
        B - Begin (đầu từ)
        I - Inside (giữa từ) 
        E - End (cuối từ)
        S - Single (từ đơn)
        """
        words = text_with_spaces.split()
        labels = []
        
        for word in words:
            word_clean = self.remove_diacritics(word.lower())
            word_clean = re.sub(r'[^\w]', '', word_clean)
            
            if len(word_clean) == 1:
                labels.append('S')
            elif len(word_clean) > 1:
                labels.append('B')
                for _ in range(len(word_clean) - 2):
                    labels.append('I')
                labels.append('E')
        
        return labels
    
    def create_char_label_pairs(self, x_raw: str, y_gold: str) -> List[Tuple[str, str]]:
        """
        Tạo cặp (ký_tự, nhãn) cho sequence labeling
        """
        # Tạo nhãn từ y_gold
        labels = self.create_sequence_labels(y_gold)
        
        # Tạo danh sách ký tự từ x_raw
        chars = list(x_raw)
        
        # Đảm bảo số ký tự và nhãn khớp nhau
        if len(chars) != len(labels):
            # Cần xử lý trường hợp không khớp
            return []
        
        return list(zip(chars, labels))
    
    def split_large_dataset_file(self, input_file: str, 
                               train_ratio: float = 0.7, 
                               dev_ratio: float = 0.15, 
                               test_ratio: float = 0.15,
                               random_seed: int = 42):
        """
        Chia dataset lớn thành train/dev/test bằng cách đọc streaming
        """
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
        
        # Đầu tiên đếm tổng số dòng
        print("Đang đếm tổng số dòng...")
        total_lines = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        
        print(f"Tổng số dòng: {total_lines}")
        
        # Tính toán số dòng cho mỗi split
        train_lines = int(total_lines * train_ratio)
        dev_lines = int(total_lines * dev_ratio)
        test_lines = total_lines - train_lines - dev_lines
        
        print(f"Train: {train_lines}, Dev: {dev_lines}, Test: {test_lines}")
        
        # Tạo danh sách index và shuffle
        random.seed(random_seed)
        indices = list(range(total_lines))
        random.shuffle(indices)
        
        train_indices = set(indices[:train_lines])
        dev_indices = set(indices[train_lines:train_lines + dev_lines])
        test_indices = set(indices[train_lines + dev_lines:])
        
        # Chia file
        train_file = input_file.replace('.txt', '_train.txt')
        dev_file = input_file.replace('.txt', '_dev.txt')
        test_file = input_file.replace('.txt', '_test.txt')
        
        with open(input_file, 'r', encoding='utf-8') as in_f, \
             open(train_file, 'w', encoding='utf-8') as train_f, \
             open(dev_file, 'w', encoding='utf-8') as dev_f, \
             open(test_file, 'w', encoding='utf-8') as test_f:
            
            for i, line in enumerate(in_f):
                if i in train_indices:
                    train_f.write(line)
                elif i in dev_indices:
                    dev_f.write(line)
                elif i in test_indices:
                    test_f.write(line)
                
                if i % 100000 == 0:
                    print(f"Đã xử lý {i}/{total_lines} dòng...", end='\r')
        
        print(f"\nHoàn thành chia dataset!")
        return train_file, dev_file, test_file
    
    def split_dataset(self, pairs: List[Tuple[str, str]], 
                     train_ratio: float = 0.7, 
                     dev_ratio: float = 0.15, 
                     test_ratio: float = 0.15,
                     random_seed: int = 42) -> Tuple[List, List, List]:
        """Chia dataset thành train/dev/test"""
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
        
        random.seed(random_seed)
        random.shuffle(pairs)
        
        n = len(pairs)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))
        
        train_set = pairs[:train_end]
        dev_set = pairs[train_end:dev_end]
        test_set = pairs[dev_end:]
        
        return train_set, dev_set, test_set
    
    def save_dataset(self, dataset: List[Tuple[str, str]], file_path: str):
        """Lưu dataset ra file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for x_raw, y_gold in dataset:
                f.write(f"{x_raw}\t{y_gold}\n")
    
    def load_dataset(self, file_path: str) -> List[Tuple[str, str]]:
        """Đọc dataset từ file"""
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    x_raw, y_gold = line.strip().split('\t', 1)
                    pairs.append((x_raw, y_gold))
        return pairs
    
    def split_corpus_into_chunks(self, corpus_file: str, output_dir: str, 
                               num_chunks: int = 100, prefix: str = "corpus_chunk"):
        """
        Chia corpus lớn thành nhiều file nhỏ cho parallel training.
        
        Args:
            corpus_file: File corpus lớn
            output_dir: Thư mục output
            num_chunks: Số lượng chunks muốn chia
            prefix: Prefix cho tên file chunks
        """
        print(f"🔪 Chia {corpus_file} thành {num_chunks} chunks...")
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Đếm tổng số dòng
        print("📊 Đang đếm tổng số dòng...")
        total_lines = 0
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        
        print(f"📝 Tổng số dòng: {total_lines:,}")
        
        # Tính số dòng mỗi chunk
        lines_per_chunk = max(1, total_lines // num_chunks)
        print(f"📄 Mỗi chunk: ~{lines_per_chunk:,} dòng")
        
        # Chia file
        current_chunk = 0
        current_lines = 0
        output_files = []
        
        with open(corpus_file, 'r', encoding='utf-8') as input_file:
            output_file = None
            
            for line_num, line in enumerate(input_file):
                # Mở file chunk mới nếu cần
                if current_lines == 0:
                    if output_file:
                        output_file.close()
                    
                    chunk_filename = f"{prefix}_{current_chunk:03d}.txt"
                    chunk_path = os.path.join(output_dir, chunk_filename)
                    output_file = open(chunk_path, 'w', encoding='utf-8')
                    output_files.append(chunk_path)
                    print(f"📁 Tạo chunk {current_chunk + 1}/{num_chunks}: {chunk_filename}")
                
                # Ghi dòng vào chunk hiện tại
                output_file.write(line)
                current_lines += 1
                
                # Chuyển sang chunk mới nếu đủ dòng (trừ chunk cuối)
                if current_lines >= lines_per_chunk and current_chunk < num_chunks - 1:
                    current_lines = 0
                    current_chunk += 1
            
            # Đóng file cuối cùng
            if output_file:
                output_file.close()
        
        print(f"✅ Đã chia thành {len(output_files)} chunks trong {output_dir}")
        return output_files
    
    def process_corpus_chunks_parallel(self, chunks_dir: str, output_dir: str, 
                                     max_samples_per_chunk: int = None):
        """
        Xử lý các chunks song song để tạo training data.
        
        Args:
            chunks_dir: Thư mục chứa các chunks
            output_dir: Thư mục output cho processed data
            max_samples_per_chunk: Giới hạn samples mỗi chunk
        """
        print(f"⚡ Xử lý chunks song song từ {chunks_dir}")
        
        # Tìm tất cả chunk files
        chunk_files = []
        for filename in os.listdir(chunks_dir):
            if filename.endswith('.txt') and 'chunk' in filename:
                chunk_files.append(os.path.join(chunks_dir, filename))
        
        chunk_files.sort()
        print(f"📁 Tìm thấy {len(chunk_files)} chunks")
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Xử lý từng chunk
        processed_files = []
        total_processed = 0
        
        for i, chunk_file in enumerate(chunk_files):
            print(f"\n🔄 Xử lý chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
            
            # Tên file output cho chunk này
            chunk_name = os.path.splitext(os.path.basename(chunk_file))[0]
            output_file = os.path.join(output_dir, f"{chunk_name}_processed.txt")
            
            # Xử lý chunk
            processed_count = self.process_large_corpus_streaming(
                chunk_file, output_file, max_samples_per_chunk
            )
            
            if processed_count > 0:
                processed_files.append(output_file)
                total_processed += processed_count
                print(f"✅ Chunk {i+1} hoàn thành: {processed_count:,} samples")
            else:
                print(f"⚠️ Chunk {i+1} không có data")
        
        print(f"\n🎉 Hoàn thành xử lý {len(processed_files)} chunks")
        print(f"📊 Tổng cộng: {total_processed:,} training samples")
        
        return processed_files
    
    def combine_processed_chunks(self, processed_files: List[str], 
                               output_file: str, shuffle: bool = True):
        """
        Kết hợp các processed chunks thành một file training duy nhất.
        
        Args:
            processed_files: List các file đã processed
            output_file: File output
            shuffle: Có shuffle data không
        """
        print(f"🔄 Kết hợp {len(processed_files)} chunks thành {output_file}")
        
        all_lines = []
        
        # Đọc tất cả lines từ các chunks
        for file_path in processed_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    all_lines.extend(lines)
                    print(f"📄 {os.path.basename(file_path)}: {len(lines):,} samples")
        
        print(f"📝 Tổng cộng: {len(all_lines):,} samples")
        
        # Shuffle nếu cần
        if shuffle:
            print("🎲 Shuffling data...")
            import random
            random.shuffle(all_lines)
        
        # Ghi ra file cuối
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in all_lines:
                f.write(line + '\n')
        
        print(f"✅ Đã kết hợp thành {output_file}")
        return len(all_lines)
    
    def smart_corpus_processing(self, large_corpus_file: str, 
                              output_base_dir: str = "data/processed_chunks",
                              num_chunks: int = 100,
                              max_samples_per_chunk: int = 50000,
                              final_output: str = None):
        """
        Pipeline xử lý corpus lớn một cách thông minh.
        
        Args:
            large_corpus_file: File corpus lớn gốc
            output_base_dir: Thư mục base cho output
            num_chunks: Số chunks muốn chia
            max_samples_per_chunk: Max samples mỗi chunk
            final_output: File output cuối (optional)
        
        Returns:
            Path to final processed file
        """
        print("🚀 BẮT ĐẦU SMART CORPUS PROCESSING")
        print("=" * 60)
        
        # Tạo các thư mục cần thiết
        chunks_dir = os.path.join(output_base_dir, "raw_chunks")
        processed_dir = os.path.join(output_base_dir, "processed_chunks")
        
        # Bước 1: Chia corpus thành chunks
        print("\n📋 BƯỚC 1: Chia corpus thành chunks")
        chunk_files = self.split_corpus_into_chunks(
            large_corpus_file, chunks_dir, num_chunks
        )
        
        # Bước 2: Xử lý các chunks
        print("\n🔧 BƯỚC 2: Xử lý các chunks")
        processed_files = self.process_corpus_chunks_parallel(
            chunks_dir, processed_dir, max_samples_per_chunk
        )
        
        # Bước 3: Kết hợp nếu cần
        if final_output:
            print("\n🔗 BƯỚC 3: Kết hợp chunks")
            total_samples = self.combine_processed_chunks(
                processed_files, final_output
            )
            result_file = final_output
        else:
            # Giữ nguyên các chunks riêng lẻ
            total_samples = sum(
                len(open(f, 'r', encoding='utf-8').readlines()) 
                for f in processed_files if os.path.exists(f)
            )
            result_file = processed_dir
        
        print("\n" + "=" * 60)
        print("🎊 HOÀN THÀNH SMART CORPUS PROCESSING")
        print(f"📊 Tổng samples: {total_samples:,}")
        print(f"📁 Kết quả: {result_file}")
        print("=" * 60)
        
        return result_file

def process_full_corpus():
    """
    Xử lý file corpus-full.txt lớn và tạo dataset training
    """
    preprocessor = VietnameseDataPreprocessor()
    
    # Kiểm tra file corpus-full.txt có tồn tại không
    corpus_file = 'data/corpus-full.txt'
    if not os.path.exists(corpus_file):
        print(f"Không tìm thấy file: {corpus_file}")
        return
    
    # Xử lý file lớn với streaming
    print("Bắt đầu xử lý corpus-full.txt...")
    processed_file = 'data/corpus_processed.txt'
    
    # Giới hạn số samples để test (có thể bỏ giới hạn này để xử lý toàn bộ)
    max_samples = 1000000  # 1 triệu samples để test
    
    processed_count = preprocessor.process_large_corpus_streaming(
        corpus_file, processed_file, max_samples=max_samples, chunk_size=10000
    )
    
    print(f"Đã xử lý {processed_count} samples từ corpus-full.txt")
    
    # Chia dataset thành train/dev/test
    print("Chia dataset thành train/dev/test...")
    train_file, dev_file, test_file = preprocessor.split_large_dataset_file(processed_file)
    
    print(f"Dataset đã được chia:")
    print(f"- Train: {train_file}")
    print(f"- Dev: {dev_file}")  
    print(f"- Test: {test_file}")
    
    return train_file, dev_file, test_file

def main():
    """Demo sử dụng data preprocessor với cả file nhỏ và lớn"""
    preprocessor = VietnameseDataPreprocessor()
    
    print("=== XỬ LÝ FILE NHỎ (Viet74K_clean.txt) ===")
    # Đọc corpus nhỏ
    if os.path.exists('data/Viet74K_clean.txt'):
        print("Đang đọc Viet74K_clean.txt...")
        corpus_lines = preprocessor.load_corpus('data/Viet74K_clean.txt')
        print(f"Đã đọc {len(corpus_lines)} dòng")
        
        # Tạo training pairs
        print("Đang tạo training pairs...")
        pairs = preprocessor.create_training_pairs(corpus_lines)
        print(f"Đã tạo {len(pairs)} pairs")
        
        # Hiển thị một số ví dụ
        print("\nVí dụ training pairs từ Viet74K:")
        for i, (x_raw, y_gold) in enumerate(pairs[:3]):
            print(f"{i+1}. X_raw: '{x_raw}' -> Y_gold: '{y_gold}'")
        
        # Chia dataset
        print("\nChia dataset...")
        train_set, dev_set, test_set = preprocessor.split_dataset(pairs)
        print(f"Train: {len(train_set)}, Dev: {len(dev_set)}, Test: {len(test_set)}")
        
        # Lưu dataset từ Viet74K
        print("Lưu dataset từ Viet74K...")
        preprocessor.save_dataset(train_set, 'data/viet74k_train.txt')
        preprocessor.save_dataset(dev_set, 'data/viet74k_dev.txt')
        preprocessor.save_dataset(test_set, 'data/viet74k_test.txt')
    
    print("\n=== XỬ LÝ FILE LỚN (corpus-full.txt) ===")
    # Xử lý file lớn
    if os.path.exists('data/corpus-full.txt'):
        train_file, dev_file, test_file = process_full_corpus()
        print("Hoàn thành xử lý corpus-full.txt!")
    else:
        print("Không tìm thấy corpus-full.txt")

if __name__ == "__main__":
    main() 