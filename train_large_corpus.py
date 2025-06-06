#!/usr/bin/env python3
"""
Training script cho Vietnamese Word Segmentation với LARGE CORPUS
Tận dụng dataset từ corpus-full.txt đã được xử lý
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.training import CRFModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train CRF model với large corpus từ corpus-full.txt')
    parser.add_argument('--samples', type=int, default=100000, 
                       help='Số lượng training samples (default: 100,000)')
    parser.add_argument('--model-dir', type=str, default='models/crf_large',
                       help='Thư mục lưu model')
    parser.add_argument('--use-dict', action='store_true', default=True,
                       help='Sử dụng dictionary features')
    parser.add_argument('--test-run', action='store_true',
                       help='Chạy test với 10K samples')
    parser.add_argument('--combine-all', action='store_true',
                       help='Kết hợp TẤT CẢ datasets (Viet74K + corpus-full.txt)')
    parser.add_argument('--full-training', action='store_true',
                       help='Training với toàn bộ dữ liệu có sẵn')
    
    args = parser.parse_args()
    
    # Test run with smaller dataset
    if args.test_run:
        args.samples = 10000
        args.model_dir = 'models/crf_test'
        print("🧪 TEST RUN: Sử dụng 10K samples để test nhanh")
    
    # Full training mode
    if args.full_training:
        args.combine_all = True
        args.samples = None  # No limit
        args.model_dir = 'models/crf_full'
        print("🏆 FULL TRAINING: Sử dụng TOÀN BỘ dữ liệu (Viet74K + corpus-full.txt)")
    
    # Combine all datasets mode
    if args.combine_all:
        print("🚀 COMBINE MODE: Kết hợp tất cả datasets có sẵn")
    
    print("🇻🇳 TRAINING CRF MODEL VỚI LARGE CORPUS")
    print("=" * 60)
    print(f"📊 Training samples: {args.samples if args.samples else 'TẤT CẢ'}")
    print(f"💾 Model directory: {args.model_dir}")
    print(f"🔧 Dictionary features: {args.use_dict}")
    print(f"📈 Combine all datasets: {args.combine_all}")
    print(f"🕐 Bắt đầu: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Kiểm tra files dataset có tồn tại không
    required_files = [
        'data/corpus_full_processed_train.txt',
        'data/corpus_full_processed_dev.txt',
        'data/corpus_full_processed_test.txt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Không tìm thấy files dataset:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Chạy lệnh sau để tạo dataset:")
        print("   python src/process_full_corpus_background.py")
        return
    
    # Initialize trainer
    trainer = CRFModelTrainer()
    
    # Configuration
    config = {
        'corpus_path': None,
        'train_size': args.samples,
        'use_dictionary': args.use_dict,
        'use_large_corpus': not args.combine_all,  # Use large corpus mode unless combining all
        'combine_all_datasets': args.combine_all,
        'model_output_dir': args.model_dir
    }
    
    try:
        # Run training
        start_time = datetime.now()
        model, test_f1 = trainer.run_training_pipeline(**config)
        end_time = datetime.now()
        
        training_time = end_time - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("🎊 KẾT QUẢ TRAINING")
        print("=" * 60)
        print(f"✅ F1-score: {test_f1:.4f}")
        print(f"⏱️  Thời gian training: {training_time}")
        print(f"📚 Samples đã train: {args.samples:,}")
        print(f"💾 Model location: {args.model_dir}/best_model.pkl")
        
        # Performance analysis
        if test_f1 >= 0.9:
            print("🏆 Performance xuất sắc!")
            print("🚀 Sẵn sàng deploy!")
        elif test_f1 >= 0.8:
            print("👍 Performance tốt!")
            print("📈 Có thể tăng samples để cải thiện thêm")
        elif test_f1 >= 0.7:
            print("📊 Performance khá ổn")
            print("💡 Đề xuất: Tăng training samples hoặc tune hyperparameters")
        else:
            print("⚠️  Performance cần cải thiện")
            print("🔍 Kiểm tra chất lượng dữ liệu")
        
        print("\n🎯 Lệnh test model:")
        print(f"   python test_inference.py --model {args.model_dir}/best_model.pkl")
        
        print("\n🚀 Lệnh chạy demo:")
        print("   python demo.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training bị dừng bởi user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 