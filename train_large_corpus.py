#!/usr/bin/env python3
"""
Enhanced Training Script cho Vietnamese Word Segmentation
Hỗ trợ:
- Chunked corpus processing cho datasets lớn
- Structure-aware training để học cấu trúc tiếng Việt
- Context-aware suggestions với meaningfulness scoring
- Punctuation và capitalization preservation
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.training import CRFModelTrainer, StructureAwareTrainer
from src.data_preparation import VietnameseDataPreprocessor

# Force flush output for real-time logging
def log_print(msg):
    print(msg)
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Vietnamese Word Segmentation Training')
    
    # Data options
    parser.add_argument('--samples', type=int, default=100000, 
                       help='Số lượng training samples (default: 100,000)')
    parser.add_argument('--model-dir', type=str, default='models/crf_enhanced',
                       help='Thư mục lưu model')
    parser.add_argument('--corpus-path', type=str, default='data/corpus-full.txt',
                       help='Path to large corpus file')
    
    # Processing options
    parser.add_argument('--use-chunks', action='store_true',
                       help='Chia corpus thành chunks nhỏ trước khi training')
    parser.add_argument('--num-chunks', type=int, default=100,
                       help='Số lượng chunks để chia (default: 100)')
    parser.add_argument('--chunks-dir', type=str, default='data/processed_chunks',
                       help='Thư mục cho processed chunks')
    
    # Training options
    parser.add_argument('--structure-aware', action='store_true', default=True,
                       help='Sử dụng structure-aware training')
    parser.add_argument('--use-dict', action='store_true', default=True,
                       help='Sử dụng dictionary features')
    parser.add_argument('--balance-data', action='store_true', default=True,
                       help='Balance training data để giảm bias từ phổ biến')
    
    # Quick test options
    parser.add_argument('--test-run', action='store_true',
                       help='Chạy test với 10K samples')
    parser.add_argument('--full-training', action='store_true',
                       help='Training với toàn bộ dữ liệu có sẵn')
    
    args = parser.parse_args()
    
    # Test run configuration
    if args.test_run:
        args.samples = 10000
        args.model_dir = 'models/crf_test'
        args.num_chunks = 10
        print("🧪 TEST RUN: Sử dụng 10K samples để test nhanh")
    
    # Full training configuration
    if args.full_training:
        args.samples = None  # No limit
        args.model_dir = 'models/crf_full_enhanced'
        args.use_chunks = True
        print("🏆 FULL TRAINING: Sử dụng TOÀN BỘ dữ liệu")
    
    log_print("🇻🇳 ENHANCED VIETNAMESE WORD SEGMENTATION TRAINING")
    log_print("=" * 70)
    log_print(f"📊 Training samples: {args.samples if args.samples else 'TẤT CẢ'}")
    log_print(f"💾 Model directory: {args.model_dir}")
    log_print(f"🧠 Structure-aware: {args.structure_aware}")
    log_print(f"🔧 Dictionary features: {args.use_dict}")
    log_print(f"⚖️ Balance data: {args.balance_data}")
    log_print(f"📁 Use chunks: {args.use_chunks}")
    if args.use_chunks:
        log_print(f"📄 Number of chunks: {args.num_chunks}")
    log_print(f"🕐 Bắt đầu: {datetime.now().strftime('%H:%M:%S')}")
    log_print("=" * 70)
    
    try:
        # Step 1: Process corpus if using chunks
        if args.use_chunks:
            log_print("\n📋 BƯỚC 1: XỬ LÝ CORPUS THÀNH CHUNKS")
            log_print("-" * 50)
            
            preprocessor = VietnameseDataPreprocessor()
            
            if os.path.exists(args.corpus_path):
                log_print(f"📂 Processing corpus: {args.corpus_path}")
                
                # Smart corpus processing
                result_path = preprocessor.smart_corpus_processing(
                    large_corpus_file=args.corpus_path,
                    output_base_dir=args.chunks_dir,
                    num_chunks=args.num_chunks,
                    max_samples_per_chunk=args.samples // args.num_chunks if args.samples else 50000,
                    final_output=None  # Keep chunks separate
                )
                
                log_print(f"✅ Chunks processed and saved to: {result_path}")
            else:
                log_print(f"⚠️ Corpus file not found: {args.corpus_path}")
                log_print("Using existing processed chunks if available...")
        
        # Step 2: Initialize trainer
        log_print(f"\n🏋️ BƯỚC 2: KHỞI TẠO TRAINER")
        log_print("-" * 50)
        
        if args.structure_aware:
            log_print("🧠 Using Structure-Aware Trainer")
            trainer = StructureAwareTrainer()
        else:
            log_print("📚 Using Standard CRF Trainer")
            trainer = CRFModelTrainer()
        
        # Step 3: Run training
        log_print(f"\n🚀 BƯỚC 3: TRAINING MODEL")
        log_print("-" * 50)
        log_print(f"⏰ Bắt đầu training: {datetime.now().strftime('%H:%M:%S')}")
        
        start_time = datetime.now()
        
        if args.structure_aware:
            # Structure-aware training
            model, test_f1 = trainer.run_structure_aware_training(
                corpus_path=args.corpus_path if not args.use_chunks else None,
                use_chunked_corpus=args.use_chunks,
                chunks_dir=os.path.join(args.chunks_dir, "processed_chunks") if args.use_chunks else None,
                train_size=args.samples,
                model_output_dir=args.model_dir
            )
        else:
            # Standard training
            config = {
                'corpus_path': args.corpus_path if not args.use_chunks else None,
                'train_size': args.samples,
                'use_dictionary': args.use_dict,
                'use_large_corpus': not args.use_chunks,
                'model_output_dir': args.model_dir
            }
            model, test_f1 = trainer.run_training_pipeline(**config)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # Step 4: Results and demo
        print("\n" + "=" * 70)
        print("🎊 KẾT QUẢ TRAINING")
        print("=" * 70)
        print(f"✅ F1-score: {test_f1:.4f}")
        print(f"⏱️  Thời gian training: {training_time}")
        print(f"📚 Samples đã train: {args.samples:,}" if args.samples else "📚 Samples: TẤT CẢ dữ liệu")
        print(f"💾 Model location: {args.model_dir}/best_model.pkl")
        print(f"🧠 Structure-aware: {args.structure_aware}")
        
        # Performance analysis
        if test_f1 >= 0.9:
            print("🏆 Performance xuất sắc!")
            print("🚀 Sẵn sàng deploy!")
        elif test_f1 >= 0.8:
            print("👍 Performance tốt!")
            print("📈 Có thể tăng samples hoặc tune hyperparameters")
        elif test_f1 >= 0.7:
            print("📊 Performance khá ổn")
            print("💡 Đề xuất: Sử dụng structure-aware training")
        else:
            print("⚠️  Performance cần cải thiện")
            print("🔍 Kiểm tra chất lượng dữ liệu và parameters")
        
        # Demo với context-aware suggestions
        print(f"\n🎯 DEMO CONTEXT-AWARE SUGGESTIONS")
        print("-" * 50)
        
        try:
            if args.structure_aware:
                # Demo with multiple suggestions
                test_cases = [
                    "sonha",
                    "xinchaocanha",
                    "toilasinhhvien", 
                    "homnaytroisangdep"
                ]
                
                for test_text in test_cases:
                    print(f"\n📝 Input: '{test_text}'")
                    
                    if hasattr(model, 'segment_with_context'):
                        suggestions = model.segment_with_context(test_text, n_best=3)
                        for i, (suggestion, score) in enumerate(suggestions, 1):
                            print(f"  {i}. '{suggestion}' (score: {score:.3f})")
                    else:
                        result = model.segment(test_text)
                        print(f"  → '{result}'")
        except Exception as e:
            print(f"⚠️ Demo failed: {e}")
        
        print(f"\n🎯 LỆNH SỬ DỤNG:")
        print(f"   📊 Test model: python test_inference.py --model {args.model_dir}/best_model.pkl")
        print(f"   🚀 Chạy demo: python demo.py")
        print(f"   📈 Evaluation: python -m src.evaluation")
        
        if args.structure_aware:
            print(f"   🧠 Context-aware demo: python test_multiple_suggestions.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training bị dừng bởi user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 