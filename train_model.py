"""
Vietnamese Non-accented Keyboard Training Script
Complete pipeline from data preprocessing to model training
"""

from ml.training.trainer import create_trainer
from ml.training.dataset import create_data_loaders
from ml.tokenizer import VietnameseNonAccentedTokenizer
from ml.data_preprocessor import VietnameseNonAccentedPreprocessor
import os
import sys
import argparse
import torch

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Vietnamese GPT Model")

    # Data arguments
    parser.add_argument("--corpus_path", type=str, default="New_version/data/corpus-full.txt",
                        help="Path to corpus file")
    parser.add_argument("--data_dir", type=str, default="ml/data",
                        help="Directory to save processed data")
    parser.add_argument("--sample_size", type=int, default=100000,
                        help="Number of lines to process from corpus")
    parser.add_argument("--min_freq", type=int, default=5,
                        help="Minimum word frequency for vocabulary")

    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=50000,
                        help="Vocabulary size")
    parser.add_argument("--block_size", type=int, default=32,
                        help="Max sequence length")
    parser.add_argument("--n_layer", type=int, default=8,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Maximum training steps")

    # Other arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip data preprocessing step")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_corpus_samples", type=int, default=50000,
                        help="Number of corpus samples for context training")

    return parser.parse_args()


def preprocess_data(args):
    """Preprocess corpus data"""
    print("="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)

    # Check if data already exists
    data_files = [
        os.path.join(args.data_dir, "vocab.json"),
        os.path.join(args.data_dir, "word_to_non_accented.json"),
        os.path.join(args.data_dir, "training_pairs.csv")
    ]

    if all(os.path.exists(f) for f in data_files) and args.skip_preprocessing:
        print("Processed data already exists. Skipping preprocessing...")
        return

    # Create preprocessor
    preprocessor = VietnameseNonAccentedPreprocessor(args.corpus_path)

    # Process corpus
    print(f"Processing {args.sample_size} lines from {args.corpus_path}")
    preprocessor.process_corpus(sample_size=args.sample_size)

    # Build vocabulary
    print(f"Building vocabulary with min frequency {args.min_freq}")
    preprocessor.build_vocabulary(min_freq=args.min_freq)

    # Save processed data
    stats = preprocessor.save_processed_data(args.data_dir)

    print(f"\nData preprocessing completed!")
    print(f"Statistics: {stats}")

    return stats


def create_model_and_trainer(args, vocab_size=None):
    """Create model and trainer"""
    print("="*60)
    print("STEP 2: MODEL SETUP")
    print("="*60)

    # Get vocabulary size from tokenizer if not provided
    if vocab_size is None:
        tokenizer = VietnameseNonAccentedTokenizer(args.data_dir)
        vocab_size = tokenizer.get_vocab_size()

    print(f"Vocabulary size: {vocab_size}")

    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'block_size': args.block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'dropout': args.dropout
    }

    print(f"Model configuration: {model_config}")

    # Create trainer
    trainer = create_trainer(
        vocab_size=vocab_size,
        data_dir=args.data_dir,
        model_config=model_config,
        device=args.device
    )

    # Setup optimizer and scheduler
    trainer.setup_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    trainer.setup_scheduler(
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps
    )

    return trainer


def create_datasets(args, tokenizer):
    """Create training and validation datasets"""
    print("="*60)
    print("STEP 3: DATASET CREATION")
    print("="*60)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        corpus_path=args.corpus_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.block_size,
        num_corpus_samples=args.num_corpus_samples
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def train_model(args, trainer, train_loader, val_loader):
    """Train the model"""
    print("="*60)
    print("STEP 4: TRAINING")
    print("="*60)

    # Resume from checkpoint if provided
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Test cases for evaluation
    test_cases = [
        ("xinchao", "xin chào"),
        ("chao", "chào"),
        ("xin", "xin"),
        ("cam", "cảm"),
        ("on", "ơn"),
        ("ban", "bạn"),
        ("toi", "tôi"),
        ("la", "là"),
        ("hoc", "học"),
        ("sinh", "sinh")
    ]

    # Test initial predictions
    print("\nTesting initial predictions...")
    trainer.test_predictions(test_cases[:3])

    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_every=2,
        eval_every=1
    )

    # Test final predictions
    print("\nTesting final predictions...")
    trainer.test_predictions(test_cases)

    return trainer


def main():
    """Main training pipeline"""
    args = parse_args()

    print("🚀 VIETNAMESE NON-ACCENTED KEYBOARD TRAINING PIPELINE")
    print("=" * 60)
    print(f"Arguments: {vars(args)}")
    print("=" * 60)

    # Check if corpus file exists
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus file not found at {args.corpus_path}")
        sys.exit(1)

    # Create output directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    try:
        # Step 1: Preprocess data
        stats = preprocess_data(args)

        # Step 2: Create model and trainer
        vocab_size = stats.get('vocab_size') if stats else None
        trainer = create_model_and_trainer(args, vocab_size)

        # Step 3: Create datasets
        train_loader, val_loader = create_datasets(args, trainer.tokenizer)

        # Step 4: Train model
        trained_trainer = train_model(args, trainer, train_loader, val_loader)

        print("="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Model checkpoints saved in: checkpoints/")
        print(f"📊 Training curves saved in: checkpoints/training_curves.png")
        print(f"💾 Best model: checkpoints/vietnamese_non_accented_gpt_best.pth")
        print("="*60)

        # Save final statistics
        final_stats = {
            'model_config': vars(args),
            'vocab_size': trainer.tokenizer.get_vocab_size(),
            'training_samples': len(train_loader.dataset),
            'validation_samples': len(val_loader.dataset),
            'final_train_loss': trained_trainer.train_losses[-1] if trained_trainer.train_losses else None,
            'best_val_loss': trained_trainer.best_val_loss,
            'total_epochs': trained_trainer.current_epoch + 1,
            'total_steps': trained_trainer.global_step
        }

        import json
        with open('checkpoints/training_stats.json', 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

        print(f"📈 Final statistics saved to: checkpoints/training_stats.json")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
