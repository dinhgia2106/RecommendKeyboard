#!/usr/bin/env python3
"""
Complete Pipeline Runner for Vietnamese Non-accented GPT
Runs the full pipeline: data preprocessing → training → evaluation
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime


def run_command(command, description, check_success=True):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📝 Command: {command}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(command, shell=True, check=check_success,
                                capture_output=False, text=True)

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False


def check_prerequisites():
    """Check if all required files exist"""
    print("🔍 Checking prerequisites...")

    required_files = [
        "data/corpus-full.txt",
        "data/Viet74K.txt"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            # Get file size
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.1f} MB)")

    if missing_files:
        print(f"\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ All prerequisites satisfied")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Complete Vietnamese GPT Pipeline")

    # Pipeline control
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip data preprocessing if already done")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training if model exists")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip final evaluation")

    # Training parameters
    parser.add_argument("--model-size", type=str, default="base",
                        choices=['tiny', 'small', 'base', 'large'],
                        help="Model size to train")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="Corpus sample size for preprocessing")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick pipeline (smaller sample, fewer epochs)")

    # Device settings
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.sample_size = 10000
        args.epochs = 2
        print("🚀 Quick mode enabled (smaller sample, fewer epochs)")

    print("🎯 COMPLETE VIETNAMESE GPT PIPELINE")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏗️ Model size: {args.model_size}")
    print(f"📊 Sample size: {args.sample_size:,}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"💻 Device: {args.device}")
    print("=" * 80)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Create directories
    os.makedirs("ml/data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Track overall time
    pipeline_start = time.time()

    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        preprocessing_cmd = f"""python train_model.py \
            --sample_size {args.sample_size} \
            --min_freq 3 \
            --model_size {args.model_size} \
            --device {args.device} \
            --num_epochs 1"""  # Just for preprocessing

        success = run_command(
            preprocessing_cmd, "Data Preprocessing with Viet74K Integration")
        if not success:
            print("❌ Pipeline failed at preprocessing step")
            sys.exit(1)
    else:
        print("⏭️ Skipping preprocessing (--skip-preprocessing)")

    # Step 2: Model Training
    if not args.skip_training:
        # Check if model already exists
        model_path = "checkpoints/vietnamese_non_accented_gpt_best.pth"
        if os.path.exists(model_path) and not args.quick:
            response = input(
                f"Model already exists at {model_path}. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("⏭️ Skipping training (model exists)")
                args.skip_training = True

        if not args.skip_training:
            training_cmd = f"""python train_model.py \
                --model_size {args.model_size} \
                --num_epochs {args.epochs} \
                --device {args.device} \
                --skip_preprocessing"""

            success = run_command(
                training_cmd, f"Model Training ({args.epochs} epochs)")
            if not success:
                print("❌ Pipeline failed at training step")
                sys.exit(1)
    else:
        print("⏭️ Skipping training (--skip-training)")

    # Step 3: Evaluation
    if not args.skip_evaluation:
        # Check if model exists for evaluation
        model_path = "checkpoints/vietnamese_non_accented_gpt_best.pth"
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}, skipping evaluation")
        else:
            evaluation_cmd = f"""python test_evaluation.py \
                --model_path {model_path} \
                --output checkpoints/pipeline_evaluation_results.json"""

            if args.quick:
                evaluation_cmd += " --quick"

            success = run_command(
                evaluation_cmd, "Comprehensive Model Evaluation", check_success=False)
            if not success:
                print("⚠️ Evaluation failed, but pipeline continues")
    else:
        print("⏭️ Skipping evaluation (--skip-evaluation)")

    # Pipeline completion
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"⏱️ Total time: {pipeline_elapsed/60:.1f} minutes")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary of outputs
    print(f"\n📁 Generated files:")
    output_files = [
        "ml/data/vocab.json",
        "ml/data/training_pairs.csv",
        "ml/data/preprocessing_stats.json",
        "checkpoints/vietnamese_non_accented_gpt_best.pth",
        "checkpoints/enhanced_training_stats.json",
        "checkpoints/pipeline_evaluation_results.json"
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            print(f"   ✅ {file_path} ({size_str})")
        else:
            print(f"   ❌ {file_path} (not found)")

    print(f"\n📊 Quick usage:")
    print(f"   # Test the model:")
    print(f"   python test_evaluation.py --quick")
    print(f"   ")
    print(f"   # Use in application:")
    print(f"   python run_ai_keyboard.py")

    print("\n🚀 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
