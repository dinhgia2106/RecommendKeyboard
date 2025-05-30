#!/usr/bin/env python3
"""
Vietnamese AI Keyboard Launcher
Launch the AI-powered Vietnamese pinyin keyboard with various options
"""

import os
import sys
import argparse
import subprocess


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import transformers
        import pandas
        import numpy
        print("✅ All ML dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def check_trained_model():
    """Check if trained model exists"""
    model_path = "checkpoints/vietnamese_non_accented_gpt_best.pth"
    if os.path.exists(model_path):
        print(f"✅ Trained model found: {model_path}")
        return True
    else:
        print(f"⚠️  No trained model found at: {model_path}")
        print("The system will run with fallback recommendations only.")
        print("To train a model, run: python train_model.py")
        return False


def check_processed_data():
    """Check if processed data exists"""
    data_files = [
        "ml/data/vocab.json",
        "ml/data/word_to_non_accented.json",
        "ml/data/training_pairs.csv"
    ]

    existing_files = [f for f in data_files if os.path.exists(f)]

    if len(existing_files) == len(data_files):
        print(f"✅ Processed data found: {len(existing_files)} files")
        return True
    else:
        print(
            f"⚠️  Processed data incomplete: {len(existing_files)}/{len(data_files)} files")
        print("Missing files:")
        for f in data_files:
            if not os.path.exists(f):
                print(f"  - {f}")
        print("To process data, run: python train_model.py")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese AI Keyboard Launcher")

    parser.add_argument("--ui", choices=["ai", "v4"], default="ai",
                        help="Choose UI version (ai=new AI version, v4=old v4 version)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check dependencies and data, don't launch UI")
    parser.add_argument("--train", action="store_true",
                        help="Run training before launching UI")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run data preprocessing")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after loading")

    args = parser.parse_args()

    print("🚀 Vietnamese AI Keyboard Launcher")
    print("=" * 50)

    # Check dependencies
    deps_ok = check_dependencies()

    # Check processed data
    data_ok = check_processed_data()

    # Check trained model
    model_ok = check_trained_model()

    print("\n📊 System Status:")
    print(f"  Dependencies: {'✅ Ready' if deps_ok else '❌ Missing'}")
    print(f"  Processed Data: {'✅ Ready' if data_ok else '⚠️  Incomplete'}")
    print(f"  Trained Model: {'✅ Ready' if model_ok else '⚠️  Missing'}")

    if args.check_only:
        print("\n✅ Check completed.")
        return

    # If dependencies missing, cannot continue
    if not deps_ok:
        print("\n❌ Cannot continue without dependencies.")
        print("Please install: pip install -r requirements.txt")
        return

    # Handle preprocessing only
    if args.preprocess_only:
        print("\n🔄 Running data preprocessing only...")
        try:
            from ml.data_preprocessor import main as preprocess_main
            preprocess_main()
            print("✅ Data preprocessing completed!")
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
        return

    # Handle training
    if args.train or (not data_ok and not model_ok):
        if not os.path.exists("data/corpus-full.txt"):
            print("\n❌ Cannot train: corpus-full.txt not found")
            print("Please ensure data/corpus-full.txt exists")
            return

        print("\n🔄 Running training pipeline...")
        try:
            # Run training script
            cmd = [sys.executable, "train_model.py", "--sample_size", "100000"]
            result = subprocess.run(cmd, capture_output=False)

            if result.returncode == 0:
                print("✅ Training completed successfully!")
                model_ok = True
                data_ok = True
            else:
                print("❌ Training failed!")
                print("Continuing with fallback mode...")

        except Exception as e:
            print(f"❌ Training error: {e}")
            print("Continuing with fallback mode...")

    # Launch UI
    print(f"\n🚀 Launching {args.ui.upper()} Keyboard UI...")

    try:
        if args.ui == "ai":
            from ui.ai_keyboard_ui import AIKeyboardUI
            app = AIKeyboardUI()

            # Run benchmark if requested
            if args.benchmark and app.recommender:
                print("🔄 Running benchmark...")
                app.recommender.benchmark_performance()

        else:  # v4 UI
            from ui.v4_keyboard_ui import V4KeyboardUI
            app = V4KeyboardUI()

        print("✅ UI loaded successfully!")
        print("\n🎯 Usage Instructions:")
        print("  1. Type non-accented text (e.g., 'xinchao')")
        print("  2. Use arrow keys or mouse to select")
        print("  3. Press Enter or click to choose")
        print("  4. Numbers 1-9 for quick selection")
        print("\n🎉 Happy typing!")

        # Start the UI
        app.run()

    except Exception as e:
        print(f"❌ Failed to launch UI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
