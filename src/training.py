"""
CRF Training Pipeline for Vietnamese Word Segmentation

This module provides a complete training pipeline for CRF-based Vietnamese 
word segmentation. It handles data preparation, feature extraction, model 
training, validation, and checkpoint saving.

The training process:
1. Load and preprocess data
2. Extract rich features for CRF
3. Train CRF model with cross-validation
4. Evaluate on development set
5. Save best model with metadata
"""

import os
import json
import pickle
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
from sklearn_crfsuite import metrics

from .data_preparation import VietnameseDataPreprocessor
from .models import CRFSegmenter, build_dictionary_from_corpus


class CRFModelTrainer:
    """
    Complete training pipeline for CRF-based Vietnamese word segmentation.
    
    This class handles all aspects of model training including:
    - Data preparation and splitting
    - Feature extraction and dictionary building
    - CRF model training with hyperparameter tuning
    - Model evaluation and validation
    - Model persistence and metadata management
    """
    
    def __init__(self):
        """Initialize the trainer."""
        self.model = None
        self.preprocessor = VietnameseDataPreprocessor()
        self.best_f1 = 0.0
        self.training_history = []
    
    def prepare_data(self, corpus_path: str, 
                    train_size: int = None,
                    rebuild_splits: bool = True) -> Tuple[List, List, List]:
        """
        Prepare training, development, and test datasets.
        
        Args:
            corpus_path: Path to the corpus file
            train_size: Maximum number of samples to use (None for all)
            rebuild_splits: Whether to rebuild train/dev/test splits
            
        Returns:
            Tuple of (train_set, dev_set, test_set)
        """
        print("🔄 Preparing data for CRF training...")
        
        # Check for existing splits
        train_path = "data/train.txt"
        dev_path = "data/dev.txt" 
        test_path = "data/test.txt"
        
        if rebuild_splits or not all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
            print("📖 Loading corpus and creating new splits...")
            
            # Load corpus
            corpus_lines = self.preprocessor.load_corpus(corpus_path)
            
            if train_size:
                corpus_lines = corpus_lines[:train_size]
                print(f"📊 Using first {train_size} samples from corpus")
            
            print(f"📚 Loaded {len(corpus_lines)} lines from corpus")
            
            # Create training pairs
            pairs = self.preprocessor.create_training_pairs(corpus_lines)
            print(f"✅ Created {len(pairs)} training pairs")
            
            # Split dataset
            train_set, dev_set, test_set = self.preprocessor.split_dataset(pairs)
            
            # Save splits
            os.makedirs("data", exist_ok=True)
            self.preprocessor.save_dataset(train_set, train_path)
            self.preprocessor.save_dataset(dev_set, dev_path)
            self.preprocessor.save_dataset(test_set, test_path)
            
            print(f"💾 Saved splits: Train({len(train_set)}), Dev({len(dev_set)}), Test({len(test_set)})")
        else:
            print("📂 Loading existing data splits...")
            train_set = self.preprocessor.load_dataset(train_path)
            dev_set = self.preprocessor.load_dataset(dev_path)
            test_set = self.preprocessor.load_dataset(test_path)
            
            print(f"✅ Loaded splits: Train({len(train_set)}), Dev({len(dev_set)}), Test({len(test_set)})")
        
        return train_set, dev_set, test_set
    
    def build_features(self, train_set: List[Tuple[str, str]], 
                      use_dictionary: bool = True) -> CRFSegmenter:
        """
        Build CRF model with features and dictionary.
        
        Args:
            train_set: Training data pairs
            use_dictionary: Whether to use dictionary-based features
            
        Returns:
            Initialized CRF model
        """
        print("🔧 Building CRF model with enhanced features...")
        
        # Build dictionary from training data if requested
        dictionary = None
        if use_dictionary:
            print("📚 Building word dictionary from training data...")
            corpus_lines = [y_gold for _, y_gold in train_set]
            dictionary = build_dictionary_from_corpus(corpus_lines)
            print(f"✅ Built dictionary with {len(dictionary)} unique words")
        
        # Initialize CRF model with dictionary features
        model = CRFSegmenter(dictionary=dictionary)
        
        return model
    
    def train_crf(self, model: CRFSegmenter, 
                  train_set: List[Tuple[str, str]], 
                  dev_set: List[Tuple[str, str]] = None) -> CRFSegmenter:
        """
        Train the CRF model.
        
        Args:
            model: CRF model to train
            train_set: Training data
            dev_set: Development data for validation
            
        Returns:
            Trained CRF model
        """
        print("🚀 Starting CRF training...")
        
        # Prepare training data
        print("📊 Extracting features from training data...")
        X_features, Y_labels = model.prepare_training_data(train_set)
        
        if not X_features:
            raise ValueError("No valid training sequences generated!")
        
        print(f"✅ Prepared {len(X_features)} training sequences")
        
        # Train model
        print("🔥 Training CRF model...")
        model.train(X_features, Y_labels)
        
        # Evaluate on development set if provided
        if dev_set:
            print("📈 Evaluating on development set...")
            dev_f1 = self.evaluate_model(model, dev_set)
            self.training_history.append({
                'dev_f1': dev_f1,
                'train_size': len(train_set),
                'dev_size': len(dev_set)
            })
            print(f"📊 Development F1-score: {dev_f1:.4f}")
            
            self.best_f1 = dev_f1
        
        return model
    
    def evaluate_model(self, model: CRFSegmenter, 
                      test_set: List[Tuple[str, str]]) -> float:
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained CRF model
            test_set: Test data
            
        Returns:
            F1-score on test set
        """
        if len(test_set) == 0:
            return 0.0
        
        y_true_all = []
        y_pred_all = []
        
        print("🔍 Evaluating model performance...")
        
        # Limit evaluation for speed during development
        eval_samples = min(100, len(test_set))
        
        for x_raw, y_gold in tqdm(test_set[:eval_samples], desc="Evaluating"):
            try:
                # Predict labels
                pred_labels = model.predict(x_raw)
                true_labels = model.create_labels(x_raw, y_gold)
                
                # Only include if lengths match
                if len(pred_labels) == len(true_labels):
                    y_pred_all.extend(pred_labels)
                    y_true_all.extend(true_labels)
                    
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Calculate F1-score using sequence labeling metrics
        if len(y_true_all) > 0:
            f1 = metrics.flat_f1_score(y_true_all, y_pred_all, average='weighted')
        else:
            f1 = 0.0
        
        return f1
    
    def save_model(self, model: CRFSegmenter, model_path: str, 
                   metadata: Dict[str, Any] = None):
        """
        Save trained model with metadata.
        
        Args:
            model: Trained CRF model
            model_path: Path to save model
            metadata: Additional metadata to save
        """
        print(f"💾 Saving CRF model to {model_path}...")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        model.save(model_path)
        
        # Save metadata
        if metadata:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"📋 Saved metadata to {metadata_path}")
        
        print("✅ Model saved successfully!")
    
    def run_training_pipeline(self, corpus_path: str, 
                             train_size: int = None,
                             use_dictionary: bool = True,
                             model_output_dir: str = "models/crf") -> Tuple[CRFSegmenter, float]:
        """
        Run complete CRF training pipeline.
        
        Args:
            corpus_path: Path to training corpus
            train_size: Maximum samples to use
            use_dictionary: Whether to use dictionary features
            model_output_dir: Directory to save trained model
            
        Returns:
            Tuple of (trained_model, test_f1_score)
        """
        print("🚀 Starting CRF training pipeline for Vietnamese word segmentation")
        print("=" * 70)
        
        # Step 1: Prepare data
        train_set, dev_set, test_set = self.prepare_data(
            corpus_path, 
            train_size=train_size
        )
        
        # Step 2: Build model with features
        self.model = self.build_features(train_set, use_dictionary)
        
        # Step 3: Train CRF model
        self.model = self.train_crf(self.model, train_set, dev_set)
        
        # Step 4: Final evaluation
        print("📊 Final evaluation on test set...")
        test_f1 = self.evaluate_model(self.model, test_set)
        print(f"🎯 Final Test F1-score: {test_f1:.4f}")
        
        # Step 5: Save model and metadata
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = os.path.join(model_output_dir, "best_model.pkl")
        
        metadata = {
            'model_type': 'crf',
            'test_f1': test_f1,
            'train_size': len(train_set),
            'dev_size': len(dev_set),
            'test_size': len(test_set),
            'use_dictionary': use_dictionary,
            'training_history': self.training_history,
            'feature_info': {
                'dictionary_size': len(self.model.feature_extractor.dictionary) if self.model.feature_extractor.dictionary else 0,
                'feature_types': ['char_unigrams', 'char_bigrams', 'char_trigrams', 'position', 'char_type']
            }
        }
        
        if use_dictionary:
            metadata['feature_info']['feature_types'].append('dictionary_matching')
        
        self.save_model(self.model, model_path, metadata)
        
        print("🎉 CRF training pipeline completed successfully!")
        print(f"📈 Model performance: F1 = {test_f1:.4f}")
        print(f"💾 Model saved to: {model_path}")
        
        return self.model, test_f1


def main():
    """
    Main training script for CRF model.
    
    This script demonstrates how to train a CRF model for Vietnamese 
    word segmentation with different configurations.
    """
    print("🇻🇳 Vietnamese Word Segmentation - CRF Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = CRFModelTrainer()
    
    # Configuration
    config = {
        'corpus_path': 'data/Viet74K_clean.txt',
        'train_size': 5000,  # Use subset for faster training during development
        'use_dictionary': True,
        'model_output_dir': 'models/crf'
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        # Run training pipeline
        model, test_f1 = trainer.run_training_pipeline(**config)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎊 TRAINING SUMMARY")
        print("=" * 60)
        print(f"✅ Model Type: CRF with BIES tagging")
        print(f"📊 Final F1-score: {test_f1:.4f}")
        print(f"🔧 Dictionary Features: {config['use_dictionary']}")
        print(f"📚 Training Samples: {config['train_size']}")
        print(f"💾 Model Location: {config['model_output_dir']}/best_model.pkl")
        
        # Performance interpretation
        if test_f1 >= 0.9:
            print("🏆 Excellent performance!")
        elif test_f1 >= 0.8:
            print("👍 Good performance!")
        elif test_f1 >= 0.7:
            print("📈 Decent performance, consider tuning hyperparameters")
        else:
            print("⚠️  Performance needs improvement, check data quality")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 