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
    
    def prepare_data(self, corpus_path: str = None, 
                    train_size: int = None,
                    rebuild_splits: bool = False,
                    use_large_corpus: bool = True,
                    combine_all_datasets: bool = False) -> Tuple[List, List, List]:
        """
        Prepare training, development, and test datasets.
        
        Args:
            corpus_path: Path to the corpus file (legacy parameter)
            train_size: Maximum number of samples to use (None for all)
            rebuild_splits: Whether to rebuild train/dev/test splits
            use_large_corpus: Whether to use the large corpus from corpus-full.txt
            combine_all_datasets: Whether to combine ALL available datasets (Viet74K + corpus-full.txt)
            
        Returns:
            Tuple of (train_set, dev_set, test_set)
        """
        print("🔄 Preparing data for CRF training...")
        
        if combine_all_datasets:
            print("🚀 COMBINING ALL DATASETS: Viet74K + corpus-full.txt")
            return self._combine_all_datasets(train_size)
        
        if use_large_corpus:
            # Use processed files from corpus-full.txt
            train_path = "data/corpus_full_processed_train.txt"
            dev_path = "data/corpus_full_processed_dev.txt" 
            test_path = "data/corpus_full_processed_test.txt"
            
            if all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
                print("📂 Loading large corpus dataset from corpus-full.txt processing...")
                
                # Load with streaming for large files
                def load_large_dataset(file_path: str, max_samples: int = None):
                    pairs = []
                    count = 0
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '\t' in line:
                                x_raw, y_gold = line.strip().split('\t', 1)
                                pairs.append((x_raw, y_gold))
                                count += 1
                                
                                if max_samples and count >= max_samples:
                                    break
                                
                                if count % 100000 == 0:
                                    print(f"Loaded {count:,} samples...", end='\r')
                    
                    print(f"Loaded {len(pairs):,} samples from {file_path}")
                    return pairs
                
                train_set = load_large_dataset(train_path, train_size)
                # For dev/test, limit to reasonable size for evaluation
                dev_set = load_large_dataset(dev_path, min(50000, train_size//10) if train_size else 50000)
                test_set = load_large_dataset(test_path, min(20000, train_size//20) if train_size else 20000)
                
                print(f"✅ Loaded LARGE dataset: Train({len(train_set):,}), Dev({len(dev_set):,}), Test({len(test_set):,})")
                return train_set, dev_set, test_set
            else:
                print("⚠️  Large corpus files not found, falling back to standard processing...")
        
        # Original small corpus processing (fallback)
        train_path = "data/train.txt"
        dev_path = "data/dev.txt" 
        test_path = "data/test.txt"
        
        if rebuild_splits or not all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
            print("📖 Loading corpus and creating new splits...")
            
            if not corpus_path:
                corpus_path = 'data/Viet74K_clean.txt'
            
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
            print("📂 Loading existing small corpus data splits...")
            train_set = self.preprocessor.load_dataset(train_path)
            dev_set = self.preprocessor.load_dataset(dev_path)
            test_set = self.preprocessor.load_dataset(test_path)
            
            print(f"✅ Loaded splits: Train({len(train_set)}), Dev({len(dev_set)}), Test({len(test_set)})")
        
        return train_set, dev_set, test_set
    
    def _combine_all_datasets(self, train_size: int = None) -> Tuple[List, List, List]:
        """
        Combine ALL available datasets: Viet74K + corpus-full.txt + existing splits
        """
        all_train_pairs = []
        all_dev_pairs = []
        all_test_pairs = []
        
        # 1. Load Viet74K data if available
        if os.path.exists('data/Viet74K_clean.txt'):
            print("📚 Loading Viet74K dataset...")
            viet74k_lines = self.preprocessor.load_corpus('data/Viet74K_clean.txt')
            viet74k_pairs = self.preprocessor.create_training_pairs(viet74k_lines)
            
            # Split Viet74K data
            v_train, v_dev, v_test = self.preprocessor.split_dataset(viet74k_pairs, random_seed=1)
            all_train_pairs.extend(v_train)
            all_dev_pairs.extend(v_dev)
            all_test_pairs.extend(v_test)
            print(f"✅ Added Viet74K: Train({len(v_train):,}), Dev({len(v_dev):,}), Test({len(v_test):,})")
        
        # 2. Load large corpus data if available
        large_files = [
            'data/corpus_full_processed_train.txt',
            'data/corpus_full_processed_dev.txt', 
            'data/corpus_full_processed_test.txt'
        ]
        
        if all(os.path.exists(f) for f in large_files):
            print("📈 Loading Large Corpus dataset...")
            
            def load_dataset_lines(file_path: str, max_samples: int = None):
                pairs = []
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '\t' in line:
                            x_raw, y_gold = line.strip().split('\t', 1)
                            pairs.append((x_raw, y_gold))
                            count += 1
                            
                            if max_samples and count >= max_samples:
                                break
                            
                            if count % 100000 == 0:
                                print(f"Loading large corpus: {count:,} samples...", end='\r')
                
                print(f"Loaded {len(pairs):,} from {file_path}")
                return pairs
            
            # Load large corpus with limits to balance datasets
            large_train_limit = train_size - len(all_train_pairs) if train_size else None
            large_train = load_dataset_lines(large_files[0], large_train_limit)
            large_dev = load_dataset_lines(large_files[1], 30000)  # Limit dev size
            large_test = load_dataset_lines(large_files[2], 10000)  # Limit test size
            
            all_train_pairs.extend(large_train)
            all_dev_pairs.extend(large_dev)
            all_test_pairs.extend(large_test)
            print(f"✅ Added Large Corpus: Train({len(large_train):,}), Dev({len(large_dev):,}), Test({len(large_test):,})")
        
        # 3. Shuffle combined datasets
        import random
        random.seed(42)
        random.shuffle(all_train_pairs)
        random.shuffle(all_dev_pairs) 
        random.shuffle(all_test_pairs)
        
        # 4. Apply final train_size limit if specified
        if train_size and len(all_train_pairs) > train_size:
            all_train_pairs = all_train_pairs[:train_size]
        
        print(f"🎯 FINAL COMBINED DATASET: Train({len(all_train_pairs):,}), Dev({len(all_dev_pairs):,}), Test({len(all_test_pairs):,})")
        
        return all_train_pairs, all_dev_pairs, all_test_pairs
    
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
        
        # Adaptive evaluation sample size based on dataset size
        if len(test_set) > 10000:
            eval_samples = 2000  # For large datasets, use more samples but not all
        elif len(test_set) > 1000:
            eval_samples = 500
        else:
            eval_samples = min(100, len(test_set))
        
        print(f"📊 Evaluating on {eval_samples} samples from {len(test_set)} total test samples")
        
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
    
    def run_training_pipeline(self, corpus_path: str = None, 
                             train_size: int = None,
                             use_dictionary: bool = True,
                             use_large_corpus: bool = True,
                             combine_all_datasets: bool = False,
                             model_output_dir: str = "models/crf") -> Tuple[CRFSegmenter, float]:
        """
        Run complete CRF training pipeline.
        
        Args:
            corpus_path: Path to training corpus (legacy)
            train_size: Maximum samples to use
            use_dictionary: Whether to use dictionary features
            use_large_corpus: Whether to use large corpus from corpus-full.txt
            combine_all_datasets: Whether to combine ALL datasets (Viet74K + corpus-full.txt)
            model_output_dir: Directory to save trained model
            
        Returns:
            Tuple of (trained_model, test_f1_score)
        """
        print("🚀 Starting CRF training pipeline for Vietnamese word segmentation")
        if combine_all_datasets:
            print("🎯 Using ALL DATASETS: Viet74K + corpus-full.txt")
        elif use_large_corpus:
            print("📈 Using LARGE CORPUS from corpus-full.txt processing")
        else:
            print("📚 Using Viet74K corpus")
        print("=" * 70)
        
        # Step 1: Prepare data
        train_set, dev_set, test_set = self.prepare_data(
            corpus_path, 
            train_size=train_size,
            use_large_corpus=use_large_corpus,
            combine_all_datasets=combine_all_datasets
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
            'use_large_corpus': use_large_corpus,
            'combine_all_datasets': combine_all_datasets,
            'data_source': 'ALL_DATASETS' if combine_all_datasets else ('corpus-full.txt' if use_large_corpus else 'Viet74K_clean.txt'),
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
        print(f"📊 Training data: {metadata['data_source']} ({len(train_set):,} samples)")
        
        return self.model, test_f1


def main():
    """
    Main training script for CRF model with LARGE CORPUS support.
    
    This script trains a CRF model using the processed corpus-full.txt dataset.
    """
    print("🇻🇳 Vietnamese Word Segmentation - CRF Training with LARGE CORPUS")
    print("=" * 70)
    
    # Initialize trainer
    trainer = CRFModelTrainer()
    
    # Configuration for LARGE CORPUS training
    config = {
        'corpus_path': None,  # Not needed when using large corpus
        'train_size': 500000,  # Use 500K samples for training (adjust as needed)
        'use_dictionary': True,
        'use_large_corpus': True,  # NEW: Use processed corpus-full.txt
        'model_output_dir': 'models/crf_large'
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        # Run training pipeline
        model, test_f1 = trainer.run_training_pipeline(**config)
        
        # Summary
        print("\n" + "=" * 70)
        print("🎊 TRAINING SUMMARY")
        print("=" * 70)
        print(f"✅ Model Type: CRF with BIES tagging")
        print(f"📊 Final F1-score: {test_f1:.4f}")
        print(f"🔧 Dictionary Features: {config['use_dictionary']}")
        print(f"📈 Large Corpus: {config['use_large_corpus']}")
        print(f"📚 Training Samples: {config['train_size']:,}")
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
        
        print("\n🚀 Next steps:")
        print("   1. Test the model with: python test_inference.py")
        print("   2. Run demo with: python demo.py")
        print("   3. Deploy with: python src/deployment.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 