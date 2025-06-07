"""
Vietnamese Word Segmentation using Conditional Random Fields (CRF)

This module implements a CRF-based approach for Vietnamese word segmentation.
The task involves converting unsegmented text (without spaces and diacritics) 
into properly segmented text with spaces between words.

Example:
    Input:  "xinchao" (no spaces, no diacritics)
    Output: "xin chao" (properly segmented)

The CRF model uses BIES (Begin, Inside, End, Single) tagging scheme for 
sequence labeling to identify word boundaries.
"""

import re
import pickle
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import os


class CRFFeatureExtractor:
    """
    Feature extractor for CRF model in Vietnamese word segmentation.
    
    This class extracts rich features for each character position including:
    - Character-level features (unigrams, bigrams, trigrams)
    - Positional features (beginning/end of sequence)
    - Dictionary-based features (if dictionary provided)
    - Character type features (alphabetic, numeric)
    """
    
    def __init__(self, dictionary: Set[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            dictionary: Optional set of known words for dictionary-based features
        """
        self.dictionary = dictionary or set()
    
    def char_features(self, text: str, i: int) -> Dict[str, bool]:
        """
        Extract features for character at position i in the text.
        
        Args:
            text: Input text sequence
            i: Character position (0-indexed)
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        char = text[i]
        
        # Basic character features
        features['char'] = char
        features['char.lower'] = char.lower()
        features['char.isdigit'] = char.isdigit()
        features['char.isalpha'] = char.isalpha()
        
        # Position features - important for sequence boundaries
        features['BOS'] = i == 0  # Beginning of sequence
        features['EOS'] = i == len(text) - 1  # End of sequence
        
        # Context features - previous character
        if i > 0:
            features['char-1'] = text[i-1]
            features['char-1.lower'] = text[i-1].lower()
            features['bigram-1'] = text[i-1:i+1]  # Previous + current
        else:
            features['BOS-1'] = True
            
        # Context features - next character
        if i < len(text) - 1:
            features['char+1'] = text[i+1]
            features['char+1.lower'] = text[i+1].lower()
            features['bigram+1'] = text[i:i+2]  # Current + next
        else:
            features['EOS+1'] = True
        
        # Trigram features for wider context
        if i > 1:
            features['trigram-2'] = text[i-2:i+1]  # Previous 2 + current
        if i < len(text) - 2:
            features['trigram+2'] = text[i:i+3]  # Current + next 2
        
        # Dictionary-based features
        if self.dictionary:
            # Check if any word in dictionary starts at this position
            for length in range(1, min(8, len(text) - i + 1)):
                substring = text[i:i+length].lower()
                if substring in self.dictionary:
                    features[f'dict_match_{length}'] = True
        
        return features
    
    def text_to_features(self, text: str) -> List[Dict[str, bool]]:
        """
        Convert text sequence to feature sequence.
        
        Args:
            text: Input text
            
        Returns:
            List of feature dictionaries, one per character
        """
        return [self.char_features(text, i) for i in range(len(text))]


class CRFSegmenter:
    """
    CRF-based Vietnamese word segmentation model.
    
    This model uses Conditional Random Fields with BIES tagging scheme:
    - B: Beginning of word
    - I: Inside word (continuation)
    - E: End of word
    - S: Single character word
    
    The model learns to predict these tags for each character, then reconstructs
    word boundaries from the tag sequence.
    """
    
    def __init__(self, dictionary: Set[str] = None):
        """
        Initialize CRF segmenter.
        
        Args:
            dictionary: Optional dictionary for enhanced features
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',        # L-BFGS optimization algorithm
            c1=0.1,                   # L1 regularization coefficient
            c2=0.1,                   # L2 regularization coefficient
            max_iterations=100,       # Maximum training iterations
            all_possible_transitions=True  # Allow all tag transitions
        )
        self.feature_extractor = CRFFeatureExtractor(dictionary)
        self.is_trained = False
    
    def prepare_training_data(self, pairs: List[Tuple[str, str]]) -> Tuple[List, List]:
        """
        Prepare training data for CRF model.
        
        Args:
            pairs: List of (input_text, target_text) pairs
            
        Returns:
            Tuple of (features, labels) for training
        """
        X_features = []
        Y_labels = []
        
        for x_raw, y_gold in pairs:
            # Create BIES labels from gold segmentation
            labels = self.create_labels(x_raw, y_gold)
            if len(labels) == len(x_raw):
                features = self.feature_extractor.text_to_features(x_raw)
                X_features.append(features)
                Y_labels.append(labels)
        
        return X_features, Y_labels
    
    def create_labels(self, x_raw: str, y_gold: str) -> List[str]:
        """
        Create BIES labels from gold segmentation.
        
        Args:
            x_raw: Input text without spaces
            y_gold: Gold standard text with spaces
            
        Returns:
            List of BIES labels
        """
        # Remove spaces and normalize for comparison
        y_clean = re.sub(r'[^\w]', '', y_gold.lower())
        
        # Ensure input and gold text match after normalization
        if len(y_clean) != len(x_raw):
            # Handle mismatched lengths by character-level alignment
            return self._align_and_label(x_raw, y_gold)
        
        # Create labels based on word boundaries in y_gold
        labels = []
        words = y_gold.split()
        char_idx = 0
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            word_length = len(word_clean)
            
            if word_length == 1:
                labels.append('S')  # Single character word
            elif word_length > 1:
                labels.append('B')  # Beginning
                for _ in range(word_length - 2):
                    labels.append('I')  # Inside
                labels.append('E')  # End
            
            char_idx += word_length
        
        return labels
    
    def _align_and_label(self, x_raw: str, y_gold: str) -> List[str]:
        """
        Handle misaligned text by simple heuristic labeling.
        
        Args:
            x_raw: Input text
            y_gold: Gold segmentation
            
        Returns:
            List of labels (may be approximate)
        """
        # Simple fallback: assume each 2-3 characters form a word
        labels = []
        i = 0
        while i < len(x_raw):
            word_len = min(3, len(x_raw) - i)  # Default word length
            
            if word_len == 1:
                labels.append('S')
            else:
                labels.append('B')
                for _ in range(word_len - 2):
                    labels.append('I')
                labels.append('E')
            
            i += word_len
        
        return labels[:len(x_raw)]  # Ensure correct length
    
    def train(self, X_features: List, Y_labels: List):
        """
        Train the CRF model.
        
        Args:
            X_features: List of feature sequences
            Y_labels: List of label sequences
        """
        print(f"Training CRF with {len(X_features)} sequences...")
        self.model.fit(X_features, Y_labels)
        self.is_trained = True
        print("CRF training completed!")
    
    def predict(self, text: str) -> List[str]:
        """
        Predict BIES labels for input text.
        
        Args:
            text: Input text without spaces
            
        Returns:
            List of BIES labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        features = self.feature_extractor.text_to_features(text)
        return self.model.predict_single(features)
    
    def predict_multiple(self, text: str, n_best: int = 5) -> List[Tuple[List[str], float]]:
        """
        Predict multiple BIES label sequences with confidence scores.
        
        Args:
            text: Input text without spaces
            n_best: Number of best predictions to return
            
        Returns:
            List of (labels, confidence_score) tuples, sorted by confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        features = self.feature_extractor.text_to_features(text)
        
        # Get probability distributions for each position
        try:
            # Try to get marginal probabilities if available
            marginal_prob = self.model.predict_marginals_single(features)
            
            # Generate multiple hypotheses using beam search-like approach
            predictions_with_scores = self._beam_search_decode(marginal_prob, n_best)
            
        except AttributeError:
            # Fallback: Generate variations from single best prediction
            best_labels = self.model.predict_single(features)
            predictions_with_scores = self._generate_variations(text, best_labels, n_best)
        
        return predictions_with_scores[:n_best]
    
    def _beam_search_decode(self, marginal_prob: List[Dict], n_best: int = 5) -> List[Tuple[List[str], float]]:
        """
        Decode multiple sequences using marginal probabilities.
        
        Args:
            marginal_prob: List of probability distributions for each position
            n_best: Number of best sequences to keep
            
        Returns:
            List of (labels, score) tuples
        """
        labels = ['B', 'I', 'E', 'S']
        
        # Simple beam search implementation
        beams = [([''], 0.0)]  # (sequence, score)
        
        for pos_probs in marginal_prob:
            new_beams = []
            
            for sequence, score in beams:
                for label in labels:
                    if label in pos_probs:
                        new_score = score + np.log(pos_probs[label] + 1e-10)
                        new_sequence = sequence + [label]
                        new_beams.append((new_sequence, new_score))
            
            # Keep only top n_best beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:n_best]
        
        # Convert scores to normalized confidence
        final_results = []
        scores = [score for _, score in beams]
        max_score = max(scores) if scores else 0
        
        for sequence, score in beams:
            confidence = np.exp(score - max_score)  # Normalize
            final_results.append((sequence[1:], confidence))  # Remove initial empty string
        
        return final_results
    
    def _generate_variations(self, text: str, best_labels: List[str], n_best: int = 5) -> List[Tuple[List[str], float]]:
        """
        Generate variations from the best prediction by modifying some boundaries.
        
        Args:
            text: Original text
            best_labels: Best predicted labels
            n_best: Number of variations to generate
            
        Returns:
            List of (labels, confidence) tuples
        """
        variations = [(best_labels, 1.0)]  # Original prediction with highest confidence
        
        # Generate variations by changing some B/E boundaries
        for _ in range(n_best - 1):
            new_labels = best_labels.copy()
            
            # Find potential split points
            for i in range(1, len(new_labels) - 1):
                if best_labels[i] in ['I', 'E'] and best_labels[i-1] in ['B', 'I']:
                    # Try splitting here
                    variation = best_labels.copy()
                    variation[i-1] = 'E' if variation[i-1] == 'I' else 'S' if variation[i-1] == 'B' else variation[i-1]
                    variation[i] = 'B' if variation[i] == 'I' else 'S' if variation[i] == 'E' else variation[i]
                    
                    # Ensure valid BIES sequence
                    variation = self._fix_bies_sequence(variation)
                    
                    confidence = 0.9 - (len(variations) * 0.1)  # Decreasing confidence
                    variations.append((variation, max(confidence, 0.1)))
                    
                    if len(variations) >= n_best:
                        break
            
            if len(variations) >= n_best:
                break
        
        return variations[:n_best]
    
    def _fix_bies_sequence(self, labels: List[str]) -> List[str]:
        """
        Fix BIES sequence to ensure valid transitions.
        
        Args:
            labels: Potentially invalid BIES sequence
            
        Returns:
            Valid BIES sequence
        """
        fixed = []
        
        for i, label in enumerate(labels):
            if i == 0:
                # First position can only be B or S
                fixed.append('B' if label in ['B', 'I'] else 'S')
            else:
                prev_label = fixed[i-1]
                
                if prev_label in ['B', 'I']:
                    # After B or I, can only have I or E
                    fixed.append('I' if label in ['B', 'I'] else 'E')
                else:  # prev_label in ['E', 'S']
                    # After E or S, can only have B or S
                    fixed.append('B' if label in ['B', 'I'] else 'S')
        
        return fixed

    def segment(self, text: str) -> str:
        """
        Segment text using trained CRF model.
        
        Args:
            text: Input text without spaces/diacritics
            
        Returns:
            Segmented text with spaces
        """
        if not text:
            return ""
        
        labels = self.predict(text)
        
        # Reconstruct words from BIES labels
        result = []
        current_word = ""
        
        for char, label in zip(text, labels):
            if label in ['S', 'B']:
                # Start new word
                if current_word:
                    result.append(current_word)
                current_word = char
            elif label in ['I', 'E']:
                # Continue current word
                current_word += char
                if label == 'E':
                    # End current word
                    result.append(current_word)
                    current_word = ""
        
        # Handle any remaining characters
        if current_word:
            result.append(current_word)
        
        return ' '.join(result)
    
    def segment_multiple(self, text: str, n_best: int = 5) -> List[Tuple[str, float]]:
        """
        Generate multiple segmentation candidates with confidence scores.
        
        Args:
            text: Input text without spaces/diacritics
            n_best: Number of segmentation candidates to return
            
        Returns:
            List of (segmented_text, confidence) tuples, sorted by confidence
        """
        if not text:
            return [("", 1.0)]
        
        predictions = self.predict_multiple(text, n_best)
        results = []
        
        for labels, confidence in predictions:
            segmented = self._labels_to_text(text, labels)
            results.append((segmented, confidence))
        
        return results
    
    def _labels_to_text(self, text: str, labels: List[str]) -> str:
        """
        Convert BIES labels back to segmented text.
        
        Args:
            text: Original text
            labels: BIES labels
            
        Returns:
            Segmented text with spaces
        """
        if len(text) != len(labels):
            # Fallback to original segmentation
            return self.segment(text)
        
        result = []
        current_word = ""
        
        for char, label in zip(text, labels):
            if label in ['S', 'B']:
                # Start new word
                if current_word:
                    result.append(current_word)
                current_word = char
            elif label in ['I', 'E']:
                # Continue current word
                current_word += char
                if label == 'E':
                    # End current word
                    result.append(current_word)
                    current_word = ""
        
        # Handle any remaining characters
        if current_word:
            result.append(current_word)
        
        return ' '.join(result)
    
    def save(self, file_path: str):
        """Save trained model to file."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, file_path: str):
        """Load trained model from file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_extractor = data['feature_extractor']
            self.is_trained = data['is_trained']


class ContextAwareCRFSegmenter(CRFSegmenter):
    """
    Enhanced CRF segmenter with context-aware suggestions and meaningful predictions.
    
    This class extends the basic CRF segmenter with:
    - Context-aware multiple suggestions
    - Dictionary-based validation for meaningful words
    - Punctuation and capitalization preservation
    - Linguistic structure understanding
    """
    
    def __init__(self, dictionary: Set[str] = None, vietnamese_dict: Set[str] = None):
        """
        Initialize context-aware CRF segmenter.
        
        Args:
            dictionary: Basic dictionary for features
            vietnamese_dict: Vietnamese vocabulary for validation
        """
        super().__init__(dictionary)
        self.vietnamese_dict = vietnamese_dict or set()
        self.punctuation_map = {
            ',': ',', '.': '.', '?': '?', '!': '!', 
            ':': ':', ';': ';', '-': '-', '(': '(', ')': ')'
        }
        
    def load_vietnamese_dictionary(self, dict_path: str = None):
        """Load Vietnamese dictionary for validation."""
        if dict_path and os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.vietnamese_dict = set(line.strip().lower() for line in f)
        else:
            # Build from training data if no external dict
            print("📚 Building Vietnamese dictionary from training data...")
            
    def extract_punctuation_and_case(self, text: str) -> Dict:
        """
        Extract punctuation positions and capitalization info.
        
        Args:
            text: Input text with possible punctuation and mixed case
            
        Returns:
            Dictionary with punctuation and case information
        """
        info = {
            'punctuation': {},  # position -> punctuation char
            'capitals': set(),  # positions of capital letters
            'sentence_starts': []  # positions where sentences start
        }
        
        for i, char in enumerate(text):
            if char.isupper():
                info['capitals'].add(i)
            if char in self.punctuation_map:
                info['punctuation'][i] = char
            # Detect sentence start (after . ! ? or beginning)
            if i == 0 or (i > 0 and text[i-1] in '.!?'):
                info['sentence_starts'].append(i)
                
        return info
        
    def restore_punctuation_and_case(self, segmented: str, original_info: Dict) -> str:
        """
        Restore punctuation and capitalization to segmented text.
        
        Args:
            segmented: Segmented text without punctuation/case
            original_info: Information extracted from original text
            
        Returns:
            Text with restored punctuation and capitalization
        """
        result = list(segmented)
        char_mapping = []  # Track original position for each result char
        
        # Build mapping from result position to original position
        result_pos = 0
        for orig_pos, char in enumerate(segmented.replace(' ', '')):
            if result_pos < len(result) and result[result_pos] == char:
                char_mapping.append(orig_pos)
                result_pos += 1
        
        # Apply capitalization
        for orig_pos in original_info['capitals']:
            if orig_pos < len(char_mapping):
                result_pos = char_mapping.index(orig_pos) if orig_pos in char_mapping else -1
                if result_pos >= 0 and result_pos < len(result):
                    result[result_pos] = result[result_pos].upper()
        
        # Apply sentence capitalization
        words = segmented.split()
        if words:
            # Capitalize first word
            words[0] = words[0].capitalize()
            
        # Insert punctuation (simplified approach)
        for pos, punct in original_info['punctuation'].items():
            # Insert at appropriate word boundaries
            if punct in ',.!?:;':
                segmented += punct  # Simplified - append at end
        
        return ' '.join(words) + ''.join(original_info['punctuation'].values())
        
    def calculate_meaningfulness_score(self, words: List[str]) -> float:
        """
        Calculate how meaningful a segmentation is based on dictionary matches.
        
        Args:
            words: List of segmented words
            
        Returns:
            Score between 0 and 1 (higher = more meaningful)
        """
        if not words:
            return 0.0
            
        meaningful_words = 0
        total_words = len(words)
        
        for word in words:
            word_clean = word.lower().strip()
            # Check in dictionary
            if word_clean in self.vietnamese_dict:
                meaningful_words += 1
            # Prefer longer meaningful words
            elif len(word_clean) >= 3:
                # Partial credit for longer words (might be compound words)
                meaningful_words += 0.5
            elif len(word_clean) == 1:
                # Single characters get some credit (pronouns, particles)
                meaningful_words += 0.3
                
        return meaningful_words / total_words
        
    def segment_with_context(self, text: str, n_best: int = 5) -> List[Tuple[str, float]]:
        """
        Generate context-aware segmentations with meaningfulness scores.
        
        Args:
            text: Input text
            n_best: Number of suggestions to return
            
        Returns:
            List of (segmented_text, score) sorted by meaningfulness
        """
        if not text:
            return [("", 1.0)]
            
        # Extract original formatting info
        original_info = self.extract_punctuation_and_case(text)
        
        # Clean text for segmentation
        clean_text = re.sub(r'[^\w]', '', text.lower())
        # Import here to avoid circular import
        from .data_preparation import VietnameseDataPreprocessor
        preprocessor = VietnameseDataPreprocessor()
        clean_text = preprocessor.remove_diacritics(clean_text)
        
        # Get multiple predictions
        multiple_predictions = self.segment_multiple(clean_text, n_best * 2)  # Get more to filter
        
        # Score and rank by meaningfulness
        scored_results = []
        for segmented, base_score in multiple_predictions:
            words = segmented.split()
            meaningfulness = self.calculate_meaningfulness_score(words)
            
            # Combined score: base CRF confidence + meaningfulness
            combined_score = 0.4 * base_score + 0.6 * meaningfulness
            
            # Restore formatting
            formatted = self.restore_punctuation_and_case(segmented, original_info)
            
            scored_results.append((formatted, combined_score))
            
        # Sort by combined score and return top n_best
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:n_best]
        
    def segment_smart(self, text: str) -> str:
        """
        Smart segmentation that returns the most meaningful result.
        
        Args:
            text: Input text
            
        Returns:
            Best segmented text with restored formatting
        """
        results = self.segment_with_context(text, n_best=1)
        return results[0][0] if results else text


def create_char_vocab(corpus_lines: List[str]) -> Dict[str, int]:
    """
    Create character vocabulary from corpus.
    
    Args:
        corpus_lines: List of text lines
        
    Returns:
        Dictionary mapping characters to indices
    """
    char_counter = Counter()
    
    for line in corpus_lines:
        # Remove spaces and special characters for vocabulary
        clean_line = re.sub(r'[^\w]', '', line.lower())
        for char in clean_line:
            char_counter[char] += 1
    
    # Create vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for char, count in char_counter.most_common():
        if count >= 2:  # Filter rare characters
            vocab[char] = len(vocab)
    
    return vocab


def build_dictionary_from_corpus(corpus_lines: List[str]) -> Set[str]:
    """
    Build word dictionary from corpus for enhanced CRF features.
    
    Args:
        corpus_lines: List of segmented text lines
        
    Returns:
        Set of known words
    """
    dictionary = set()
    
    for line in corpus_lines:
        words = line.strip().split()
        for word in words:
            # Clean and normalize word
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean and len(word_clean) > 0:
                dictionary.add(word_clean)
    
    return dictionary


def create_vietnamese_dictionary_from_data(data_files: List[str]) -> Set[str]:
    """
    Create Vietnamese dictionary from training data files.
    
    Args:
        data_files: List of data file paths
        
    Returns:
        Set of Vietnamese words
    """
    dictionary = set()
    # Import here to avoid circular import
    from .data_preparation import VietnameseDataPreprocessor
    preprocessor = VietnameseDataPreprocessor()
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '\t' in line:
                            _, y_gold = line.strip().split('\t', 1)
                            # Extract words and add to dictionary
                            words = y_gold.split()
                            for word in words:
                                # Clean and normalize word
                                word_clean = re.sub(r'[^\w]', '', word.lower())
                                if len(word_clean) >= 1:
                                    dictionary.add(word_clean)
                                    # Also add without diacritics
                                    word_no_diacritic = preprocessor.remove_diacritics(word_clean)
                                    dictionary.add(word_no_diacritic)
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                
    print(f"📚 Created Vietnamese dictionary with {len(dictionary)} words")
    return dictionary 