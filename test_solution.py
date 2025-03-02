import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import EEGDataLoader
from preprocessing import preprocess_eeg, apply_bandpass_filter, detect_bad_channels
from feature_extraction import extract_band_powers, compute_hjorth_parameters
from model import AttentionClassifier
from config import CHANNELS, SAMPLING_RATE

class TestEEGSolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample EEG data
        cls.n_samples = 1000
        cls.n_channels = len(CHANNELS)
        np.random.seed(42)
        
        # Generate synthetic EEG data
        cls.sample_eeg = pd.DataFrame(
            np.random.randn(cls.n_samples, cls.n_channels),
            columns=CHANNELS
        )
        
        # Add some artificial oscillations
        t = np.linspace(0, 4, cls.n_samples)
        alpha = 10 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha rhythm
        beta = 5 * np.sin(2 * np.pi * 20 * t)    # 20 Hz beta rhythm
        cls.sample_eeg['Oz'] += alpha
        cls.sample_eeg['Fz'] += beta
        
        # Create sample markers
        cls.sample_markers = pd.DataFrame({
            'timestamp': np.arange(0, cls.n_samples, 100),
            'event': ['stimulus'] * 10
        })

    def test_data_loader(self):
        """Test data loader functionality."""
        loader = EEGDataLoader()
        
        # Test dataset validation
        validation_results = loader.validate_dataset()
        self.assertIsInstance(validation_results, dict)
        self.assertTrue('missing_files' in validation_results)
        self.assertTrue('corrupt_files' in validation_results)
        
    def test_preprocessing(self):
        """Test preprocessing functions."""
        # Test bandpass filter
        filtered_data = apply_bandpass_filter(self.sample_eeg)
        self.assertEqual(filtered_data.shape, self.sample_eeg.shape)
        self.assertTrue(all(col in filtered_data.columns for col in CHANNELS))
        
        # Test complete preprocessing pipeline
        preprocessed_data = preprocess_eeg(self.sample_eeg)
        self.assertEqual(preprocessed_data.shape, self.sample_eeg.shape)
        
        # Test bad channel detection
        # Add artificial noise to one channel
        noisy_data = self.sample_eeg.copy()
        noisy_data['Fp1'] = noisy_data['Fp1'] * 100
        bad_channels = detect_bad_channels(noisy_data)
        self.assertIn('Fp1', bad_channels)
        
    def test_feature_extraction(self):
        """Test feature extraction functions."""
        # Test band power extraction
        powers = extract_band_powers(self.sample_eeg['Oz'], SAMPLING_RATE)
        self.assertTrue(all(band in powers for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']))
        
        # Test Hjorth parameters
        hjorth = compute_hjorth_parameters(self.sample_eeg['Oz'].values)
        self.assertTrue(all(param in hjorth for param in ['activity', 'mobility', 'complexity']))
        
    def test_model(self):
        """Test model functionality."""
        # Create sample data for classification
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        y = np.random.choice(['oddball', 'stroop', 'switching', 'dual'], 100)
        
        # Initialize and train model
        classifier = AttentionClassifier()
        X, y_encoded = classifier.prepare_data(pd.DataFrame(X), y)
        
        # Test training
        classifier.train(X, y_encoded)
        self.assertTrue(hasattr(classifier.model, 'feature_importances_'))
        
        # Test prediction
        predictions = classifier.predict(X)
        self.assertEqual(len(predictions), len(y))
        
        # Test evaluation
        results = classifier.evaluate(X, y_encoded)
        self.assertTrue('accuracy' in results)
        self.assertTrue('confusion_matrix' in results)
        
    def test_data_integrity(self):
        """Test data integrity and format."""
        # Test EEG data format
        self.assertEqual(len(self.sample_eeg.columns), len(CHANNELS))
        self.assertEqual(self.sample_eeg.index.dtype, np.dtype('int64'))
        
        # Test for NaN values
        self.assertFalse(self.sample_eeg.isnull().any().any())
        
        # Test marker data format
        self.assertTrue('timestamp' in self.sample_markers.columns)
        self.assertTrue('event' in self.sample_markers.columns)
        
    def test_preprocessing_edge_cases(self):
        """Test preprocessing with edge cases."""
        # Test with zeros
        zero_data = pd.DataFrame(np.zeros((100, len(CHANNELS))), columns=CHANNELS)
        preprocessed_zeros = preprocess_eeg(zero_data)
        self.assertFalse(preprocessed_zeros.isnull().any().any())
        
        # Test with very large values
        large_data = self.sample_eeg * 1e6
        preprocessed_large = preprocess_eeg(large_data)
        self.assertFalse(preprocessed_large.isnull().any().any())
        
        # Test with missing values
        data_with_nan = self.sample_eeg.copy()
        data_with_nan.iloc[0, 0] = np.nan
        preprocessed_nan = preprocess_eeg(data_with_nan)
        self.assertFalse(preprocessed_nan.isnull().any().any())
        
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Create and train a model
        X = np.random.randn(100, 10)
        y = np.random.choice(['oddball', 'stroop', 'switching', 'dual'], 100)
        
        classifier = AttentionClassifier()
        X, y_encoded = classifier.prepare_data(pd.DataFrame(X), y)
        classifier.train(X, y_encoded)
        
        # Save model
        temp_path = 'temp_model.joblib'
        classifier.save_model(temp_path)
        
        # Load model
        loaded_classifier = AttentionClassifier.load_model(temp_path)
        
        # Compare predictions
        original_pred = classifier.predict(X)
        loaded_pred = loaded_classifier.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        Path(temp_path).unlink()

if __name__ == '__main__':
    unittest.main(verbosity=2)
