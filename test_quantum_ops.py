"""
Test module for quantum operations.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from quantum_ops import (
    QuantumFeatureMapper,
    QuantumKernelClassifier,
    QuantumCircuitAnalyzer,
    create_quantum_pipeline,
    analyze_quantum_results
)

class TestQuantumOps(unittest.TestCase):
    """Test cases for quantum operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.n_qubits = 4
        cls.n_samples = 10
        cls.feature_dim = 8
        
        # Create sample data
        cls.sample_features = np.random.rand(cls.n_samples, cls.feature_dim)
        cls.sample_labels = np.random.randint(0, 2, cls.n_samples)
        
        # Initialize quantum components
        cls.feature_mapper = QuantumFeatureMapper(n_qubits=cls.n_qubits)
        cls.classifier = QuantumKernelClassifier(n_qubits=cls.n_qubits)
    
    def test_quantum_feature_mapper_initialization(self):
        """Test QuantumFeatureMapper initialization."""
        self.assertEqual(self.feature_mapper.n_qubits, self.n_qubits)
        self.assertEqual(self.feature_mapper.reps, 2)
        self.assertIsNotNone(self.feature_mapper.feature_map)
        self.assertIsNotNone(self.feature_mapper.backend)
    
    def test_feature_encoding(self):
        """Test quantum feature encoding."""
        # Test single sample encoding
        single_features = self.sample_features[0]
        encoded_features = self.feature_mapper.encode_features(single_features)
        
        # Check output shape (2^n_qubits for statevector)
        expected_length = 2 ** self.n_qubits
        self.assertEqual(len(encoded_features), expected_length)
        
        # Check if output is normalized
        self.assertAlmostEqual(np.sum(np.abs(encoded_features) ** 2), 1.0, places=5)
    
    def test_batch_feature_encoding(self):
        """Test batch quantum feature encoding."""
        encoded_batch = self.feature_mapper.batch_encode_features(self.sample_features)
        
        # Check batch output shape
        expected_shape = (self.n_samples, 2 ** self.n_qubits)
        self.assertEqual(encoded_batch.shape, expected_shape)
        
        # Check if all states are normalized
        norms = np.sum(np.abs(encoded_batch) ** 2, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(self.n_samples))
    
    def test_quantum_classifier(self):
        """Test quantum classifier training and prediction."""
        # Prepare quantum features
        X_quantum = self.feature_mapper.batch_encode_features(self.sample_features)
        
        # Split data
        train_size = 8
        X_train = X_quantum[:train_size]
        y_train = self.sample_labels[:train_size]
        X_test = X_quantum[train_size:]
        
        # Train classifier
        self.classifier.train(X_train, y_train)
        
        # Make predictions
        predictions = self.classifier.predict(X_test)
        
        # Check prediction shape
        self.assertEqual(len(predictions), len(X_test))
        
        # Check prediction values are valid
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_quantum_pipeline(self):
        """Test complete quantum pipeline creation."""
        feature_mapper, classifier = create_quantum_pipeline(n_qubits=self.n_qubits)
        
        self.assertIsInstance(feature_mapper, QuantumFeatureMapper)
        self.assertIsInstance(classifier, QuantumKernelClassifier)
        self.assertEqual(feature_mapper.n_qubits, self.n_qubits)
        self.assertEqual(classifier.n_qubits, self.n_qubits)
    
    def test_result_analysis(self):
        """Test quantum result analysis."""
        predictions = np.array([0, 1, 0, 1, 0])
        true_labels = np.array([0, 1, 1, 1, 0])
        
        results = analyze_quantum_results(predictions, true_labels)
        
        self.assertIn('accuracy', results)
        self.assertIn('error_count', results)
        self.assertIn('total_samples', results)
        
        self.assertEqual(results['total_samples'], 5)
        self.assertEqual(results['error_count'], 1)
        self.assertAlmostEqual(results['accuracy'], 0.8)
    
    def test_circuit_analyzer(self):
        """Test quantum circuit analysis."""
        # Create a simple test circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.h(0)  # Add Hadamard gate
        qc.cx(0, 1)  # Add CNOT gate
        
        # Test circuit depth
        depth = QuantumCircuitAnalyzer.get_circuit_depth(qc)
        self.assertEqual(depth, 2)
        
        # Test circuit width
        width = QuantumCircuitAnalyzer.get_circuit_width(qc)
        self.assertEqual(width, self.n_qubits)
    
    def test_state_similarity(self):
        """Test quantum state similarity computation."""
        # Create two similar quantum states
        state1 = np.array([1, 0, 0, 0]) / np.sqrt(2)
        state2 = np.array([1, 0.1, 0, 0]) / np.sqrt(1.01)
        
        similarity = QuantumCircuitAnalyzer.compute_state_similarity(state1, state2)
        
        # Check if similarity is between 0 and 1
        self.assertTrue(0 <= similarity <= 1)
    
    def test_error_handling(self):
        """Test error handling in quantum operations."""
        # Test invalid number of qubits
        with self.assertRaises(ValueError):
            QuantumFeatureMapper(n_qubits=0)
        
        # Test invalid feature dimensions
        invalid_features = np.random.rand(5, 3)  # Wrong feature dimension
        with self.assertRaises(ValueError):
            self.feature_mapper.encode_features(invalid_features[0])

if __name__ == '__main__':
    unittest.main(verbosity=2)
