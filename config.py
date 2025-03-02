import numpy as np

# Dataset Configuration
DATASET_PATH = '../Quantum_Brainathon-2025_Dataset'
SAMPLING_RATE = 250  # Hz
N_SUBJECTS = 24
CHANNELS = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'Cz', 'Pz', 'Oz']

# Frequency Bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Preprocessing Configuration
PREPROCESSING = {
    'bandpass_filter': {
        'lowcut': 0.5,
        'highcut': 45
    },
    'notch_filter': {
        'freq': 50  # Power line frequency
    },
    'artifact_threshold': 100,  # μV
    'ica': {
        'n_components': 8,
        'random_state': 42
    },
    'kalman_filter': {
        'process_variance': 1e-5,
        'measurement_variance': 0.1
    },
    'savgol_filter': {
        'window_length': 15,
        'polyorder': 3
    },
    'signal_quality': {
        'snr_threshold': 10,  # dB
        'correlation_threshold': 0.7
    },
    'adaptive_threshold': {
        'window_size': 1000,  # samples
        'n_std': 3  # number of standard deviations
    }
}

# Feature Extraction Configuration
FEATURE_EXTRACTION = {
    'window_size': 500,  # samples (2 seconds)
    'overlap': 0.5,  # 50% overlap
    'wavelet': 'db4',
    'entropy_params': {
        'sample_entropy_m': 2,
        'sample_entropy_r': 0.2
    }
}

# Model Configuration
MODEL = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}
