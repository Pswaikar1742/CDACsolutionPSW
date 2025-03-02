"""
Configuration parameters for quantum computing operations.
"""

# Quantum Circuit Parameters
QUANTUM_PARAMS = {
    'n_qubits': 4,              # Number of qubits to use
    'feature_map_reps': 3,      # Number of repetitions in feature map
    'feature_map_type': 'pauli', # Type of feature map ('pauli' or 'zz')
    'entanglement': 'full',     # Entanglement pattern ('full', 'linear', or 'circular')
    'measurement_basis': 'z',    # Measurement basis
    'shots': 1024               # Number of shots for quantum measurements
}

# Quantum Feature Mapping
FEATURE_MAP_PARAMS = {
    'pauli_operators': ['Z', 'ZZ', 'ZZZ'],  # Pauli operators for feature map
    'rotation_blocks': True,    # Whether to include rotation blocks
    'barrier': True,            # Whether to include barriers in circuit
    'parameter_prefix': 'θ'     # Prefix for parameter names
}

# Quantum Dimensionality Reduction
DIMENSION_REDUCTION = {
    'n_components': 4,          # Number of components in reduced space
    'optimization_level': 1,    # Circuit optimization level
    'seed_transpiler': 42,      # Seed for circuit transpilation
    'seed_simulator': 42        # Seed for quantum simulator
}

# Variational Quantum Classifier
VQC_PARAMS = {
    'n_layers': 2,             # Number of variational layers
    'optimizer': {
        'name': 'SPSA',        # Optimizer name
        'maxiter': 100,        # Maximum iterations
        'learning_rate': 0.1,  # Learning rate
        'perturbation': 0.1    # Perturbation size
    },
    'loss': 'cross_entropy',   # Loss function
    'callback_metrics': [      # Metrics to track during training
        'loss',
        'accuracy',
        'quantum_cost'
    ]
}

# Quantum Kernel Classifier
KERNEL_PARAMS = {
    'kernel_type': 'quantum',  # Type of kernel ('quantum' or 'classical')
    'optimizer': {
        'name': 'COBYLA',     # Optimizer name
        'maxiter': 100,       # Maximum iterations
        'tol': 1e-3          # Tolerance
    },
    'feature_dimension': 4,    # Dimension of feature space
    'gamma': 'auto'           # Kernel coefficient
}

# Circuit Analysis
CIRCUIT_ANALYSIS = {
    'depth_analysis': True,    # Whether to analyze circuit depth
    'width_analysis': True,    # Whether to analyze circuit width
    'cost_analysis': True,     # Whether to analyze quantum cost
    'error_analysis': {        # Error analysis parameters
        'enabled': True,
        'error_model': 'basic',
        'noise_factors': [
            'decoherence',
            'gate_error',
            'readout_error'
        ]
    }
}

# Quantum State Analysis
STATE_ANALYSIS = {
    'visualization': {
        'enabled': True,
        'plot_type': 'bloch',  # Type of visualization ('bloch', 'histogram', 'heatmap')
        'save_figures': True
    },
    'metrics': [
        'fidelity',
        'purity',
        'entropy'
    ]
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'backend_name': 'aer_simulator',  # Quantum backend to use
    'backend_options': {
        'method': 'statevector',
        'device': 'CPU',
        'precision': 'double',
        'max_parallel_threads': 8,
        'max_parallel_experiments': 1,
        'max_parallel_shots': 1024
    },
    'noise_model': {
        'enabled': False,
        'basis_gates': ['u1', 'u2', 'u3', 'cx'],
        'coupling_map': None
    }
}

# Result Analysis
RESULT_ANALYSIS = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'quantum_cost'
    ],
    'visualization': {
        'confusion_matrix': True,
        'roc_curve': True,
        'learning_curves': True,
        'quantum_states': True
    },
    'statistical_tests': [
        'wilcoxon',
        'friedman'
    ]
}

# Quantum Pipeline
PIPELINE_CONFIG = {
    'preprocessing': {
        'feature_scaling': True,
        'dimensionality_reduction': True
    },
    'feature_mapping': {
        'type': 'pauli',
        'optimization': True
    },
    'classification': {
        'method': 'kernel',  # 'kernel' or 'variational'
        'cross_validation': True,
        'n_splits': 5
    },
    'postprocessing': {
        'ensemble_voting': True,
        'calibration': True
    }
}

# Experiment Tracking
EXPERIMENT_TRACKING = {
    'enabled': True,
    'metrics_to_track': [
        'accuracy',
        'quantum_cost',
        'circuit_depth',
        'training_time'
    ],
    'save_checkpoints': True,
    'checkpoint_frequency': 10
}
