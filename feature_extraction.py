import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
import antropy
import networkx as nx
from scipy.integrate import simps
from config import FREQ_BANDS, SAMPLING_RATE, FEATURE_EXTRACTION

def compute_sample_entropy(data, m=FEATURE_EXTRACTION['entropy_params']['sample_entropy_m'],
                         r=FEATURE_EXTRACTION['entropy_params']['sample_entropy_r']):
    """
    Compute Sample Entropy of the signal.
    
    Parameters:
    -----------
    data : array-like
        Input signal
    m : int
        Embedding dimension
    r : float
        Tolerance
        
    Returns:
    --------
    float : Sample entropy value
    """
    return antropy.sample_entropy(data, order=m, metric='chebyshev', r=r)

def compute_lyapunov_exp(data, fs=SAMPLING_RATE):
    """
    Estimate the largest Lyapunov exponent.
    
    Parameters:
    -----------
    data : array-like
        Input signal
    fs : int
        Sampling frequency
        
    Returns:
    --------
    float : Largest Lyapunov exponent
    """
    return antropy.lyap_r(data, emb_dim=10, lag=int(fs/10))

def compute_phase_connectivity(data1, data2, fs=SAMPLING_RATE):
    """
    Compute phase-based connectivity metrics.
    
    Parameters:
    -----------
    data1, data2 : array-like
        Input signals
    fs : int
        Sampling frequency
        
    Returns:
    --------
    dict : Phase connectivity metrics
    """
    # Hilbert transform to get analytic signal
    analytic1 = signal.hilbert(data1)
    analytic2 = signal.hilbert(data2)
    
    # Extract instantaneous phase
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    
    # Phase difference
    phase_diff = phase1 - phase2
    
    # Phase Locking Value (PLV)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    # Phase Lag Index (PLI)
    pli = np.abs(np.mean(np.sign(phase_diff)))
    
    return {
        'plv': plv,
        'pli': pli
    }

def compute_spectral_connectivity(data1, data2, fs=SAMPLING_RATE):
    """
    Compute spectral connectivity metrics.
    
    Parameters:
    -----------
    data1, data2 : array-like
        Input signals
    fs : int
        Sampling frequency
        
    Returns:
    --------
    dict : Spectral connectivity metrics
    """
    # Compute cross-spectral density
    f, Cxy = signal.csd(data1, data2, fs=fs)
    
    # Compute coherence
    f, coh = signal.coherence(data1, data2, fs=fs)
    
    # Compute band-specific connectivity
    connectivity = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        # Find frequency indices for the band
        idx_band = np.logical_and(f >= low, f <= high)
        
        # Average coherence in band
        band_coh = np.mean(coh[idx_band])
        
        # Average cross-spectral density in band
        band_csd = np.mean(np.abs(Cxy[idx_band]))
        
        connectivity[f'{band_name}_coherence'] = band_coh
        connectivity[f'{band_name}_csd'] = band_csd
    
    return connectivity

def compute_graph_metrics(connectivity_matrix):
    """
    Compute graph theory metrics from connectivity matrix.
    
    Parameters:
    -----------
    connectivity_matrix : array-like
        Connectivity matrix between channels
        
    Returns:
    --------
    dict : Graph metrics
    """
    # Create graph from connectivity matrix
    G = nx.from_numpy_array(connectivity_matrix)
    
    metrics = {
        'density': nx.density(G),
        'clustering_coefficient': nx.average_clustering(G),
        'global_efficiency': nx.global_efficiency(G),
        'betweenness_centrality': np.mean(list(nx.betweenness_centrality(G).values())),
        'degree_centrality': np.mean(list(nx.degree_centrality(G).values()))
    }
    
    return metrics

def compute_time_frequency_features(data, fs=SAMPLING_RATE):
    """
    Compute time-frequency features using continuous wavelet transform.
    
    Parameters:
    -----------
    data : array-like
        Input signal
    fs : int
        Sampling frequency
        
    Returns:
    --------
    dict : Time-frequency features
    """
    # Compute continuous wavelet transform
    widths = np.arange(1, 31)
    cwtmatr = signal.cwt(data, signal.ricker, widths)
    
    # Compute features from the time-frequency representation
    features = {
        'tf_mean': np.mean(np.abs(cwtmatr)),
        'tf_std': np.std(np.abs(cwtmatr)),
        'tf_max': np.max(np.abs(cwtmatr)),
        'tf_entropy': stats.entropy(np.abs(cwtmatr).flatten())
    }
    
    # Compute band-specific energy
    for band_name, (low, high) in FREQ_BANDS.items():
        # Convert frequency bounds to scale indices
        scale_low = int(fs / high)
        scale_high = int(fs / low)
        if scale_high >= len(widths):
            scale_high = len(widths) - 1
        
        # Compute band energy
        band_energy = np.sum(np.abs(cwtmatr[scale_low:scale_high+1, :]))
        features[f'{band_name}_energy'] = band_energy
    
    return features

def extract_features_from_window(window_data):
    """
    Extract all features from a time window of EEG data.
    
    Parameters:
    -----------
    window_data : pd.DataFrame
        EEG data window with channels as columns
        
    Returns:
    --------
    dict : All extracted features
    """
    features = {}
    
    # Process each channel
    for channel in window_data.columns:
        channel_data = window_data[channel].values
        
        # Basic features
        features.update({
            f'{channel}_{k}': v 
            for k, v in compute_statistical_features(channel_data).items()
        })
        
        # Spectral features
        band_powers = extract_band_powers(channel_data)
        features.update({
            f'{channel}_{k}': v 
            for k, v in band_powers.items()
        })
        
        # Non-linear features
        features[f'{channel}_sample_entropy'] = compute_sample_entropy(channel_data)
        features[f'{channel}_lyapunov_exp'] = compute_lyapunov_exp(channel_data)
        
        # Time-frequency features
        tf_features = compute_time_frequency_features(channel_data)
        features.update({
            f'{channel}_{k}': v 
            for k, v in tf_features.items()
        })
    
    # Compute connectivity features between channel pairs
    n_channels = len(window_data.columns)
    connectivity_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            ch1, ch2 = window_data.columns[i], window_data.columns[j]
            
            # Phase connectivity
            phase_conn = compute_phase_connectivity(
                window_data[ch1].values,
                window_data[ch2].values
            )
            features.update({
                f'{ch1}_{ch2}_{k}': v 
                for k, v in phase_conn.items()
            })
            
            # Spectral connectivity
            spec_conn = compute_spectral_connectivity(
                window_data[ch1].values,
                window_data[ch2].values
            )
            features.update({
                f'{ch1}_{ch2}_{k}': v 
                for k, v in spec_conn.items()
            })
            
            # Update connectivity matrix
            connectivity_matrix[i, j] = phase_conn['plv']
            connectivity_matrix[j, i] = phase_conn['plv']
    
    # Compute graph metrics
    graph_metrics = compute_graph_metrics(connectivity_matrix)
    features.update(graph_metrics)
    
    return features

def extract_features_from_trial(trial_data, window_size=FEATURE_EXTRACTION['window_size'],
                              overlap=FEATURE_EXTRACTION['overlap']):
    """
    Extract features from a trial using sliding windows.
    
    Parameters:
    -----------
    trial_data : pd.DataFrame
        EEG data for one trial
    window_size : int
        Size of sliding window in samples
    overlap : float
        Overlap between windows (0-1)
        
    Returns:
    --------
    pd.DataFrame : Features for each window
    """
    step = int(window_size * (1 - overlap))
    windows = []
    
    for start in range(0, len(trial_data) - window_size + 1, step):
        window = trial_data.iloc[start:start + window_size]
        features = extract_features_from_window(window)
        windows.append(features)
    
    return pd.DataFrame(windows)

def extract_all_features(trials):
    """
    Extract features from all trials.
    
    Parameters:
    -----------
    trials : list
        List of trial DataFrames
        
    Returns:
    --------
    pd.DataFrame : Features for all trials
    """
    all_features = []
    
    for trial in trials:
        trial_features = extract_features_from_trial(trial)
        all_features.append(trial_features)
    
    return pd.concat(all_features, ignore_index=True)

# Keep existing helper functions
def extract_band_powers(data, fs=SAMPLING_RATE):
    """Existing band power extraction implementation"""
    freqs, psd = signal.welch(data, fs, nperseg=fs)
    
    powers = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        freq_mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = np.mean(psd[freq_mask])
    
    return powers

def compute_statistical_features(data):
    """Existing statistical features implementation"""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'zero_crossings': len(np.where(np.diff(np.signbit(data)))[0]),
        'peak_to_peak': np.ptp(data),
        'rms': np.sqrt(np.mean(np.square(data)))
    }
