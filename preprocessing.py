import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import mne
from mne.preprocessing import ICA
from config import PREPROCESSING, SAMPLING_RATE, CHANNELS

def apply_kalman_filter(data, process_var=PREPROCESSING['kalman_filter']['process_variance'],
                       measurement_var=PREPROCESSING['kalman_filter']['measurement_variance']):
    """
    Apply Kalman filter for noise reduction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    process_var : float
        Process variance
    measurement_var : float
        Measurement variance
    
    Returns:
    --------
    pd.DataFrame : Filtered EEG data
    """
    filtered_data = pd.DataFrame(index=data.index)
    
    for channel in data.columns:
        # Initialize Kalman filter parameters
        x_hat = data[channel].iloc[0]  # Initial state estimate
        p = 1.0  # Initial estimate uncertainty
        
        x_hat_series = []
        
        # Apply Kalman filter
        for measurement in data[channel]:
            # Prediction
            p = p + process_var
            
            # Update
            k = p / (p + measurement_var)  # Kalman gain
            x_hat = x_hat + k * (measurement - x_hat)
            p = (1 - k) * p
            
            x_hat_series.append(x_hat)
        
        filtered_data[channel] = x_hat_series
    
    return filtered_data

def apply_savgol_filter(data, window_length=PREPROCESSING['savgol_filter']['window_length'],
                       polyorder=PREPROCESSING['savgol_filter']['polyorder']):
    """
    Apply Savitzky-Golay filter for smoothing.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    window_length : int
        Window length for filtering
    polyorder : int
        Polynomial order
    
    Returns:
    --------
    pd.DataFrame : Filtered EEG data
    """
    filtered_data = pd.DataFrame(index=data.index)
    
    for channel in data.columns:
        filtered_data[channel] = signal.savgol_filter(
            data[channel],
            window_length=window_length,
            polyorder=polyorder
        )
    
    return filtered_data

def apply_ica(data, n_components=PREPROCESSING['ica']['n_components']):
    """
    Apply ICA for artifact removal.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    n_components : int
        Number of ICA components
    
    Returns:
    --------
    pd.DataFrame : Clean EEG data
    """
    # Convert to MNE object
    raw = create_mne_raw(data)
    
    # Apply ICA
    ica = ICA(n_components=n_components, random_state=PREPROCESSING['ica']['random_state'])
    ica.fit(raw)
    
    # Find artifacts using correlation with EOG channel (if available)
    # Otherwise use automated detection
    eog_indices = []
    for idx, component in enumerate(ica.get_components()):
        if np.max(np.abs(component)) > 0.5:  # Threshold for artifact detection
            eog_indices.append(idx)
    
    # Remove artifacts
    ica.exclude = eog_indices
    clean_raw = raw.copy()
    ica.apply(clean_raw)
    
    # Convert back to DataFrame
    clean_data = pd.DataFrame(
        clean_raw.get_data().T,
        columns=data.columns,
        index=data.index
    )
    
    return clean_data

def compute_signal_quality(data):
    """
    Compute signal quality metrics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    
    Returns:
    --------
    dict : Signal quality metrics
    """
    metrics = {}
    
    for channel in data.columns:
        signal = data[channel].values
        
        # Compute SNR
        signal_power = np.mean(signal ** 2)
        noise = signal - np.convolve(signal, np.ones(10)/10, mode='same')  # Simple denoising
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Line noise at power frequency
        freqs, psd = signal.welch(signal, fs=SAMPLING_RATE)
        line_noise_mask = (freqs >= 49) & (freqs <= 51)  # Around 50 Hz
        line_noise_power = np.mean(psd[line_noise_mask])
        
        metrics[channel] = {
            'snr': snr,
            'line_noise_power': line_noise_power,
            'variance': np.var(signal),
            'kurtosis': pd.Series(signal).kurtosis()
        }
    
    return metrics

def apply_adaptive_threshold(data, window_size=PREPROCESSING['adaptive_threshold']['window_size'],
                           n_std=PREPROCESSING['adaptive_threshold']['n_std']):
    """
    Apply adaptive thresholding for artifact removal.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    window_size : int
        Size of sliding window
    n_std : float
        Number of standard deviations for threshold
    
    Returns:
    --------
    pd.DataFrame : Clean EEG data
    """
    clean_data = data.copy()
    
    for channel in data.columns:
        signal = data[channel].values
        
        # Calculate rolling mean and std
        rolling_mean = pd.Series(signal).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(signal).rolling(window=window_size, center=True).std()
        
        # Define adaptive thresholds
        upper_threshold = rolling_mean + n_std * rolling_std
        lower_threshold = rolling_mean - n_std * rolling_std
        
        # Detect artifacts
        artifacts = (signal > upper_threshold) | (signal < lower_threshold)
        
        # Interpolate artifacts
        clean_signal = signal.copy()
        clean_signal[artifacts] = np.nan
        clean_signal = pd.Series(clean_signal).interpolate(method='cubic')
        
        clean_data[channel] = clean_signal
    
    return clean_data

def compute_cross_channel_correlation(data):
    """
    Compute correlation between channels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    
    Returns:
    --------
    pd.DataFrame : Correlation matrix
    """
    return data.corr()

def preprocess_eeg(eeg_data):
    """
    Complete preprocessing pipeline for EEG data.
    
    Parameters:
    -----------
    eeg_data : pd.DataFrame
        Raw EEG data
    
    Returns:
    --------
    tuple : (preprocessed_data, quality_metrics)
    """
    # 1. Apply bandpass filter
    filtered_data = apply_bandpass_filter(eeg_data)
    
    # 2. Apply notch filter
    filtered_data = apply_notch_filter(filtered_data)
    
    # 3. Apply Kalman filter
    filtered_data = apply_kalman_filter(filtered_data)
    
    # 4. Apply Savitzky-Golay filter
    filtered_data = apply_savgol_filter(filtered_data)
    
    # 5. Apply ICA for artifact removal
    clean_data = apply_ica(filtered_data)
    
    # 6. Apply adaptive thresholding
    clean_data = apply_adaptive_threshold(clean_data)
    
    # 7. Normalize data
    normalized_data = normalize_data(clean_data)
    
    # 8. Compute signal quality metrics
    quality_metrics = compute_signal_quality(normalized_data)
    
    # 9. Compute cross-channel correlation
    correlation_matrix = compute_cross_channel_correlation(normalized_data)
    
    return normalized_data, {
        'quality_metrics': quality_metrics,
        'correlation_matrix': correlation_matrix
    }

# Keep existing helper functions
def apply_bandpass_filter(data, lowcut=PREPROCESSING['bandpass_filter']['lowcut'], 
                         highcut=PREPROCESSING['bandpass_filter']['highcut'], 
                         fs=SAMPLING_RATE):
    """Existing bandpass filter implementation"""
    nyquist = fs / 2
    b, a = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
    
    filtered_data = pd.DataFrame(index=data.index)
    for channel in data.columns:
        filtered_data[channel] = signal.filtfilt(b, a, data[channel])
    
    return filtered_data

def apply_notch_filter(data, freq=PREPROCESSING['notch_filter']['freq'], fs=SAMPLING_RATE):
    """Existing notch filter implementation"""
    b, a = signal.iirnotch(freq, Q=30, fs=fs)
    
    filtered_data = pd.DataFrame(index=data.index)
    for channel in data.columns:
        filtered_data[channel] = signal.filtfilt(b, a, data[channel])
    
    return filtered_data

def normalize_data(data):
    """Existing normalization implementation"""
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    return normalized_data

def create_mne_raw(data, ch_names=CHANNELS, sfreq=SAMPLING_RATE):
    """Existing MNE raw object creation implementation"""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data.T, info)
    
    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw
