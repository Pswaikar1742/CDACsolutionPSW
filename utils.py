import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import mne
from mne.viz import plot_topomap
import networkx as nx
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_eeg_channels(data: pd.DataFrame, sampling_rate: int, title: str = 'EEG Signals',
                     save_path: Optional[str] = None) -> None:
    """
    Plot multiple EEG channels with proper scaling and labels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data with channels as columns
    sampling_rate : int
        Sampling rate in Hz
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    n_channels = len(data.columns)
    time = np.arange(len(data)) / sampling_rate
    
    plt.figure(figsize=(15, 2*n_channels))
    for i, channel in enumerate(data.columns):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(time, data[channel], 'b-', linewidth=0.5)
        plt.ylabel(f'{channel} (μV)')
        plt.grid(True)
        
    plt.xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_power_spectrum(data: pd.DataFrame, sampling_rate: int, 
                       save_path: Optional[str] = None) -> None:
    """
    Plot power spectrum for each channel.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    sampling_rate : int
        Sampling rate in Hz
    save_path : str, optional
        Path to save the plot
    """
    n_channels = len(data.columns)
    plt.figure(figsize=(15, 2*n_channels))
    
    for i, channel in enumerate(data.columns):
        plt.subplot(n_channels, 1, i+1)
        freqs, psd = signal.welch(data[channel], sampling_rate, nperseg=sampling_rate)
        plt.semilogy(freqs, psd)
        plt.ylabel(f'{channel} (μV²/Hz)')
        plt.grid(True)
        
    plt.xlabel('Frequency (Hz)')
    plt.suptitle('Power Spectral Density')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_time_frequency(data: pd.DataFrame, sampling_rate: int, 
                       save_path: Optional[str] = None) -> None:
    """
    Plot time-frequency representation using continuous wavelet transform.
    
    Parameters:
    -----------
    data : pd.DataFrame
        EEG data
    sampling_rate : int
        Sampling rate in Hz
    save_path : str, optional
        Path to save the plot
    """
    n_channels = len(data.columns)
    time = np.arange(len(data)) / sampling_rate
    
    plt.figure(figsize=(15, 3*n_channels))
    for i, channel in enumerate(data.columns):
        plt.subplot(n_channels, 1, i+1)
        
        # Compute CWT
        frequencies = np.linspace(1, 50, 50)
        scales = sampling_rate / (2 * frequencies)
        cwt = signal.cwt(data[channel], signal.morlet2, scales)
        
        # Plot
        plt.imshow(np.abs(cwt), aspect='auto', extent=[0, time[-1], 1, 50])
        plt.colorbar(label='Power')
        plt.ylabel(f'{channel}\nFrequency (Hz)')
        
    plt.xlabel('Time (s)')
    plt.suptitle('Time-Frequency Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_connectivity_matrix(connectivity: np.ndarray, channel_names: List[str],
                           title: str = 'Connectivity Matrix', 
                           save_path: Optional[str] = None) -> None:
    """
    Plot connectivity matrix as a heatmap.
    
    Parameters:
    -----------
    connectivity : np.ndarray
        Connectivity matrix
    channel_names : list
        Channel names
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(connectivity, xticklabels=channel_names, yticklabels=channel_names,
                cmap='RdBu_r', center=0, annot=True)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_brain_connectivity(connectivity: np.ndarray, channel_names: List[str],
                          threshold: float = 0.5, save_path: Optional[str] = None) -> None:
    """
    Plot brain connectivity as a network graph.
    
    Parameters:
    -----------
    connectivity : np.ndarray
        Connectivity matrix
    channel_names : list
        Channel names
    threshold : float
        Threshold for showing connections
    save_path : str, optional
        Path to save the plot
    """
    # Create graph
    G = nx.Graph()
    for i in range(len(channel_names)):
        G.add_node(channel_names[i])
        
    # Add edges above threshold
    for i in range(len(channel_names)):
        for j in range(i+1, len(channel_names)):
            if abs(connectivity[i,j]) > threshold:
                G.add_edge(channel_names[i], channel_names[j], 
                          weight=abs(connectivity[i,j]))
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    
    plt.title('Brain Connectivity Network')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_classification_results(y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot classification results including confusion matrix and ROC curves.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        Names of classes
    save_path : str, optional
        Path to save the plot
    """
    # Create subplot figure
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Confusion Matrix', 'ROC Curves'))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    heatmap = go.Heatmap(z=cm, x=class_names, y=class_names,
                         colorscale='RdBu', showscale=True)
    fig.add_trace(heatmap, row=1, col=1)
    
    # ROC Curves
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'{class_name} (AUC = {roc_auc:.2f})'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, width=1000, showlegend=True)
    
    if save_path:
        fig.write_image(save_path)

def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                          title: str = 'Feature Importance',
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    importances : np.ndarray
        Importance scores
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_quantum_circuit_results(quantum_features: np.ndarray, 
                               predictions: np.ndarray,
                               save_path: Optional[str] = None) -> None:
    """
    Visualize quantum circuit results.
    
    Parameters:
    -----------
    quantum_features : np.ndarray
        Quantum feature vectors
    predictions : np.ndarray
        Predicted labels
    save_path : str, optional
        Path to save the plot
    """
    # Create subplot figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Quantum Feature Space', 'Prediction Distribution'))
    
    # Plot quantum features
    fig.add_trace(
        go.Scatter(x=quantum_features[:,0], y=quantum_features[:,1],
                  mode='markers', marker=dict(color=predictions),
                  name='Quantum Features'),
        row=1, col=1
    )
    
    # Plot prediction distribution
    unique_labels, counts = np.unique(predictions, return_counts=True)
    fig.add_trace(
        go.Bar(x=unique_labels, y=counts, name='Predictions'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, width=1000, showlegend=True)
    
    if save_path:
        fig.write_image(save_path)

def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        title: str = 'Learning Curves',
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curves during training.
    
    Parameters:
    -----------
    train_scores : list
        Training scores
    val_scores : list
        Validation scores
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Training Score')
    plt.plot(epochs, val_scores, 'r-', label='Validation Score')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_html_report(results: Dict, save_path: str) -> None:
    """
    Create an HTML report with all results and visualizations.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all results and file paths to visualizations
    save_path : str
        Path to save the HTML report
    """
    html_content = f"""
    <html>
    <head>
        <title>EEG Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; }}
            img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>EEG Analysis Report</h1>
        
        <div class="section">
            <h2>Classification Results</h2>
            <p>Accuracy: {results['accuracy']:.2f}</p>
            <p>Error Count: {results['error_count']}</p>
            <img src="{results['confusion_matrix_plot']}" alt="Confusion Matrix">
        </div>
        
        <div class="section">
            <h2>Feature Analysis</h2>
            <img src="{results['feature_importance_plot']}" alt="Feature Importance">
        </div>
        
        <div class="section">
            <h2>Quantum Results</h2>
            <img src="{results['quantum_results_plot']}" alt="Quantum Results">
        </div>
        
        <div class="section">
            <h2>Learning Progress</h2>
            <img src="{results['learning_curves_plot']}" alt="Learning Curves">
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
