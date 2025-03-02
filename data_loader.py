import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import DATASET_PATH, CHANNELS, PARADIGMS, BASELINE_CONDITIONS

class EEGDataLoader:
    """
    Class to handle loading and organizing EEG data from the dataset.
    """
    
    def __init__(self, dataset_path=DATASET_PATH):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        dataset_path : str
            Path to the dataset directory
        """
        self.dataset_path = dataset_path
        self.channels = CHANNELS
        self.paradigms = PARADIGMS
        self.baseline_conditions = BASELINE_CONDITIONS
    
    def get_subject_path(self, subject_id):
        """
        Get the path to a subject's directory.
        
        Parameters:
        -----------
        subject_id : int
            Subject ID number
        
        Returns:
        --------
        str : Path to subject directory
        """
        return os.path.join(self.dataset_path, f'Subject_{subject_id}')
    
    def load_eeg_file(self, filepath):
        """
        Load an EEG data file.
        
        Parameters:
        -----------
        filepath : str
            Path to the EEG file
        
        Returns:
        --------
        pd.DataFrame : EEG data
        """
        try:
            data = pd.read_csv(filepath)
            # Ensure all channel names are present
            for channel in self.channels:
                if channel not in data.columns:
                    raise ValueError(f"Channel {channel} not found in {filepath}")
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None
    
    def load_markers_file(self, filepath):
        """
        Load a markers file.
        
        Parameters:
        -----------
        filepath : str
            Path to the markers file
        
        Returns:
        --------
        pd.DataFrame : Marker data
        """
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None
    
    def load_subject_data(self, subject_id):
        """
        Load all data for a single subject.
        
        Parameters:
        -----------
        subject_id : int
            Subject ID number
        
        Returns:
        --------
        dict : Dictionary containing all subject data
        """
        subject_path = self.get_subject_path(subject_id)
        subject_data = {
            'baseline': {},
            'paradigms': {}
        }
        
        # Load baseline data
        for condition, info in self.baseline_conditions.items():
            eeg_file = f'Subject_{subject_id}_{info["file_suffix"]}_eeg.csv'
            markers_file = f'Subject_{subject_id}_{info["file_suffix"]}_markers.csv'
            
            eeg_path = os.path.join(subject_path, eeg_file)
            markers_path = os.path.join(subject_path, markers_file)
            
            subject_data['baseline'][condition] = {
                'eeg': self.load_eeg_file(eeg_path),
                'markers': self.load_markers_file(markers_path)
            }
        
        # Load paradigm data
        for paradigm in self.paradigms.keys():
            eeg_file = f'Subject_{subject_id}_{paradigm}_eeg.csv'
            markers_file = f'Subject_{subject_id}_{paradigm}_markers.csv'
            
            eeg_path = os.path.join(subject_path, eeg_file)
            markers_path = os.path.join(subject_path, markers_file)
            
            subject_data['paradigms'][paradigm] = {
                'eeg': self.load_eeg_file(eeg_path),
                'markers': self.load_markers_file(markers_path)
            }
        
        return subject_data
    
    def load_all_subjects(self, subject_range=range(1, 25)):
        """
        Load data for all subjects.
        
        Parameters:
        -----------
        subject_range : range
            Range of subject IDs to load
        
        Returns:
        --------
        dict : Dictionary containing all subjects' data
        """
        all_subjects = {}
        
        for subject_id in tqdm(subject_range, desc="Loading subjects"):
            subject_data = self.load_subject_data(subject_id)
            if subject_data is not None:
                all_subjects[subject_id] = subject_data
        
        return all_subjects
    
    def get_paradigm_data(self, all_subjects, paradigm):
        """
        Extract data for a specific paradigm across all subjects.
        
        Parameters:
        -----------
        all_subjects : dict
            Dictionary containing all subjects' data
        paradigm : str
            Name of the paradigm to extract
        
        Returns:
        --------
        tuple : (X, y) where X is the EEG data and y is the labels
        """
        X = []
        y = []
        
        for subject_id, subject_data in all_subjects.items():
            if paradigm in subject_data['paradigms']:
                paradigm_data = subject_data['paradigms'][paradigm]
                if paradigm_data['eeg'] is not None:
                    X.append(paradigm_data['eeg'])
                    y.extend([paradigm] * len(paradigm_data['eeg']))
        
        return pd.concat(X, ignore_index=True), np.array(y)
    
    def get_all_paradigm_data(self, all_subjects):
        """
        Extract data for all paradigms across all subjects.
        
        Parameters:
        -----------
        all_subjects : dict
            Dictionary containing all subjects' data
        
        Returns:
        --------
        tuple : (X, y) where X is the EEG data and y is the labels
        """
        X = []
        y = []
        
        for paradigm in self.paradigms.keys():
            paradigm_X, paradigm_y = self.get_paradigm_data(all_subjects, paradigm)
            X.append(paradigm_X)
            y.extend(paradigm_y)
        
        return pd.concat(X, ignore_index=True), np.array(y)
    
    def get_baseline_stats(self, all_subjects):
        """
        Calculate baseline statistics across all subjects.
        
        Parameters:
        -----------
        all_subjects : dict
            Dictionary containing all subjects' data
        
        Returns:
        --------
        dict : Baseline statistics for each condition
        """
        baseline_stats = {}
        
        for condition in self.baseline_conditions.keys():
            condition_data = []
            
            for subject_data in all_subjects.values():
                if condition in subject_data['baseline']:
                    baseline_eeg = subject_data['baseline'][condition]['eeg']
                    if baseline_eeg is not None:
                        condition_data.append(baseline_eeg)
            
            if condition_data:
                combined_data = pd.concat(condition_data, ignore_index=True)
                baseline_stats[condition] = {
                    'mean': combined_data.mean(),
                    'std': combined_data.std(),
                    'min': combined_data.min(),
                    'max': combined_data.max()
                }
        
        return baseline_stats
    
    def validate_dataset(self):
        """
        Validate the dataset structure and contents.
        
        Returns:
        --------
        dict : Validation results
        """
        validation = {
            'missing_files': [],
            'corrupt_files': [],
            'incomplete_subjects': []
        }
        
        for subject_id in range(1, 25):
            subject_path = self.get_subject_path(subject_id)
            
            if not os.path.exists(subject_path):
                validation['incomplete_subjects'].append(subject_id)
                continue
            
            # Check baseline files
            for condition, info in self.baseline_conditions.items():
                eeg_file = f'Subject_{subject_id}_{info["file_suffix"]}_eeg.csv'
                markers_file = f'Subject_{subject_id}_{info["file_suffix"]}_markers.csv'
                
                eeg_path = os.path.join(subject_path, eeg_file)
                markers_path = os.path.join(subject_path, markers_file)
                
                if not os.path.exists(eeg_path):
                    validation['missing_files'].append(eeg_path)
                if not os.path.exists(markers_path):
                    validation['missing_files'].append(markers_path)
            
            # Check paradigm files
            for paradigm in self.paradigms.keys():
                eeg_file = f'Subject_{subject_id}_{paradigm}_eeg.csv'
                markers_file = f'Subject_{subject_id}_{paradigm}_markers.csv'
                
                eeg_path = os.path.join(subject_path, eeg_file)
                markers_path = os.path.join(subject_path, markers_file)
                
                if not os.path.exists(eeg_path):
                    validation['missing_files'].append(eeg_path)
                if not os.path.exists(markers_path):
                    validation['missing_files'].append(markers_path)
        
        return validation
