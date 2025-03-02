import argparse
import os
import json
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import EEGDataLoader
from preprocessing import preprocess_eeg
from feature_extraction import extract_all_features
from model import AttentionClassifier, train_and_evaluate
import utils
from config import MODEL_PARAMS

def setup_logging(output_dir):
    """Set up logging configuration."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_results(results, output_dir):
    """Save analysis results."""
    results_file = os.path.join(output_dir, 'results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)

def run_analysis(args):
    """Run the complete EEG analysis pipeline."""
    # Set up logging
    setup_logging(args.output_dir)
    logging.info("Starting EEG analysis pipeline")
    
    try:
        # Load data
        logging.info("Loading dataset...")
        loader = EEGDataLoader()
        if args.validate:
            validation_results = loader.validate_dataset()
            if validation_results['missing_files'] or validation_results['corrupt_files']:
                logging.error("Dataset validation failed")
                return validation_results
        
        all_subjects = loader.load_all_subjects()
        logging.info(f"Loaded data for {len(all_subjects)} subjects")
        
        # Get all paradigm data
        X, y = loader.get_all_paradigm_data(all_subjects)
        logging.info(f"Total samples: {len(X)}")
        
        # Preprocess data
        logging.info("Preprocessing EEG data...")
        X_preprocessed = preprocess_eeg(X)
        
        # Extract features
        logging.info("Extracting features...")
        X_features = extract_all_features(X_preprocessed)
        logging.info(f"Extracted {X_features.shape[1]} features")
        
        # Train and evaluate model
        logging.info("Training and evaluating model...")
        classifier, results = train_and_evaluate(
            X_features, y,
            test_size=MODEL_PARAMS['train_test_split']['test_size'],
            random_state=MODEL_PARAMS['train_test_split']['random_state']
        )
        
        # Get feature importance
        feature_importance = classifier.get_feature_importance(X_features.columns)
        
        # Save model if requested
        if args.save_model:
            model_path = os.path.join(args.output_dir, 'model.joblib')
            classifier.save_model(model_path)
            logging.info(f"Model saved to {model_path}")
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        utils.plot_confusion_matrix_heatmap(
            results['confusion_matrix'],
            list(classifier.label_encoder.classes_)
        )
        plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        utils.plot_feature_importance(classifier, X_features.columns)
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
        plt.close()
        
        # Save results
        results['feature_importance'] = feature_importance.to_dict()
        save_results(results, args.output_dir)
        
        logging.info("Analysis completed successfully")
        return results
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='EEG Attention Classification Analysis')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save results and artifacts')
    parser.add_argument('--validate', action='store_true',
                        help='Validate dataset before running analysis')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model to disk')
    
    args = parser.parse_args()
    
    try:
        results = run_analysis(args)
        if results:
            print("\nAnalysis Results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print("\nClassification Report:")
            print(results['classification_report'])
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
