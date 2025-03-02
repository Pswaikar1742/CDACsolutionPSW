import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from config import MODEL_PARAMS

class AttentionClassifier:
    """
    A class to handle the training and evaluation of the attention classification model.
    """
    
    def __init__(self, random_state=MODEL_PARAMS['random_forest']['random_state']):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=MODEL_PARAMS['random_forest']['n_estimators'],
            max_depth=MODEL_PARAMS['random_forest']['max_depth'],
            min_samples_split=MODEL_PARAMS['random_forest']['min_samples_split'],
            min_samples_leaf=MODEL_PARAMS['random_forest']['min_samples_leaf'],
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, X, y):
        """
        Prepare data for training/testing.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Target labels
        
        Returns:
        --------
        tuple : (X, encoded_y)
        """
        # Remove any constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        X = X.drop(columns=constant_cols)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode labels
        encoded_y = self.label_encoder.fit_transform(y)
        
        return X, encoded_y
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : array-like
            Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict
        
        Returns:
        --------
        array : Predicted labels
        """
        return self.label_encoder.inverse_transform(self.model.predict(X))
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : array-like
            True labels
        
        Returns:
        --------
        dict : Performance metrics
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : array-like
            Labels
        cv : int
            Number of folds
        
        Returns:
        --------
        dict : Cross-validation results
        """
        cv_scores = cross_val_score(self.model, X, y, cv=cv)
        
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def tune_hyperparameters(self, X, y):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : array-like
            Labels
        
        Returns:
        --------
        dict : Best parameters and scores
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        
        Returns:
        --------
        pd.DataFrame : Feature importance scores
        """
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        
        Returns:
        --------
        AttentionClassifier : Loaded model
        """
        model_data = joblib.load(filepath)
        
        classifier = cls()
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        
        return classifier

def train_and_evaluate(X, y, test_size=MODEL_PARAMS['train_test_split']['test_size'],
                      random_state=MODEL_PARAMS['train_test_split']['random_state']):
    """
    Train and evaluate the model in one go.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : array-like
        Labels
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple : (trained_model, evaluation_results)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and train model
    classifier = AttentionClassifier(random_state=random_state)
    X_train, y_train_encoded = classifier.prepare_data(X_train, y_train)
    X_test = X_test.fillna(X_train.mean())
    
    classifier.train(X_train, y_train_encoded)
    
    # Evaluate
    results = classifier.evaluate(X_test, classifier.label_encoder.transform(y_test))
    
    return classifier, results
