# models/model_trainer.py
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
import time
import logging

class ModelTrainer:
    def __init__(self, config, model_ensemble):
        self.config = config
        self.model_ensemble = model_ensemble
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
        
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    feature_names: List[str]) -> Dict:
        """Train all models in the ensemble"""
        self.logger.info("Starting model training...")
        
        try:
            # Convert to numpy arrays if they're pandas objects
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
                
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Store performances for each model
            performances = {}
            fold_predictions = {}
            
            # Train each model
            for name, model in self.model_ensemble.models.items():
                start_time = time.time()
                self.logger.info(f"\nTraining {name} model...")
                
                # Initialize performance metrics
                fold_metrics = []
                model_predictions = []
                
                # Cross-validation
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    # Split data
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_val)
                    model_predictions.extend(y_pred)
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_val, y_pred)
                    fold_metrics.append(metrics)
                    
                    self.logger.info(f"Fold {fold+1} metrics:")
                    for metric_name, value in metrics.items():
                        self.logger.info(f"{metric_name}: {value:.4f}")
                
                # Store fold predictions
                fold_predictions[name] = model_predictions
                
                # Average metrics across folds
                avg_metrics = {}
                for metric in fold_metrics[0].keys():
                    avg_metrics[metric] = np.mean([m[metric] for m in fold_metrics])
                
                # Store performance
                performances[name] = avg_metrics
                
                # Calculate training time
                training_time = time.time() - start_time
                self.logger.info(f"{name} training completed in {training_time:.2f} seconds")
            
            # Create and train ensemble
            self._train_ensemble(X, y, performances)
            
            # Calculate feature importance
            importance = self._calculate_feature_importance(feature_names)
            
            return {
                'performances': performances,
                'feature_importance': importance,
                'best_model': self.model_ensemble.best_model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                       performances: Dict) -> None:
        """Train ensemble model with weighted voting"""
        # Calculate weights based on performance
        total_f1 = sum(perf['f1'] for perf in performances.values())
        weights = [perf['f1']/total_f1 for perf in performances.values()]
        
        # Update ensemble weights
        self.model_ensemble.create_ensemble(weights)
        
        # Train ensemble
        self.model_ensemble.ensemble.fit(X, y)
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance across all models"""
        try:
            importance = {}
            for feature in feature_names:
                importance[feature] = 0.0
                
            # Get feature importance from each model
            for name, model in self.model_ensemble.models.items():
                if hasattr(model, 'feature_importances_'):
                    for idx, importance_value in enumerate(model.feature_importances_):
                        if idx < len(feature_names):
                            importance[feature_names[idx]] += importance_value
            
            # Average importance across models
            n_models = len(self.model_ensemble.models)
            for feature in importance:
                importance[feature] /= n_models
                
            return dict(sorted(importance.items(), 
                             key=lambda x: x[1], 
                             reverse=True))
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {str(e)}")
            return {}
        
    def save_training_history(self, filepath: str):
        """Save training history to file"""
        try:
            pd.DataFrame(self.performance_history).to_csv(filepath, index=False)
            self.logger.info(f"Training history saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving training history: {str(e)}")