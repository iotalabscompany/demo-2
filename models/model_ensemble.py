# models/model_ensemble.py
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple
import joblib

class ModelEnsemble:
    def __init__(self, config):
        self.config = config
        self.models = self._initialize_models()
        self.ensemble = None
        self.feature_importance = {}
        
    def _initialize_models(self) -> Dict:
        """Initialize all models in the ensemble"""
        models = {
            'rf': RandomForestClassifier(**self.config.ENSEMBLE_MODELS['rf']),
            'xgb': xgb.XGBClassifier(**self.config.ENSEMBLE_MODELS['xgb']),
            'lgb': lgb.LGBMClassifier(**self.config.ENSEMBLE_MODELS['lgb']),
            'nn': MLPClassifier(**self.config.ENSEMBLE_MODELS['nn']),
            'gb': xgb.XGBClassifier(**self.config.ENSEMBLE_MODELS['gb'])
        }
        return models
    
    def create_ensemble(self, weights: List[float] = None) -> VotingClassifier:
        """Create voting ensemble from base models"""
        if weights is None:
            weights = [1] * len(self.models)
            
        estimators = [(name, model) for name, model in self.models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        return self.ensemble
    
    def update_weights(self, performances: Dict[str, float]) -> List[float]:
        """Update ensemble weights based on model performances"""
        # Normalize performances to get weights
        total_perf = sum(performances.values())
        weights = [performances[model]/total_perf for model in self.models.keys()]
        
        # Update ensemble with new weights
        self.create_ensemble(weights)
        
        return weights
    
    def calculate_feature_importance(self, X, feature_names):
        """Calculate feature importance across all models"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict[name] = dict(zip(feature_names, importances))
            
        # Average importance across models
        avg_importance = {}
        for feature in feature_names:
            importance_values = [
                imp[feature] for imp in importance_dict.values() 
                if feature in imp
            ]
            if importance_values:
                avg_importance[feature] = np.mean(importance_values)
        
        self.feature_importance = avg_importance
        return avg_importance
    
    def get_prediction_confidence(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble predictions with confidence scores"""
        predictions = []
        confidences = []
        
        # Get predictions from each model
        for model in self.models.values():
            pred_proba = model.predict_proba(X)
            predictions.append(np.argmax(pred_proba, axis=1))
            confidences.append(np.max(pred_proba, axis=1))
        
        # Calculate ensemble prediction and confidence
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        ensemble_conf = np.mean(confidences, axis=0)
        
        return ensemble_pred, ensemble_conf
    
    def save_models(self, path: str):
        """Save all models and ensemble"""
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}_model.joblib")
        joblib.dump(self.ensemble, f"{path}/ensemble_model.joblib")
        
    def load_models(self, path: str):
        """Load all models and ensemble"""
        for name in self.models.keys():
            self.models[name] = joblib.load(f"{path}/{name}_model.joblib")
        self.ensemble = joblib.load(f"{path}/ensemble_model.joblib")
