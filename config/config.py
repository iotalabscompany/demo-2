# config/config.py
import json
from dataclasses import dataclass, asdict
from typing import Dict, List

@dataclass
class TradingConfig:
    # Data parameters
    TICKER: str = "BTC-USD"
    CORRELATED_ASSETS: list = None
    TIMEFRAMES: list = None
    HISTORY_YEARS: int = 2
    
    # Technical Analysis Parameters
    MOVING_AVERAGES: list = None
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: int = 2
    ATR_PERIOD: int = 14
    VOLUME_MA_PERIOD: int = 20
    
    # Model parameters
    ENSEMBLE_MODELS: Dict = None
    VALIDATION_SPLIT: float = 0.2
    TRAINING_EPOCHS: int = 100
    
    # Risk parameters
    INITIAL_CAPITAL: float = 100000.0
    RISK_PER_TRADE: float = 0.02
    MAX_DRAWDOWN: float = 0.10
    POSITION_SIZING: Dict = None
    
    # Trading parameters
    TARGET_DAILY_RETURN: float = 0.04
    SLIPPAGE: float = 0.001
    TRANSACTION_COST: float = 0.001
    
    # Backtesting parameters
    BACKTEST_WINDOWS: list = None
    MONTE_CARLO_SIMS: int = 1000
    
    def __post_init__(self):
        # Initialize default values
        if self.CORRELATED_ASSETS is None:
            self.CORRELATED_ASSETS = ["ETH-USD", "^GSPC"]
            
        if self.TIMEFRAMES is None:
            self.TIMEFRAMES = ["1h", "4h", "1d"]
            
        if self.MOVING_AVERAGES is None:
            self.MOVING_AVERAGES = [5, 10, 20, 50, 100, 200]
            
        if self.ENSEMBLE_MODELS is None:
            self.ENSEMBLE_MODELS = {
                'rf': {
                    'n_estimators': 500,
                    'max_depth': 20,
                    'random_state': 42
                },
                'xgb': {
                    'n_estimators': 500,
                    'learning_rate': 0.01,
                    'max_depth': 10,
                    'random_state': 42
                },
                'lgb': {
                    'n_estimators': 500,
                    'num_leaves': 31,
                    'random_state': 42
                },
                'nn': {
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 1000,
                    'random_state': 42
                },
                'gb': {
                    'n_estimators': 500,
                    'learning_rate': 0.01,
                    'random_state': 42
                }
            }
            
        if self.POSITION_SIZING is None:
            self.POSITION_SIZING = {
                'trend_following': 1.2,
                'mean_reverting': 0.8,
                'ranging': 0.6
            }
            
        if self.BACKTEST_WINDOWS is None:
            self.BACKTEST_WINDOWS = [30, 90, 180]

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    def load(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
    
    def get_model_params(self, model_name: str) -> Dict:
        """Get parameters for specific model"""
        return self.ENSEMBLE_MODELS.get(model_name, {})
    
    def get_position_sizing(self, regime: str) -> float:
        """Get position sizing multiplier for given regime"""
        return self.POSITION_SIZING.get(regime, 1.0)
    
    def get_indicator_params(self) -> Dict:
        """Get technical indicator parameters"""
        return {
            'moving_averages': self.MOVING_AVERAGES,
            'rsi_period': self.RSI_PERIOD,
            'macd_fast': self.MACD_FAST,
            'macd_slow': self.MACD_SLOW,
            'macd_signal': self.MACD_SIGNAL,
            'bollinger_period': self.BOLLINGER_PERIOD,
            'bollinger_std': self.BOLLINGER_STD,
            'atr_period': self.ATR_PERIOD,
            'volume_ma_period': self.VOLUME_MA_PERIOD
        }