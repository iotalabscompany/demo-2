# data/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ta
import logging
from scipy.stats import linregress

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_features(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Create all features for model training"""
        try:
            # Get main trading timeframe data
            df = data.get(f"BTC_{self.config.TIMEFRAMES[0]}", pd.DataFrame())
            
            if df.empty:
                self.logger.error("No data available for feature engineering")
                raise ValueError("Empty DataFrame received")

            # Add features sequentially with error handling
            df = self.add_price_features(df)
            df = self.add_technical_indicators(df)
            df = self.add_volatility_features(df)
            df = self.add_volume_features(df)
            
            # Prepare final feature set
            feature_names = [col for col in df.columns if col not in ['Target']]
            X = df[feature_names]
            y = df['Target'] if 'Target' in df.columns else None

            return {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }

        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Basic price features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log1p(df['Returns'])
            
            # Price levels
            df['Price_Level'] = df['Close'] / df['Close'].rolling(200).mean()
            
            # Price momentum
            df['Price_Momentum'] = df['Close'] / df['Close'].shift(10) - 1
            
            # Price acceleration
            df['Price_Acceleration'] = df['Returns'].diff()
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'Price_MA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Price_Std_{window}'] = df['Close'].rolling(window=window).std()
            
            return df
        except Exception as e:
            self.logger.error(f"Error adding price features: {str(e)}")
            return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        try:
            # Moving averages
            moving_averages = [5, 10, 20, 50, 100, 200]  # Default if not in config
            for period in moving_averages:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'])
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Lower'] = bollinger.bollinger_lband()
            
            return df
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            # ATR
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Volatility
            returns = df['Returns'].fillna(0)
            for window in [5, 10, 20]:
                df[f'Volatility_{window}'] = returns.rolling(window).std()
            
            # Garman-Klass volatility
            df['GK_Volatility'] = np.sqrt(
                0.5 * np.log(df['High']/df['Low'])**2 -
                (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2
            )
            
            return df
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {str(e)}")
            return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Basic volume features
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Volume moving averages
            for window in [5, 10, 20]:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window).mean()
                df[f'Volume_Std_{window}'] = df['Volume'].rolling(window).std()
            
            # Volume price correlation
            df['Volume_Price_Corr'] = (
                df['Volume'].rolling(20)
                .corr(df['Close'])
                .fillna(0)
            )
            
            # On-balance volume
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error adding volume features: {str(e)}")
            return df

    def prepare_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Prepare target variable"""
        try:
            # Future returns
            df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
            
            # Remove last `horizon` rows since we don't have targets for them
            df = df.iloc[:-horizon]
            
            return df
        except Exception as e:
            self.logger.error(f"Error preparing target: {str(e)}")
            return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       n_std: float = 3) -> pd.DataFrame:
        """Remove outliers from specified columns"""
        for column in columns:
            if column in df.columns:
                mean = df[column].mean()
                std = df[column].std()
                df = df[
                    (df[column] <= mean + n_std * std) & 
                    (df[column] >= mean - n_std * std)
                ]
        return df