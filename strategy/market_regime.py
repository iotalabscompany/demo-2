# strategy/market_regime.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import linregress
from dataclasses import dataclass
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller
import ta

@dataclass
class MarketState:
    regime: str                    # Current market regime
    trend_strength: float          # Measure of trend strength (0-1)
    volatility: float              # Current volatility level
    efficiency_ratio: float        # Market efficiency ratio
    market_phase: str              # Market phase (accumulation/distribution/trend)
    momentum: float               # Current momentum
    support_resistance: Dict       # Key support/resistance levels
    correlation_state: Dict        # Correlations with other assets

class MarketRegimeDetector:
    def __init__(self, config):
        self.config = config
        self.regime_history = []
        self.current_state = None
        
    def detect_regime(self, data: pd.DataFrame) -> MarketState:
        """Detect current market regime using multiple indicators"""
        try:
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(data)
            
            # Calculate volatility state
            volatility = self.calculate_volatility_state(data)
            
            # Calculate market efficiency
            efficiency = self.calculate_market_efficiency(data)
            
            # Determine market phase
            market_phase = self.determine_market_phase(data)
            
            # Calculate momentum
            momentum = self.calculate_momentum(data)
            
            # Find support/resistance levels
            sr_levels = self.find_support_resistance(data)
            
            # Calculate correlations
            correlations = self.calculate_correlations(data)
            
            # Determine overall regime
            regime = self.determine_regime(
                trend_strength,
                volatility,
                efficiency,
                market_phase
            )
            
            # Create market state
            state = MarketState(
                regime=regime,
                trend_strength=trend_strength,
                volatility=volatility,
                efficiency_ratio=efficiency,
                market_phase=market_phase,
                momentum=momentum,
                support_resistance=sr_levels,
                correlation_state=correlations
            )
            
            self.current_state = state
            self.regime_history.append(state)
            
            return state
            
        except Exception as e:
            print(f"Error detecting market regime: {str(e)}")
            return self.current_state
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        # ADX for trend strength
        adx = ta.trend.adx(data['High'], data['Low'], data['Close'])
        
        # Moving average alignment
        ma_short = data['Close'].rolling(window=20).mean()
        ma_medium = data['Close'].rolling(window=50).mean()
        ma_long = data['Close'].rolling(window=200).mean()
        
        ma_alignment = (
            (ma_short > ma_medium).astype(int) + 
            (ma_medium > ma_long).astype(int)
        ) / 2
        
        # Linear regression slope
        prices = data['Close'].values
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = linregress(x, prices)
        r_squared = r_value ** 2
        
        # Combine indicators
        trend_strength = (
            0.4 * (adx.iloc[-1] / 100) +
            0.3 * ma_alignment.iloc[-1] +
            0.3 * r_squared
        )
        
        return min(1.0, max(0.0, trend_strength))
    
    def calculate_volatility_state(self, data: pd.DataFrame) -> float:
        """Calculate volatility state"""
        # Calculate ATR-based volatility
        atr = ta.volatility.average_true_range(
            data['High'], 
            data['Low'], 
            data['Close']
        )
        
        # Normalize ATR
        atr_normalized = atr / data['Close']
        
        # Calculate historical volatility
        returns = np.log(data['Close'] / data['Close'].shift(1))
        hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Combine volatility measures
        if not atr_normalized.empty and not hist_vol.empty:
            volatility = (
                0.5 * atr_normalized.iloc[-1] +
                0.5 * hist_vol.iloc[-1]
            )
            return volatility
        return 0.0
    
    def calculate_market_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate market efficiency ratio"""
        if len(data) < 2:
            return 0.0
        
        price_change = abs(data['Close'].iloc[-1] - data['Close'].iloc[0])
        path_length = np.sum(abs(data['Close'].diff().dropna()))
        
        if path_length == 0:
            return 0.0
            
        efficiency = price_change / path_length
        return efficiency
    
    def determine_market_phase(self, data: pd.DataFrame) -> str:
        """Determine market phase"""
        if len(data) < 20:
            return 'trend'
            
        # Volume analysis
        volume_sma = data['Volume'].rolling(window=20).mean()
        price_sma = data['Close'].rolling(window=20).mean()
        
        # Recent price action
        recent_high = data['High'].rolling(window=20).max()
        recent_low = data['Low'].rolling(window=20).min()
        
        # Check for accumulation
        if (data['Volume'].iloc[-1] > volume_sma.iloc[-1] and
            data['Close'].iloc[-1] > price_sma.iloc[-1] and
            data['Close'].iloc[-1] <= recent_low.iloc[-1]):
            return 'accumulation'
            
        # Check for distribution
        elif (data['Volume'].iloc[-1] > volume_sma.iloc[-1] and
              data['Close'].iloc[-1] < price_sma.iloc[-1] and
              data['Close'].iloc[-1] >= recent_high.iloc[-1]):
            return 'distribution'
            
        # Default to trend
        else:
            return 'trend'
    
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate market momentum"""
        if len(data) < 14:
            return 0.0
            
        # RSI
        rsi = ta.momentum.rsi(data['Close'])
        
        # MACD
        macd = ta.trend.macd_diff(data['Close'])
        
        # Rate of change
        roc = data['Close'].pct_change(periods=14)
        
        if rsi.empty or macd.empty or roc.empty:
            return 0.0
            
        # Combine momentum indicators
        momentum = (
            0.4 * (rsi.iloc[-1] / 100) +
            0.3 * (macd.iloc[-1] / data['Close'].iloc[-1]) +
            0.3 * roc.iloc[-1]
        )
        
        return momentum
    
    def find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        if len(data) < 20:
            return {'resistance': [], 'support': []}
            
        # Find peaks and troughs
        peaks, _ = find_peaks(data['High'].values)
        troughs, _ = find_peaks(-data['Low'].values)
        
        # Get price levels
        resistance_levels = data['High'].iloc[peaks].values if len(peaks) > 0 else []
        support_levels = data['Low'].iloc[troughs].values if len(troughs) > 0 else []
        
        # Cluster levels
        resistance_clusters = self.cluster_levels(resistance_levels)
        support_clusters = self.cluster_levels(support_levels)
        
        return {
            'resistance': resistance_clusters,
            'support': support_clusters
        }
    
    def calculate_correlations(self, data: pd.DataFrame) -> Dict:
        """Calculate correlations with other assets"""
        correlations = {}
        
        # Calculate correlation with market indices
        if 'SPY' in data.columns:
            correlations['spy'] = data['Close'].corr(data['SPY'])
            
        # Calculate correlation with other crypto
        if 'ETH' in data.columns:
            correlations['eth'] = data['Close'].corr(data['ETH'])
            
        return correlations
    
    def determine_regime(self, trend_strength: float, volatility: float,
                        efficiency: float, phase: str) -> str:
        """Determine overall market regime"""
        # Check for strong trend
        if trend_strength > 0.7 and efficiency > 0.6:
            return 'trend_following'
            
        # Check for mean reversion
        elif trend_strength < 0.3 and volatility > 0.02:
            return 'mean_reverting'
            
        # Check for ranging market
        elif 0.3 <= trend_strength <= 0.7 and efficiency < 0.5:
            return 'ranging'
            
        # Default to trend following
        else:
            return 'trend_following'
    
    @staticmethod
    def cluster_levels(levels: np.ndarray, tolerance: float = 0.02) -> List[float]:
        """Cluster nearby price levels"""
        if len(levels) == 0:
            return []
            
        # Sort levels
        sorted_levels = np.sort(levels)
        clusters = [[sorted_levels[0]]]
        
        # Cluster nearby levels
        for level in sorted_levels[1:]:
            if level - clusters[-1][-1] <= tolerance * level:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        
        # Return average of each cluster
        return [np.mean(cluster) for cluster in clusters]
    
    def get_regime_summary(self) -> Dict:
        """Get summary of current market regime"""
        if not self.current_state:
            return {}
            
        return {
            'regime': self.current_state.regime,
            'trend_strength': self.current_state.trend_strength,
            'volatility': self.current_state.volatility,
            'market_phase': self.current_state.market_phase,
            'momentum': self.current_state.momentum
        }