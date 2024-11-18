# strategy/signals.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks
import ta

@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    direction: str  # 'long' or 'short'
    confidence: float
    price: float
    signal_type: str
    indicators: Dict[str, float]
    timeframe: str
    strength: float

class SignalGenerator:
    def __init__(self, config):
        self.config = config
        self.signals_history = []
        self.current_signals = {}
        
    def generate_signals(self, data: pd.DataFrame, regime: str) -> List[TradeSignal]:
        """Generate trading signals based on multiple indicators and current regime"""
        signals = []
        
        # Technical signals
        tech_signals = self.generate_technical_signals(data)
        
        # Pattern signals
        pattern_signals = self.generate_pattern_signals(data)
        
        # Volume signals
        volume_signals = self.generate_volume_signals(data)
        
        # Combine and filter signals based on regime
        combined_signals = self.combine_signals(tech_signals, pattern_signals, 
                                             volume_signals, regime)
        
        return combined_signals
    
    def generate_technical_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals from technical indicators"""
        signals = []
        current_price = data['Close'].iloc[-1]
        timestamp = data.index[-1]
        
        # Moving Average Crossovers
        ma_signals = self.check_ma_crossovers(data)
        
        # RSI Signals
        rsi_signals = self.check_rsi_signals(data)
        
        # MACD Signals
        macd_signals = self.check_macd_signals(data)
        
        # Bollinger Band Signals
        bb_signals = self.check_bollinger_signals(data)
        
        # Combine all technical signals
        signals.extend(ma_signals + rsi_signals + macd_signals + bb_signals)
        
        return signals
    
    def generate_pattern_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals from chart patterns"""
        signals = []
        
        # Support/Resistance Levels
        sr_signals = self.check_support_resistance(data)
        
        # Candlestick Patterns
        candle_signals = self.check_candlestick_patterns(data)
        
        # Chart Patterns
        chart_signals = self.check_chart_patterns(data)
        
        # Combine pattern signals
        signals.extend(sr_signals + candle_signals + chart_signals)
        
        return signals
    
    def generate_volume_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals from volume analysis"""
        signals = []
        
        # Volume Profile
        vp_signals = self.analyze_volume_profile(data)
        
        # Volume Breakouts
        vb_signals = self.check_volume_breakouts(data)
        
        # Price-Volume Divergence
        div_signals = self.check_price_volume_divergence(data)
        
        # Combine volume signals
        signals.extend(vp_signals + vb_signals + div_signals)
        
        return signals
    
    def check_ma_crossovers(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Check for moving average crossovers"""
        signals = []
        timestamp = data.index[-1]
        
        # Calculate moving averages
        ma_short = data['Close'].rolling(window=20).mean()
        ma_long = data['Close'].rolling(window=50).mean()
        
        # Check for crossovers
        if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='long',
                    confidence=0.7,
                    price=data['Close'].iloc[-1],
                    signal_type='ma_crossover',
                    indicators={'ma_short': ma_short.iloc[-1], 
                              'ma_long': ma_long.iloc[-1]},
                    timeframe='1h',
                    strength=0.8
                )
            )
        elif ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-2] >= ma_long.iloc[-2]:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='short',
                    confidence=0.7,
                    price=data['Close'].iloc[-1],
                    signal_type='ma_crossover',
                    indicators={'ma_short': ma_short.iloc[-1], 
                              'ma_long': ma_long.iloc[-1]},
                    timeframe='1h',
                    strength=0.8
                )
            )
        
        return signals
    
    def check_rsi_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Check for RSI signals"""
        signals = []
        timestamp = data.index[-1]
        
        # Calculate RSI
        rsi = ta.momentum.rsi(data['Close'])
        
        # Generate signals for oversold/overbought conditions
        if rsi.iloc[-1] < 30:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='long',
                    confidence=0.6,
                    price=data['Close'].iloc[-1],
                    signal_type='rsi_oversold',
                    indicators={'rsi': rsi.iloc[-1]},
                    timeframe='1h',
                    strength=0.6
                )
            )
        elif rsi.iloc[-1] > 70:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='short',
                    confidence=0.6,
                    price=data['Close'].iloc[-1],
                    signal_type='rsi_overbought',
                    indicators={'rsi': rsi.iloc[-1]},
                    timeframe='1h',
                    strength=0.6
                )
            )
        
        return signals
    
    def check_macd_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Check for MACD signals"""
        signals = []
        timestamp = data.index[-1]
        
        # Calculate MACD
        macd = ta.trend.macd_diff(data['Close'])
        
        # Generate signals for MACD crossovers
        if macd.iloc[-1] > 0 and macd.iloc[-2] <= 0:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='long',
                    confidence=0.65,
                    price=data['Close'].iloc[-1],
                    signal_type='macd_crossover',
                    indicators={'macd': macd.iloc[-1]},
                    timeframe='1h',
                    strength=0.7
                )
            )
        elif macd.iloc[-1] < 0 and macd.iloc[-2] >= 0:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='short',
                    confidence=0.65,
                    price=data['Close'].iloc[-1],
                    signal_type='macd_crossover',
                    indicators={'macd': macd.iloc[-1]},
                    timeframe='1h',
                    strength=0.7
                )
            )
        
        return signals
    
    def check_bollinger_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Check for Bollinger Band signals"""
        signals = []
        timestamp = data.index[-1]
        
        # Calculate Bollinger Bands
        bb_ind = ta.volatility.BollingerBands(data['Close'])
        bb_upper = bb_ind.bollinger_hband()
        bb_lower = bb_ind.bollinger_lband()
        
        price = data['Close'].iloc[-1]
        
        # Generate signals for price crossing bands
        if price <= bb_lower.iloc[-1]:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='long',
                    confidence=0.7,
                    price=price,
                    signal_type='bollinger_bounce',
                    indicators={'bb_lower': bb_lower.iloc[-1]},
                    timeframe='1h',
                    strength=0.75
                )
            )
        elif price >= bb_upper.iloc[-1]:
            signals.append(
                TradeSignal(
                    timestamp=timestamp,
                    direction='short',
                    confidence=0.7,
                    price=price,
                    signal_type='bollinger_bounce',
                    indicators={'bb_upper': bb_upper.iloc[-1]},
                    timeframe='1h',
                    strength=0.75
                )
            )
        
        return signals
    
    def combine_signals(self, tech_signals: List[TradeSignal], 
                       pattern_signals: List[TradeSignal],
                       volume_signals: List[TradeSignal],
                       regime: str) -> List[TradeSignal]:
        """Combine and filter signals based on current market regime"""
        all_signals = tech_signals + pattern_signals + volume_signals
        
        # Filter signals based on regime
        if regime == 'trend_following':
            filtered_signals = [s for s in all_signals 
                              if s.signal_type in ['ma_crossover', 'macd_crossover']]
        elif regime == 'mean_reverting':
            filtered_signals = [s for s in all_signals 
                              if s.signal_type in ['bollinger_bounce', 'rsi_oversold', 'rsi_overbought']]
        else:  # ranging
            filtered_signals = [s for s in all_signals 
                              if s.signal_type in ['support_resistance', 'volume_breakout']]
        
        # Sort by confidence and strength
        filtered_signals.sort(key=lambda x: (x.confidence, x.strength), reverse=True)
        
        return filtered_signals
    
    def validate_signal(self, signal: TradeSignal, regime: str) -> bool:
        """Validate signal against current market conditions"""
        # Check if signal aligns with regime
        if regime == 'trend_following' and signal.signal_type not in ['ma_crossover', 'macd_crossover']:
            return False
            
        # Check confidence threshold
        if signal.confidence < 0.6:
            return False
            
        # Check signal strength
        if signal.strength < 0.5:
            return False
            
        return True
    
    def update_signals_history(self, signal: TradeSignal):
        """Update signals history"""
        self.signals_history.append(signal)
        if len(self.signals_history) > 1000:  # Keep last 1000 signals
            self.signals_history.pop(0)
