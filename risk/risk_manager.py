# risk/risk_manager.py
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    regime: str
    risk_amount: float

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.position_history = []
        self.risk_metrics = {}
        self.current_exposure = 0
        self.current_drawdown = 0
        self.peak_capital = self.config.INITIAL_CAPITAL
        
    def calculate_position_size(self, price: float, volatility: float, 
                              regime: str) -> float:
        """Calculate optimal position size based on multiple factors"""
        # Get base position size from Kelly Criterion
        kelly_fraction = self.calculate_kelly_fraction()
        base_size = self.config.INITIAL_CAPITAL * kelly_fraction
        
        # Adjust for volatility
        vol_adjustment = self.calculate_volatility_adjustment(volatility)
        
        # Adjust for market regime
        regime_adjustment = self.config.POSITION_SIZING.get(regime, 1.0)
        
        # Adjust for current exposure
        exposure_adjustment = self.calculate_exposure_adjustment()
        
        # Calculate final position size
        position_size = (base_size * vol_adjustment * 
                        regime_adjustment * exposure_adjustment)
        
        # Apply maximum position size constraint
        max_position = self.calculate_max_position(price)
        position_size = min(position_size, max_position)
        
        return position_size
    
    def calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction"""
        if not self.position_history:
            return self.config.RISK_PER_TRADE
            
        # Calculate win rate and profit ratios
        wins = [p for p in self.position_history if p.risk_amount > 0]
        if not wins:
            return self.config.RISK_PER_TRADE / 2  # Be conservative
            
        win_rate = len(wins) / len(self.position_history)
        avg_win = np.mean([p.risk_amount for p in wins])
        avg_loss = abs(np.mean([p.risk_amount for p in self.position_history 
                               if p.risk_amount <= 0]))
        
        # Kelly fraction
        kelly = win_rate - ((1 - win_rate) / (avg_win/avg_loss))
        
        # Use half-Kelly for more conservative sizing
        kelly = max(0, min(kelly * 0.5, self.config.RISK_PER_TRADE))
        
        return kelly
    
    def calculate_volatility_adjustment(self, current_volatility: float) -> float:
        """Adjust position size based on volatility"""
        if not self.position_history:
            return 1.0
            
        # Calculate historical volatility
        hist_vol = np.std([p.risk_amount for p in self.position_history])
        
        # Adjust based on current vs historical volatility
        vol_ratio = hist_vol / current_volatility if current_volatility > 0 else 1
        
        # Normalize adjustment
        adjustment = np.clip(vol_ratio, 0.5, 2.0)
        
        return adjustment
    
    def calculate_exposure_adjustment(self) -> float:
        """Adjust position size based on current market exposure"""
        max_exposure = self.config.RISK_PER_TRADE * 3  # Maximum 3x base risk
        
        if self.current_exposure >= max_exposure:
            return 0.0  # No new positions
        
        # Linear scaling based on current exposure
        return 1.0 - (self.current_exposure / max_exposure)
    
    def calculate_max_position(self, price: float) -> float:
        """Calculate maximum allowed position size"""
        # Account for current drawdown
        drawdown_factor = 1.0 - (self.current_drawdown / self.config.MAX_DRAWDOWN)
        drawdown_factor = max(0.2, drawdown_factor)  # Minimum 20% of normal size
        
        # Calculate maximum position
        max_position = (self.config.INITIAL_CAPITAL * 
                       self.config.RISK_PER_TRADE * drawdown_factor)
        
        return max_position
    
    def calculate_stop_loss(self, price: float, volatility: float, 
                          regime: str) -> float:
        """Calculate adaptive stop loss levels"""
        # Base stop loss percentage
        base_stop = self.config.RISK_PER_TRADE
        
        # Adjust for volatility
        vol_stop = volatility * 2  # 2 standard deviations
        
        # Adjust for regime
        regime_stops = {
            'trend_following': 1.5,
            'mean_reverting': 0.8,
            'ranging': 1.0
        }
        regime_factor = regime_stops.get(regime, 1.0)
        
        # Calculate final stop loss
        stop_loss_pct = base_stop * regime_factor
        stop_loss_pct = max(stop_loss_pct, vol_stop)  # Use larger of the two
        
        return price * (1 - stop_loss_pct)
    
    def calculate_take_profit(self, price: float, stop_loss: float) -> float:
        """Calculate take profit level based on risk-reward ratio"""
        risk = price - stop_loss
        reward = risk * 2  # 2:1 reward-risk ratio
        
        return price + reward
    
    def update_position(self, position: Position):
        """Update position tracking"""
        self.position_history.append(position)
        self.current_exposure += position.size
        
        # Update drawdown
        current_capital = self.calculate_current_capital()
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
    def close_position(self, position: Position, exit_price: float):
        """Handle position closing"""
        self.current_exposure -= position.size
        
        # Calculate and store risk metrics
        risk_amount = (exit_price - position.entry_price) * position.size
        position.risk_amount = risk_amount
        
        self.update_risk_metrics(position)
    
    def update_risk_metrics(self, position: Position):
        """Update risk metrics after position close"""
        self.risk_metrics = {
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.current_drawdown
        }
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from position history"""
        if not self.position_history:
            return 0.0
        
        wins = sum(1 for p in self.position_history if p.risk_amount > 0)
        return wins / len(self.position_history)
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        profits = sum(p.risk_amount for p in self.position_history if p.risk_amount > 0)
        losses = abs(sum(p.risk_amount for p in self.position_history if p.risk_amount < 0))
        
        return profits / losses if losses != 0 else float('inf')
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns"""
        if not self.position_history:
            return 0.0
            
        returns = [p.risk_amount for p in self.position_history]
        if not returns:
            return 0.0
            
        return np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0.0
    
    def calculate_current_capital(self) -> float:
        """Calculate current capital including open positions"""
        return (self.config.INITIAL_CAPITAL + 
                sum(p.risk_amount for p in self.position_history))
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'current_exposure': self.current_exposure,
            'current_drawdown': self.current_drawdown,
            'peak_capital': self.peak_capital,
            'risk_metrics': self.risk_metrics,
            'position_count': len(self.position_history),
            'kelly_fraction': self.calculate_kelly_fraction(),
            'current_capital': self.calculate_current_capital()
        }
