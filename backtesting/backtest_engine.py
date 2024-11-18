# backtesting/backtest_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy.stats import norm
import matplotlib.pyplot as plt

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str
    size: float
    pnl: float
    exit_reason: str

class BacktestEngine:
    def __init__(self, config, model_ensemble, risk_manager):
        self.config = config
        self.model_ensemble = model_ensemble
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, data: pd.DataFrame, initial_capital: float) -> Dict:
        """Run full backtest simulation"""
        self.logger.info("Starting backtest simulation...")
        
        # Initialize tracking variables
        capital = initial_capital
        position = None
        self.trades = []
        self.equity_curve = [capital]
        drawdown = 0
        peak_capital = initial_capital
        
        # Add slippage and transaction costs
        transaction_cost = self.config.TRANSACTION_COST
        slippage = self.config.SLIPPAGE
        
        # Run simulation
        for i in range(len(data)):
            current_data = data.iloc[i]
            
            if i < len(data) - 1:
                next_data = data.iloc[i + 1]
            else:
                break
                
            # Get prediction and confidence
            features = current_data[self.get_feature_columns(data)].values.reshape(1, -1)
            prediction, confidence = self.model_ensemble.get_prediction_confidence(features)
            
            # Get market regime
            regime = self.detect_market_regime(data, i)
            
            # Calculate position size
            volatility = current_data['ATR'] / current_data['Close']
            position_size = self.risk_manager.calculate_position_size(
                current_data['Close'],
                volatility,
                regime
            )
            
            # Trading logic
            if position is None:  # No position
                if prediction[0] == 1 and confidence[0] > 0.7:
                    # Calculate entry with slippage
                    entry_price = next_data['Open'] * (1 + slippage)
                    
                    # Calculate stop loss and take profit
                    stop_loss = self.calculate_stop_loss(current_data, regime)
                    take_profit = self.calculate_take_profit(current_data, regime)
                    
                    # Open position
                    position = {
                        'entry_price': entry_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': data.index[i],
                        'direction': 'long'
                    }
                    
                    # Account for transaction costs
                    capital -= position_size * entry_price * transaction_cost
                    
            else:  # Managing existing position
                # Check stop loss
                if next_data['Low'] <= position['stop_loss']:
                    exit_price = position['stop_loss'] * (1 - slippage)
                    trade = self.close_position(position, exit_price, 'stop_loss', 
                                             data.index[i], capital)
                    self.trades.append(trade)
                    capital += trade.pnl - (trade.size * exit_price * transaction_cost)
                    position = None
                    
                # Check take profit
                elif next_data['High'] >= position['take_profit']:
                    exit_price = position['take_profit'] * (1 - slippage)
                    trade = self.close_position(position, exit_price, 'take_profit', 
                                             data.index[i], capital)
                    self.trades.append(trade)
                    capital += trade.pnl - (trade.size * exit_price * transaction_cost)
                    position = None
                    
                # Check for exit signal
                elif prediction[0] == 0 and confidence[0] > 0.7:
                    exit_price = next_data['Open'] * (1 - slippage)
                    trade = self.close_position(position, exit_price, 'signal', 
                                             data.index[i], capital)
                    self.trades.append(trade)
                    capital += trade.pnl - (trade.size * exit_price * transaction_cost)
                    position = None
            
            # Update equity curve and drawdown
            current_capital = capital
            if position is not None:
                position_value = position['size'] * (current_data['Close'] - position['entry_price'])
                current_capital += position_value
                
            self.equity_curve.append(current_capital)
            
            # Update peak capital and drawdown
            peak_capital = max(peak_capital, current_capital)
            drawdown = min(drawdown, (current_capital - peak_capital) / peak_capital)
        
        # Calculate final metrics
        self.results = self.calculate_backtest_metrics(data.index, drawdown)
        return self.results
    
    def close_position(self, position: Dict, exit_price: float, 
                      reason: str, exit_time: datetime, capital: float) -> Trade:
        """Close position and record trade"""
        pnl = (exit_price - position['entry_price']) * position['size']
        if position['direction'] == 'short':
            pnl = -pnl
            
        return Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            direction=position['direction'],
            size=position['size'],
            pnl=pnl,
            exit_reason=reason
        )
    
    def calculate_backtest_metrics(self, dates: pd.DatetimeIndex, 
                                 max_drawdown: float) -> Dict:
        """Calculate comprehensive backtest metrics"""
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate daily returns
        daily_equity = pd.Series(equity_curve, index=dates).resample('D').last()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Calculate trade metrics
        profitable_trades = [t for t in self.trades if t.pnl > 0]
        loss_trades = [t for t in self.trades if t.pnl <= 0]
        
        metrics = {
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'annual_return': self.calculate_annual_return(daily_returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(daily_returns),
            'sortino_ratio': self.calculate_sortino_ratio(daily_returns),
            'max_drawdown': max_drawdown,
            'win_rate': len(profitable_trades) / len(self.trades) if self.trades else 0,
            'profit_factor': (sum(t.pnl for t in profitable_trades) / 
                            abs(sum(t.pnl for t in loss_trades))) if loss_trades else float('inf'),
            'average_trade': np.mean([t.pnl for t in self.trades]) if self.trades else 0,
            'num_trades': len(self.trades),
            'avg_trade_duration': self.calculate_avg_trade_duration(),
            'volatility': np.std(returns) * np.sqrt(252),
            'var_95': self.calculate_var(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
            'equity_curve': equity_curve,
            'daily_returns': daily_returns
        }
        
        return metrics
    
    def monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation of trading strategy"""
        if not self.trades:
            return {}
            
        # Extract trade returns
        trade_returns = [t.pnl/self.config.INITIAL_CAPITAL for t in self.trades]
        
        # Run simulations
        simulated_equity_curves = []
        for _ in range(num_simulations):
            # Randomly sample trades with replacement
            sim_returns = np.random.choice(trade_returns, size=len(trade_returns))
            equity_curve = np.cumprod(1 + sim_returns)
            simulated_equity_curves.append(equity_curve)
        
        # Calculate confidence intervals
        equity_curves = np.array(simulated_equity_curves)
        percentiles = np.percentile(equity_curves, [5, 25, 50, 75, 95], axis=0)
        
        return {
            'median_curve': percentiles[2],
            'confidence_intervals': {
                '95': (percentiles[0], percentiles[4]),
                '50': (percentiles[1], percentiles[3])
            },
            'final_equity_distribution': equity_curves[:, -1],
            'worst_case': np.min(equity_curves[:, -1]),
            'best_case': np.max(equity_curves[:, -1])
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        return np.sqrt(252) * returns.mean() / negative_returns.std()
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in hours"""
        if not self.trades:
            return 0
        durations = [(t.exit_time - t.entry_time).total_seconds()/3600 
                    for t in self.trades]
        return np.mean(durations)
    
    def calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if returns.empty:
            return 0
        total_days = (returns.index[-1] - returns.index[0]).days
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (365 / total_days) - 1
    
    def plot_results(self):
        """Plot backtest results"""
        if not self.results:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot equity curve
        plt.subplot(2, 2, 1)
        plt.plot(self.results['equity_curve'])
        plt.title('Equity Curve')
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 2, 2)
        drawdown = (np.maximum.accumulate(self.results['equity_curve']) - 
                   self.results['equity_curve']) / np.maximum.accumulate(self.results['equity_curve'])
        plt.plot(drawdown)
        plt.title('Drawdown')
        plt.grid(True)
        
        # Plot daily returns distribution
        plt.subplot(2, 2, 3)
        self.results['daily_returns'].hist(bins=50)
        plt.title('Daily Returns Distribution')
        
        # Plot trade PnL distribution
        plt.subplot(2, 2, 4)
        trade_pnls = [t.pnl for t in self.trades]
        plt.hist(trade_pnls, bins=50)
        plt.title('Trade PnL Distribution')
        
        plt.tight_layout()
        plt.show()