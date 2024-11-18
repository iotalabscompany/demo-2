# backtesting/performance.py
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    def __init__(self, config):
        self.config = config
        
    def analyze_performance(self, results: Dict) -> Dict:
        """Analyze trading performance and generate report"""
        analysis = {}
        
        # Return metrics
        analysis['return_metrics'] = self.analyze_returns(results)
        
        # Risk metrics
        analysis['risk_metrics'] = self.analyze_risk(results)
        
        # Trade metrics
        analysis['trade_metrics'] = self.analyze_trades(results)
        
        return analysis
    
    def analyze_returns(self, results: Dict) -> Dict:
        """Analyze return-based metrics"""
        equity_curve = np.array(results['equity_curve'])
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'annual_return': self.calculate_annual_return(equity_curve),
            'monthly_returns': self.calculate_monthly_returns(equity_curve),
            'daily_returns': self.calculate_daily_returns(equity_curve),
            'best_month': np.max(self.calculate_monthly_returns(equity_curve)),
            'worst_month': np.min(self.calculate_monthly_returns(equity_curve)),
            'avg_monthly_return': np.mean(self.calculate_monthly_returns(equity_curve)),
            'monthly_std': np.std(self.calculate_monthly_returns(equity_curve))
        }
        
        return metrics
    
    def analyze_risk(self, results: Dict) -> Dict:
        """Analyze risk-based metrics"""
        equity_curve = np.array(results['equity_curve'])
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'drawdown_periods': self.analyze_drawdown_periods(equity_curve),
            'var_95': self.calculate_var(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'volatility': np.std(returns) * np.sqrt(252),
            'downside_volatility': self.calculate_downside_volatility(returns)
        }
        
        return metrics
    
    def analyze_trades(self, results: Dict) -> Dict:
        """Analyze trade-based metrics"""
        trades = results['trades']
        
        metrics = {
            'total_trades': len(trades),
            'win_rate': sum(1 for t in trades if t['profit'] > 0) / len(trades),
            'profit_factor': sum(t['profit'] for t in trades if t['profit'] > 0) / abs(sum(t['profit'] for t in trades if t['profit'] < 0)),
            'avg_winner': np.mean([t['profit'] for t in trades if t['profit'] > 0]),
            'avg_loser': np.mean([t['profit'] for t in trades if t['profit'] < 0]),
            'largest_winner': max(t['profit'] for t in trades),
            'largest_loser': min(t['profit'] for t in trades),
            'avg_hold_time': self.calculate_avg_hold_time(trades),
            'trade_pnl_distribution': self.analyze_trade_distribution(trades)
        }
        
        return metrics
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate detailed performance report"""
        report = []
        
        # Overall performance
        report.append("=== PERFORMANCE REPORT ===")
        report.append(f"\nTotal Return: {analysis['return_metrics']['total_return']:.2%}")
        report.append(f"Annual Return: {analysis['return_metrics']['annual_return']:.2%}")
        report.append(f"Sharpe Ratio: {analysis['risk_metrics']['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {analysis['risk_metrics']['max_drawdown']:.2%}")
        
        # Trade statistics
        report.append("\n=== TRADE STATISTICS ===")
        trade_metrics = analysis['trade_metrics']
        report.append(f"Total Trades: {trade_metrics['total_trades']}")
        report.append(f"Win Rate: {trade_metrics['win_rate']:.2%}")
        report.append(f"Profit Factor: {trade_metrics['profit_factor']:.2f}")
        report.append(f"Average Winner: ${trade_metrics['avg_winner']:.2f}")
        report.append(f"Average Loser: ${trade_metrics['avg_loser']:.2f}")
        
        # Risk metrics
        report.append("\n=== RISK METRICS ===")
        risk_metrics = analysis['risk_metrics']
        report.append(f"Volatility: {risk_metrics['volatility']:.2%}")
        report.append(f"VaR (95%): {risk_metrics['var_95']:.2%}")
        report.append(f"CVaR (95%): {risk_metrics['cvar_95']:.2%}")
        
        return "\n".join(report)
    
    def plot_performance(self, results: Dict):
        """Generate performance visualization"""
        plt.style.use('seaborn')
        
        # Create subplot figure
        fig = plt.figure(figsize=(15, 10))
        
        # Plot equity curve
        ax1 = plt.subplot(2, 2, 1)
        self.plot_equity_curve(results['equity_curve'], ax1)
        
        # Plot drawdown
        ax2 = plt.subplot(2, 2, 2)
        self.plot_drawdown(results['equity_curve'], ax2)
        
        # Plot monthly returns heatmap
        ax3 = plt.subplot(2, 2, 3)
        self.plot_monthly_returns_heatmap(results['equity_curve'], ax3)
        
        # Plot trade distribution
        ax4 = plt.subplot(2, 2, 4)
        self.plot_trade_distribution(results['trades'], ax4)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_equity_curve(equity_curve: np.ndarray, ax):
        """Plot equity curve"""
        ax.plot(equity_curve)
        ax.set_title('Equity Curve')
        ax.grid(True)
    
    @staticmethod
    def plot_drawdown(equity_curve: np.ndarray, ax):
        """Plot drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        ax.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.grid(True)
