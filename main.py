# main.py
import os
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
import time

from config.config import TradingConfig
from data.data_fetcher import DataFetcher
from data.feature_engineer import FeatureEngineer
from models.model_ensemble import ModelEnsemble
from models.model_trainer import ModelTrainer
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance import PerformanceAnalyzer
from risk.risk_manager import RiskManager
from strategy.market_regime import MarketRegimeDetector
from strategy.signals import SignalGenerator

class TradingSystem:
    def __init__(self):
        self.config = TradingConfig()
        self.data_fetcher = DataFetcher(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_ensemble = ModelEnsemble(self.config)
        self.model_trainer = ModelTrainer(self.config, self.model_ensemble)
        self.risk_manager = RiskManager(self.config)
        self.backtest_engine = BacktestEngine(
            self.config, 
            self.model_ensemble,
            self.risk_manager
        )
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.regime_detector = MarketRegimeDetector(self.config)
        self.signal_generator = SignalGenerator(self.config)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/trading_system_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def train_system(self):
        """Train the entire trading system"""
        self.logger.info("Starting system training...")
        
        try:
            # Fetch historical data
            self.logger.info("Fetching historical data...")
            historical_data = self.data_fetcher.fetch_all_data()
            
            # Engineer features
            self.logger.info("Engineering features...")
            feature_data = self.feature_engineer.create_features(historical_data)
            
            # Train models
            self.logger.info("Training model ensemble...")
            training_results = self.model_trainer.train_models(
                feature_data['X'],
                feature_data['y'],
                feature_data['feature_names']
            )
            
            self.logger.info(f"Training completed. Best model: {training_results['best_model']}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in system training: {str(e)}")
            raise
    
    def run_backtest(self):
        """Run system backtest"""
        self.logger.info("Starting backtest...")
        
        try:
            # Fetch backtest data
            backtest_data = self.data_fetcher.fetch_all_data()
            
            # Run backtest
            backtest_results = self.backtest_engine.run_backtest(
                backtest_data,
                self.config.INITIAL_CAPITAL
            )
            
            # Analyze performance
            performance_analysis = self.performance_analyzer.analyze_performance(backtest_results)
            
            # Generate report
            report = self.performance_analyzer.generate_report(performance_analysis)
            self.logger.info("\nBacktest Report:\n" + report)
            
            return backtest_results, performance_analysis
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            raise
    
    def optimize_strategy(self, backtest_results: Dict):
        """Optimize strategy based on backtest results"""
        self.logger.info("Starting strategy optimization...")
        
        try:
            # Run Monte Carlo simulations
            mc_results = self.backtest_engine.monte_carlo_simulation()
            
            # Update risk parameters
            self.risk_manager.update_risk_parameters(backtest_results)
            
            # Optimize model weights
            self.model_ensemble.optimize_weights(backtest_results)
            
            return {
                'monte_carlo': mc_results,
                'optimized_params': self.risk_manager.get_parameters()
            }
            
        except Exception as e:
            self.logger.error(f"Error in strategy optimization: {str(e)}")
            raise
    
    def run_live_simulation(self, hours: int = 24):
        """Run live trading simulation"""
        self.logger.info(f"Starting {hours}-hour live trading simulation...")
        
        try:
            simulation_results = self.backtest_engine.live_trading_simulation(hours)
            self.logger.info("\nSimulation Results:")
            self.logger.info(f"Final Capital: ${simulation_results['final_capital']:,.2f}")
            self.logger.info(f"Total Trades: {simulation_results['total_trades']}")
            self.logger.info(f"Win Rate: {simulation_results['win_rate']:.2%}")
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Error in live simulation: {str(e)}")
            raise
    
    def save_system_state(self):
        """Save the current state of the system"""
        save_dir = "saved_states"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save models
            self.model_ensemble.save_models(f"{save_dir}/models_{timestamp}")
            
            # Save configuration
            self.config.save(f"{save_dir}/config_{timestamp}.json")
            
            self.logger.info(f"System state saved successfully to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {str(e)}")
            raise
    
    def load_system_state(self, timestamp: str):
        """Load a saved system state"""
        save_dir = "saved_states"
        
        try:
            # Load models
            self.model_ensemble.load_models(f"{save_dir}/models_{timestamp}")
            
            # Load configuration
            self.config.load(f"{save_dir}/config_{timestamp}.json")
            
            self.logger.info(f"System state loaded successfully from {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {str(e)}")
            raise

def main():
    start_time = time.time()
    
    # Initialize trading system
    system = TradingSystem()
    
    print("Starting system training...")
    print(f"Estimated time: {2.5} hours")
    
    try:
        # Train the system
        print("Progress:")
        print("[%-20s] %d%%" % ('='*1, 5))
        training_results = system.train_system()
        
        print("[%-20s] %d%%" % ('='*10, 50))
        # Run backtest
        backtest_results, performance_analysis = system.run_backtest()
        
        print("[%-20s] %d%%" % ('='*15, 75))
        # Optimize strategy
        optimization_results = system.optimize_strategy(backtest_results)
        
        # Run live simulation
        simulation_results = system.run_live_simulation(hours=24)
        
        print("[%-20s] %d%%" % ('='*20, 100))
        
        # Save final state
        system.save_system_state()
        
        elapsed_time = (time.time() - start_time) / 3600  # in hours
        print(f"\nTotal execution time: {elapsed_time:.2f} hours")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()