# data/data_fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import ccxt
import yfinance as yf
import ta
import logging
import time
import requests

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchanges = self._initialize_exchanges()
        self.cached_data = {}
        
    def _initialize_exchanges(self):
        """Initialize multiple exchanges for redundancy"""
        exchanges = {}
        try:
            exchanges['binance'] = ccxt.binance()
        except:
            self.logger.warning("Couldn't initialize Binance")
        try:
            exchanges['kraken'] = ccxt.kraken()
        except:
            self.logger.warning("Couldn't initialize Kraken")
        try:
            exchanges['coinbase'] = ccxt.coinbasepro()
        except:
            self.logger.warning("Couldn't initialize Coinbase")
        return exchanges

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all assets and timeframes"""
        self.logger.info("Fetching all market data...")
        data = {}
        
        # Try to fetch BTC data from multiple sources
        for timeframe in self.config.TIMEFRAMES:
            try:
                df = self.fetch_crypto_data(self.config.TICKER, timeframe)
                if not df.empty:
                    data[f"BTC_{timeframe}"] = df
                else:
                    # If no data, create sample data
                    data[f"BTC_{timeframe}"] = self.create_sample_data(timeframe)
            except Exception as e:
                self.logger.error(f"Error fetching {timeframe} data: {str(e)}")
                data[f"BTC_{timeframe}"] = self.create_sample_data(timeframe)

        # Add sample data for other assets if needed
        for asset in self.config.CORRELATED_ASSETS:
            try:
                df = self.fetch_crypto_data(asset, "1d")
                if not df.empty:
                    data[asset] = df
                else:
                    data[asset] = self.create_sample_data("1d")
            except:
                data[asset] = self.create_sample_data("1d")

        # Add sentiment data
        data["sentiment"] = self.create_sample_sentiment_data()
        
        return data

    def create_sample_data(self, timeframe: str) -> pd.DataFrame:
        """Create sample price data for testing"""
        self.logger.info(f"Creating sample data for {timeframe}")
        
        # Define number of periods based on timeframe
        periods_map = {
            "1h": 24 * 365 * 2,  # 2 years of hourly data
            "4h": 6 * 365 * 2,   # 2 years of 4-hour data
            "1d": 365 * 2        # 2 years of daily data
        }
        periods = periods_map.get(timeframe, 1000)
        
        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=periods, freq=timeframe)
        
        # Generate price data
        np.random.seed(42)  # For reproducibility
        price = 30000  # Starting price
        prices = [price]
        
        for _ in range(periods-1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.01, size=len(prices)),
            'High': prices * np.random.uniform(1.01, 1.03, size=len(prices)),
            'Low': prices * np.random.uniform(0.97, 0.99, size=len(prices)),
            'Close': prices,
            'Volume': np.random.uniform(1000, 5000, size=len(prices)),
            'Adj Close': prices
        }, index=dates)
        
        return df

    def create_sample_sentiment_data(self) -> pd.DataFrame:
        """Create sample sentiment data"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=365*self.config.HISTORY_YEARS,
            freq='D'
        )
        
        sentiment_data = pd.DataFrame({
            'social_sentiment': np.random.normal(0, 1, len(dates)),
            'news_sentiment': np.random.normal(0, 1, len(dates)),
            'market_fear': np.random.normal(50, 15, len(dates))
        }, index=dates)
        
        return sentiment_data

    def fetch_crypto_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Attempt to fetch data from multiple sources"""
        for source in ['yfinance', 'exchange', 'sample']:
            try:
                if source == 'yfinance':
                    df = self._fetch_from_yfinance(symbol, timeframe)
                elif source == 'exchange':
                    df = self._fetch_from_exchanges(symbol, timeframe)
                else:
                    df = self.create_sample_data(timeframe)
                
                if not df.empty:
                    return df
            except Exception as e:
                self.logger.error(f"Error fetching from {source}: {str(e)}")
                continue
        
        return self.create_sample_data(timeframe)

    def _fetch_from_yfinance(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from yfinance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*self.config.HISTORY_YEARS)
        
        # Convert timeframe to yfinance format
        timeframe_map = {
            "1h": "1h",
            "4h": "1h",  # yfinance doesn't support 4h
            "1d": "1d"
        }
        yf_timeframe = timeframe_map.get(timeframe, "1h")
        
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=yf_timeframe
        )
        
        # Resample to 4h if needed
        if timeframe == "4h":
            df = df.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
        
        return df

    def _fetch_from_exchanges(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Try fetching from multiple exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    int((datetime.now() - timedelta(days=365*self.config.HISTORY_YEARS)).timestamp() * 1000)
                )
                
                if ohlcv:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                self.logger.error(f"Error fetching from {exchange_name}: {str(e)}")
                continue
        
        return pd.DataFrame()