"""
Yahoo Finance Data Collector
Collects market data including commodities, indices, and FX rates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_collector import BaseCollector


class YahooCollector(BaseCollector):
    """
    Collector for Yahoo Finance data.
    
    Handles:
    - Currency pairs (USD/BRL backup)
    - Commodity futures (oil, soybeans, sugar, coffee)
    - Stock indices (S&P 500, DXY, VIX)
    - Mining stocks as iron ore proxy (VALE, BHP)
    - Emerging market indicators
    """
    
    def _initialize(self) -> None:
        """Initialize Yahoo Finance specific settings."""
        
        # Get tickers from config
        yahoo_config = self.config.get('apis', {}).get('yahoo', {})
        
        self.tickers = yahoo_config.get('tickers', {
            # Tier 1 - Critical indicators
            'dxy_index': 'DX-Y.NYB',         # US Dollar Index
            'vix': '^VIX',                   # Volatility Index
            'usd_brl_yahoo': 'USDBRL=X',     # USD/BRL rate (backup)
            
            # Tier 2 - Commodities
            'brent': 'BZ=F',                 # Brent Oil Futures
            'wti': 'CL=F',                   # WTI Crude Oil
            'soybeans': 'ZS=F',              # Soybean Futures
            'corn': 'ZC=F',                  # Corn Futures
            'sugar': 'SB=F',                 # Sugar #11 Futures
            'coffee': 'KC=F',                # Coffee Futures
            'wheat': 'ZW=F',                 # Wheat Futures
            
            # Precious metals
            'gold': 'GC=F',                  # Gold Futures
            'silver': 'SI=F',                # Silver Futures
            'copper': 'HG=F',                # Copper Futures
            
            # Mining stocks (iron ore proxy)
            'vale': 'VALE',                  # Vale SA
            'bhp': 'BHP',                    # BHP Group
            'rio': 'RIO',                    # Rio Tinto
            
            # Equity indices
            'sp500': '^GSPC',                # S&P 500
            'nasdaq': '^IXIC',               # NASDAQ
            'dow': '^DJI',                   # Dow Jones
            'russell2000': '^RUT',           # Russell 2000
            
            # International indices
            'bovespa': '^BVSP',              # Brazilian Bovespa
            'emerging_markets': 'EEM',       # iShares Emerging Markets ETF
            'china_etf': 'FXI',              # iShares China ETF
            'europe_etf': 'VGK',             # Vanguard Europe ETF
            
            # Other currencies (for correlation)
            'eur_usd': 'EURUSD=X',           # EUR/USD
            'gbp_usd': 'GBPUSD=X',           # GBP/USD
            'usd_jpy': 'USDJPY=X',           # USD/JPY
            'usd_mxn': 'USDMXN=X',           # USD/MXN
            'usd_ars': 'USDARS=X',           # USD/ARS
            'usd_clp': 'USDCLP=X',           # USD/CLP
            
            # Bond yields
            'us_10y': '^TNX',                # 10-Year Treasury Yield
            'us_2y': '^IRX',                 # 13-Week Treasury Yield (proxy for short-term)
            
            # Crypto (risk sentiment)
            'bitcoin': 'BTC-USD',            # Bitcoin
            'ethereum': 'ETH-USD',           # Ethereum
        })
        
        self.timeout = yahoo_config.get('timeout', 30)
        
        # Rate limiting
        self.requests_per_second = 2
        self.last_request_time = 0
        
        # Parallel processing
        self.max_workers = 5
        
    def collect(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect Yahoo Finance data for specified period.
        
        Args:
            start_date: Start date for collection
            end_date: End date for collection
            tickers: List of tickers to collect (None = all)
            
        Returns:
            DataFrame with collected market data
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache
        cache_params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'tickers': tickers or 'all'
        }
        cache_key = self.get_cache_key(cache_params)
        
        cached_data = self.load_from_cache(cache_key, ttl_hours=1)
        if cached_data is not None:
            self.logger.info("Using cached Yahoo Finance data")
            return cached_data
        
        # Determine tickers to collect
        if tickers is None:
            tickers_to_collect = self.tickers
        else:
            tickers_to_collect = {k: v for k, v in self.tickers.items() if k in tickers}
        
        self.logger.info(f"Collecting Yahoo Finance data for {len(tickers_to_collect)} tickers")
        
        # Collect data in parallel
        all_data = self._parallel_collect(tickers_to_collect, start_date, end_date)
        
        # Process and merge data
        if all_data:
            merged_data = self._merge_ticker_data(all_data)
            
            # Standardize
            merged_data = self.standardize_dataframe(merged_data)
            
            # Calculate derived metrics
            merged_data = self._calculate_derived_metrics(merged_data)
            
            # Validate
            if self.validate_data(merged_data):
                self.save_to_cache(cache_key, merged_data)
            else:
                self.logger.warning("Data validation failed for Yahoo Finance data")
            
            return merged_data
        
        return pd.DataFrame()
    
    def _parallel_collect(
        self,
        tickers_dict: Dict[str, str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple tickers in parallel.
        
        Args:
            tickers_dict: Dictionary of name: ticker pairs
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of DataFrames
        """
        collected_data = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_ticker = {
                executor.submit(
                    self._download_ticker,
                    ticker_symbol,
                    start_date,
                    end_date,
                    name
                ): name
                for name, ticker_symbol in tickers_dict.items()
            }
            
            # Collect results
            for future in as_completed(future_to_ticker):
                name = future_to_ticker[future]
                
                try:
                    data = future.result(timeout=self.timeout)
                    if data is not None and not data.empty:
                        collected_data[name] = data
                        self.logger.debug(f"✓ Collected {name}: {len(data)} rows")
                    else:
                        self.logger.debug(f"✗ No data for {name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to collect {name}: {e}")
                    self.metrics['requests_failed'] += 1
        
        return collected_data
    
    def _download_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        name: str
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            name: Friendly name for ticker
            
        Returns:
            DataFrame with ticker data
        """
        # Rate limiting
        self._rate_limit()
        
        try:
            self.metrics['requests_made'] += 1
            
            # Download using yfinance
            ticker_obj = yf.Ticker(ticker)
            
            # Get historical data
            hist = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,
                actions=False
            )
            
            if hist.empty:
                self.logger.debug(f"No data returned for {ticker}")
                return None
            
            # Process data
            df = pd.DataFrame()
            
            # Use closing price as main value
            df[name] = hist['Close']
            
            # Add additional metrics if useful
            if name in ['vix', 'dxy_index', 'usd_brl_yahoo']:
                # For these, also keep high/low for volatility calculations
                df[f'{name}_high'] = hist['High']
                df[f'{name}_low'] = hist['Low']
                df[f'{name}_volume'] = hist['Volume'] if 'Volume' in hist.columns else np.nan
            
            # For commodities, calculate returns
            if any(x in name for x in ['brent', 'wti', 'gold', 'soybeans', 'sugar', 'coffee']):
                df[f'{name}_returns'] = hist['Close'].pct_change()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading {ticker}: {e}")
            return None
    
    def _rate_limit(self):
        """Implement rate limiting to avoid being blocked."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1.0 / self.requests_per_second):
            sleep_time = (1.0 / self.requests_per_second) - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _merge_ticker_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge data from multiple tickers.
        
        Args:
            data_dict: Dictionary of DataFrames
            
        Returns:
            Merged DataFrame
        """
        # Start with empty DataFrame
        merged = pd.DataFrame()
        
        for name, data in data_dict.items():
            if merged.empty:
                merged = data
            else:
                # Merge on index (date)
                merged = merged.join(data, how='outer')
        
        return merged
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics from raw data.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with derived metrics
        """
        # Dollar strength indicator (DXY momentum)
        if 'dxy_index' in df.columns:
            df['dxy_momentum_5d'] = df['dxy_index'].pct_change(5)
            df['dxy_momentum_22d'] = df['dxy_index'].pct_change(22)
            
            # DXY volatility
            if 'dxy_index_high' in df.columns and 'dxy_index_low' in df.columns:
                df['dxy_volatility'] = (df['dxy_index_high'] - df['dxy_index_low']) / df['dxy_index']
        
        # Commodity volatility index
        commodity_cols = [col for col in df.columns if any(
            x in col for x in ['brent', 'wti', 'gold', 'soybeans', 'sugar', 'coffee']
        ) and 'returns' in col]
        
        if commodity_cols:
            # Average volatility across commodities
            commodity_returns = df[commodity_cols]
            df['commodity_volatility'] = commodity_returns.std(axis=1) * np.sqrt(252)
        
        # EM correlation indicator
        if 'emerging_markets' in df.columns and 'sp500' in df.columns:
            # Rolling correlation between EM and S&P
            em_returns = df['emerging_markets'].pct_change()
            sp_returns = df['sp500'].pct_change()
            df['em_sp500_correlation'] = em_returns.rolling(66).corr(sp_returns)
        
        # Risk on/off indicator
        risk_on_assets = ['sp500', 'emerging_markets', 'bitcoin']
        risk_off_assets = ['gold', 'us_10y', 'dxy_index']
        
        # Calculate average performance of risk-on vs risk-off
        risk_on_cols = [col for col in df.columns if any(x in col for x in risk_on_assets)]
        risk_off_cols = [col for col in df.columns if any(x in col for x in risk_off_assets)]
        
        if risk_on_cols and risk_off_cols:
            risk_on_perf = df[risk_on_cols].pct_change().mean(axis=1)
            risk_off_perf = df[risk_off_cols].pct_change().mean(axis=1)
            df['risk_on_off_indicator'] = risk_on_perf - risk_off_perf
        
        # Yield curve (if available)
        if 'us_10y' in df.columns and 'us_2y' in df.columns:
            df['yield_curve_slope'] = df['us_10y'] - df['us_2y']
            df['yield_curve_inverted'] = (df['yield_curve_slope'] < 0).astype(int)
        
        # Commodity super index (Brazil relevant)
        brazil_commodities = ['soybeans', 'sugar', 'coffee', 'corn', 'brent', 'vale']
        brazil_comm_cols = [col for col in df.columns if any(
            x == col.split('_')[0] for x in brazil_commodities
        )]
        
        if len(brazil_comm_cols) > 2:
            # Normalize and average
            normalized = pd.DataFrame()
            for col in brazil_comm_cols:
                if df[col].first_valid_index() is not None:
                    first_val = df[col].loc[df[col].first_valid_index()]
                    normalized[col] = df[col] / first_val * 100
            
            if not normalized.empty:
                df['brazil_commodity_basket'] = normalized.mean(axis=1)
        
        # Emerging markets stress indicator
        if 'vix' in df.columns and 'emerging_markets' in df.columns:
            vix_zscore = (df['vix'] - df['vix'].rolling(252).mean()) / df['vix'].rolling(252).std()
            em_drawdown = df['emerging_markets'] / df['emerging_markets'].rolling(252).max() - 1
            df['em_stress_indicator'] = vix_zscore * abs(em_drawdown)
        
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate Yahoo Finance data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        # Check for essential columns
        essential_cols = ['dxy_index', 'vix']
        missing_essential = [col for col in essential_cols if col not in data.columns]
        
        if missing_essential:
            self.logger.warning(f"Missing essential columns: {missing_essential}")
            # Don't fail completely, but log warning
        
        # Check data quality
        validations = []
        
        # DXY should be in reasonable range
        if 'dxy_index' in data.columns:
            valid_range = (70 <= data['dxy_index']) & (data['dxy_index'] <= 120)
            if not valid_range.all():
                invalid_count = (~valid_range).sum()
                self.logger.warning(f"DXY values out of range: {invalid_count} rows")
            validations.append(valid_range.mean() > 0.95)  # Allow 5% outliers
        
        # VIX should be positive and reasonable
        if 'vix' in data.columns:
            valid_range = (5 <= data['vix']) & (data['vix'] <= 100)
            if not valid_range.all():
                invalid_count = (~valid_range).sum()
                self.logger.warning(f"VIX values out of range: {invalid_count} rows")
            validations.append(valid_range.mean() > 0.95)
        
        # Check for too many missing values
        missing_pct = (data.isnull().sum() / len(data)) * 100
        excessive_missing = missing_pct[missing_pct > 20]
        
        if not excessive_missing.empty:
            self.logger.warning(f"Excessive missing data (>20%): {excessive_missing.to_dict()}")
            # Don't fail for missing data in non-essential columns
            
        return all(validations) if validations else True
    
    def get_real_time_quote(self, ticker_name: str) -> Optional[float]:
        """
        Get real-time quote for a ticker.
        
        Args:
            ticker_name: Name of ticker in config
            
        Returns:
            Current price or None
        """
        if ticker_name not in self.tickers:
            self.logger.error(f"Unknown ticker: {ticker_name}")
            return None
        
        ticker_symbol = self.tickers[ticker_name]
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Try different price fields
            for field in ['regularMarketPrice', 'price', 'previousClose']:
                if field in info and info[field]:
                    return float(info[field])
            
            # Fallback to last historical close
            hist = ticker.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting quote for {ticker_name}: {e}")
            return None
    
    def get_options_data(self, ticker_name: str) -> Optional[pd.DataFrame]:
        """
        Get options data for volatility analysis.
        
        Args:
            ticker_name: Name of ticker
            
        Returns:
            DataFrame with options data
        """
        if ticker_name not in self.tickers:
            return None
        
        ticker_symbol = self.tickers[ticker_name]
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                return None
            
            # Get options for next expiration
            opt = ticker.option_chain(expirations[0])
            
            # Combine calls and puts
            calls = opt.calls
            calls['type'] = 'call'
            
            puts = opt.puts
            puts['type'] = 'put'
            
            options_data = pd.concat([calls, puts])
            
            # Calculate implied volatility metrics
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error getting options for {ticker_name}: {e}")
            return None