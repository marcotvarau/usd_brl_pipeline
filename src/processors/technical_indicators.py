"""
Technical Indicators Processor
Calculates technical analysis indicators for USD/BRL
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import talib as ta
from scipy import stats


class TechnicalIndicatorProcessor:
    """
    Processor for technical analysis indicators.
    
    Implements:
    - Moving averages (SMA, EMA, WMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
    - Volume indicators (OBV, Volume SMA)
    - Custom forex indicators
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize technical indicator processor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load technical indicator settings
        tech_config = config.get('features', {}).get('technical', {})
        
        self.sma_periods = tech_config.get('sma_periods', [5, 10, 21, 50, 200])
        self.ema_periods = tech_config.get('ema_periods', [12, 26])
        self.rsi_period = tech_config.get('rsi_period', 14)
        self.macd_fast = tech_config.get('macd_fast', 12)
        self.macd_slow = tech_config.get('macd_slow', 26)
        self.macd_signal = tech_config.get('macd_signal', 9)
        self.bb_period = tech_config.get('bb_period', 20)
        self.bb_std = tech_config.get('bb_std', 2)
        
        # Forex specific settings
        self.atr_period = 14
        self.adx_period = 14
        self.cci_period = 20
        self.williams_r_period = 14
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: Input DataFrame with OHLC data
            
        Returns:
            DataFrame with technical indicators
        """
        self.logger.info("Calculating technical indicators")
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Check for required price column
        price_col = self._get_price_column(data)
        if price_col is None:
            self.logger.warning("No price column found for technical indicators")
            return data
        
        # 1. Trend indicators
        data = self.calculate_moving_averages(data, price_col)
        data = self.calculate_macd(data, price_col)
        data = self.calculate_adx(data)
        data = self.calculate_ichimoku(data, price_col)
        
        # 2. Momentum indicators
        data = self.calculate_rsi(data, price_col)
        data = self.calculate_stochastic(data)
        data = self.calculate_williams_r(data)
        data = self.calculate_cci(data)
        data = self.calculate_momentum_oscillator(data, price_col)
        
        # 3. Volatility indicators
        data = self.calculate_bollinger_bands(data, price_col)
        data = self.calculate_atr(data)
        data = self.calculate_keltner_channels(data, price_col)
        data = self.calculate_donchian_channels(data)
        
        # 4. Volume indicators (if volume available)
        if self._has_volume(data):
            data = self.calculate_volume_indicators(data)
        
        # 5. Custom forex indicators
        data = self.calculate_forex_specific_indicators(data, price_col)
        
        # 6. Pattern recognition
        data = self.calculate_price_patterns(data, price_col)
        
        self.logger.info(f"Technical indicators calculated: {len([c for c in data.columns if c not in df.columns])} new features")
        
        return data
    
    def calculate_moving_averages(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate various moving averages.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with moving averages
        """
        # Simple Moving Averages
        for period in self.sma_periods:
            df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
            
            # Distance from SMA
            df[f'price_to_sma_{period}'] = (df[price_col] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
        
        # Exponential Moving Averages
        for period in self.ema_periods:
            df[f'ema_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
            
            # Distance from EMA
            df[f'price_to_ema_{period}'] = (df[price_col] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
        
        # Weighted Moving Average
        for period in [10, 20]:
            weights = np.arange(1, period + 1)
            df[f'wma_{period}'] = df[price_col].rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # Hull Moving Average (HMA)
        for period in [9, 16]:
            wma_half = df[price_col].rolling(period // 2).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            wma_full = df[price_col].rolling(period).apply(
                lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
            )
            df[f'hma_{period}'] = (2 * wma_half - wma_full).rolling(int(np.sqrt(period))).mean()
        
        # Moving Average Convergence
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['golden_cross'] = (
                (df['sma_50'] > df['sma_200']) & 
                (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
            ).astype(int)
            
            df['death_cross'] = (
                (df['sma_50'] < df['sma_200']) & 
                (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
            ).astype(int)
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate MACD indicator.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with MACD
        """
        # MACD line
        ema_fast = df[price_col].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_bullish_cross'] = (
            (df['macd_line'] > df['macd_signal']) & 
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        
        df['macd_bearish_cross'] = (
            (df['macd_line'] < df['macd_signal']) & 
            (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)
        
        # MACD divergence (simplified)
        price_higher = df[price_col] > df[price_col].shift(20)
        macd_lower = df['macd_line'] < df['macd_line'].shift(20)
        df['macd_bearish_divergence'] = (price_higher & macd_lower).astype(int)
        
        price_lower = df[price_col] < df[price_col].shift(20)
        macd_higher = df['macd_line'] > df['macd_line'].shift(20)
        df['macd_bullish_divergence'] = (price_lower & macd_higher).astype(int)
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with RSI
        """
        # Calculate price changes
        delta = df[price_col].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses (Wilder's smoothing)
        avg_gains = gains.rolling(window=self.rsi_period).mean()
        avg_losses = losses.rolling(window=self.rsi_period).mean()
        
        # Use EMA for smoothing after initial SMA
        for i in range(self.rsi_period, len(df)):
            avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (self.rsi_period - 1) + gains.iloc[i]) / self.rsi_period
            avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (self.rsi_period - 1) + losses.iloc[i]) / self.rsi_period
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI levels
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # RSI divergence
        price_higher = df[price_col] > df[price_col].shift(14)
        rsi_lower = df['rsi'] < df['rsi'].shift(14)
        df['rsi_bearish_divergence'] = (price_higher & rsi_lower).astype(int)
        
        price_lower = df[price_col] < df[price_col].shift(14)
        rsi_higher = df['rsi'] > df['rsi'].shift(14)
        df['rsi_bullish_divergence'] = (price_lower & rsi_higher).astype(int)
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(window=14).min()
        rsi_max = df['rsi'].rolling(window=14).max()
        df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-10)
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with Bollinger Bands
        """
        # Calculate middle band (SMA)
        df['bb_middle'] = df[price_col].rolling(window=self.bb_period).mean()
        
        # Calculate standard deviation
        bb_std = df[price_col].rolling(window=self.bb_period).std()
        
        # Calculate bands
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        
        # Band width and position
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_ratio'] = df['bb_width'] / df['bb_middle']
        
        # Price position within bands (0 = lower band, 1 = upper band)
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_width'] + 1e-10)
        
        # Band squeeze (low volatility)
        bb_width_ma = df['bb_width'].rolling(window=120).mean()
        bb_width_std = df['bb_width'].rolling(window=120).std()
        df['bb_squeeze'] = (df['bb_width'] < (bb_width_ma - bb_width_std)).astype(int)
        
        # Price touches
        df['bb_upper_touch'] = (df[price_col] >= df['bb_upper']).astype(int)
        df['bb_lower_touch'] = (df[price_col] <= df['bb_lower']).astype(int)
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ATR
        """
        if not all(col in df.columns for col in ['usd_brl_high', 'usd_brl_low', 'usd_brl_ptax_close']):
            # Create synthetic high/low if not available
            if 'usd_brl_ptax_close' in df.columns:
                price = df['usd_brl_ptax_close']
                df['usd_brl_high'] = price.rolling(window=1).max()
                df['usd_brl_low'] = price.rolling(window=1).min()
            else:
                return df
        
        high = df['usd_brl_high']
        low = df['usd_brl_low']
        close = df['usd_brl_ptax_close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR (Wilder's smoothing)
        df['atr'] = df['true_range'].rolling(window=self.atr_period).mean()
        
        # Smooth using Wilder's method
        for i in range(self.atr_period, len(df)):
            df.loc[df.index[i], 'atr'] = (
                (df.loc[df.index[i-1], 'atr'] * (self.atr_period - 1) + 
                 df.loc[df.index[i], 'true_range']) / self.atr_period
            )
        
        # ATR percentage
        df['atr_percent'] = (df['atr'] / close) * 100
        
        # Volatility regime based on ATR
        atr_ma = df['atr'].rolling(window=100).mean()
        df['high_volatility'] = (df['atr'] > atr_ma * 1.5).astype(int)
        df['low_volatility'] = (df['atr'] < atr_ma * 0.5).astype(int)
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Stochastic
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return df
        
        if 'usd_brl_high' not in df.columns or 'usd_brl_low' not in df.columns:
            # Create synthetic high/low
            df['usd_brl_high'] = df['usd_brl_ptax_close'].rolling(window=14).max()
            df['usd_brl_low'] = df['usd_brl_ptax_close'].rolling(window=14).min()
        
        # Calculate %K
        low_14 = df['usd_brl_low'].rolling(window=14).min()
        high_14 = df['usd_brl_high'].rolling(window=14).max()
        
        df['stoch_k'] = 100 * ((df['usd_brl_ptax_close'] - low_14) / (high_14 - low_14 + 1e-10))
        
        # Calculate %D (3-period SMA of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Stochastic levels
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        # Stochastic crossovers
        df['stoch_bullish_cross'] = (
            (df['stoch_k'] > df['stoch_d']) & 
            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        ).astype(int)
        
        df['stoch_bearish_cross'] = (
            (df['stoch_k'] < df['stoch_d']) & 
            (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        ).astype(int)
        
        return df
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ADX
        """
        if not all(col in df.columns for col in ['usd_brl_high', 'usd_brl_low', 'usd_brl_ptax_close']):
            return df
        
        high = df['usd_brl_high']
        low = df['usd_brl_low']
        close = df['usd_brl_ptax_close']
        
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Remove conflicting movements
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        
        # Calculate True Range (already calculated in ATR)
        if 'true_range' not in df.columns:
            df = self.calculate_atr(df)
        
        # Smooth using Wilder's method
        atr_smooth = df['true_range'].ewm(span=self.adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr_smooth)
        minus_di = 100 * (minus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr_smooth)
        
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # Calculate ADX
        df['adx'] = dx.ewm(span=self.adx_period, adjust=False).mean()
        
        # Trend strength
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend'] = (df['adx'] < 20).astype(int)
        
        # Trend direction
        df['bullish_trend'] = ((df['plus_di'] > df['minus_di']) & (df['adx'] > 25)).astype(int)
        df['bearish_trend'] = ((df['minus_di'] > df['plus_di']) & (df['adx'] > 25)).astype(int)
        
        return df
    
    def calculate_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Williams %R
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return df
        
        if 'usd_brl_high' not in df.columns or 'usd_brl_low' not in df.columns:
            df['usd_brl_high'] = df['usd_brl_ptax_close'].rolling(window=self.williams_r_period).max()
            df['usd_brl_low'] = df['usd_brl_ptax_close'].rolling(window=self.williams_r_period).min()
        
        # Calculate Williams %R
        high_n = df['usd_brl_high'].rolling(window=self.williams_r_period).max()
        low_n = df['usd_brl_low'].rolling(window=self.williams_r_period).min()
        
        df['williams_r'] = -100 * ((high_n - df['usd_brl_ptax_close']) / (high_n - low_n + 1e-10))
        
        # Williams %R levels
        df['williams_oversold'] = (df['williams_r'] < -80).astype(int)
        df['williams_overbought'] = (df['williams_r'] > -20).astype(int)
        
        return df
    
    def calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with CCI
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return df
        
        # Calculate Typical Price
        if all(col in df.columns for col in ['usd_brl_high', 'usd_brl_low']):
            typical_price = (df['usd_brl_high'] + df['usd_brl_low'] + df['usd_brl_ptax_close']) / 3
        else:
            typical_price = df['usd_brl_ptax_close']
        
        # Calculate CCI
        sma = typical_price.rolling(window=self.cci_period).mean()
        mad = typical_price.rolling(window=self.cci_period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        df['cci'] = (typical_price - sma) / (0.015 * mad + 1e-10)
        
        # CCI levels
        df['cci_oversold'] = (df['cci'] < -100).astype(int)
        df['cci_overbought'] = (df['cci'] > 100).astype(int)
        
        return df
    
    def calculate_ichimoku(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with Ichimoku indicators
        """
        if 'usd_brl_high' not in df.columns or 'usd_brl_low' not in df.columns:
            # Create synthetic high/low
            df['usd_brl_high'] = df[price_col].rolling(window=9).max()
            df['usd_brl_low'] = df[price_col].rolling(window=9).min()
        
        high = df['usd_brl_high']
        low = df['usd_brl_low']
        
        # Tenkan-sen (Conversion Line)
        high_9 = high.rolling(window=9).max()
        low_9 = low.rolling(window=9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = high.rolling(window=26).max()
        low_26 = low.rolling(window=26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = high.rolling(window=52).max()
        low_52 = low.rolling(window=52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['ichimoku_chikou'] = df[price_col].shift(-26)
        
        # Cloud thickness
        df['ichimoku_cloud_thickness'] = abs(df['ichimoku_senkou_a'] - df['ichimoku_senkou_b'])
        
        # Price position relative to cloud
        df['above_cloud'] = (
            (df[price_col] > df['ichimoku_senkou_a']) & 
            (df[price_col] > df['ichimoku_senkou_b'])
        ).astype(int)
        
        df['below_cloud'] = (
            (df[price_col] < df['ichimoku_senkou_a']) & 
            (df[price_col] < df['ichimoku_senkou_b'])
        ).astype(int)
        
        df['in_cloud'] = (~(df['above_cloud'].astype(bool) | df['below_cloud'].astype(bool))).astype(int)
        
        return df
    
    def calculate_keltner_channels(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with Keltner Channels
        """
        # Calculate EMA as middle line
        df['keltner_middle'] = df[price_col].ewm(span=20, adjust=False).mean()
        
        # Use ATR for channel width
        if 'atr' not in df.columns:
            df = self.calculate_atr(df)
        
        # Calculate channels (typically 2 ATR)
        df['keltner_upper'] = df['keltner_middle'] + (2 * df['atr'])
        df['keltner_lower'] = df['keltner_middle'] - (2 * df['atr'])
        
        # Position within channel
        df['keltner_position'] = (
            (df[price_col] - df['keltner_lower']) / 
            (df['keltner_upper'] - df['keltner_lower'] + 1e-10)
        )
        
        return df
    
    def calculate_donchian_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Donchian Channels
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return df
        
        # 20-period Donchian Channel
        df['donchian_upper_20'] = df['usd_brl_ptax_close'].rolling(window=20).max()
        df['donchian_lower_20'] = df['usd_brl_ptax_close'].rolling(window=20).min()
        df['donchian_middle_20'] = (df['donchian_upper_20'] + df['donchian_lower_20']) / 2
        
        # 55-period Donchian Channel (Turtle Trading)
        df['donchian_upper_55'] = df['usd_brl_ptax_close'].rolling(window=55).max()
        df['donchian_lower_55'] = df['usd_brl_ptax_close'].rolling(window=55).min()
        df['donchian_middle_55'] = (df['donchian_upper_55'] + df['donchian_lower_55']) / 2
        
        # Breakout signals
        df['donchian_breakout_up'] = (
            df['usd_brl_ptax_close'] >= df['donchian_upper_20']
        ).astype(int)
        
        df['donchian_breakout_down'] = (
            df['usd_brl_ptax_close'] <= df['donchian_lower_20']
        ).astype(int)
        
        return df
    
    def calculate_momentum_oscillator(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate Momentum Oscillator.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with Momentum
        """
        # Rate of Change (ROC)
        for period in [10, 20]:
            df[f'roc_{period}'] = (
                (df[price_col] - df[price_col].shift(period)) / 
                df[price_col].shift(period) * 100
            )
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df[price_col] - df[price_col].shift(period)
        
        # Price Oscillator
        short_ema = df[price_col].ewm(span=12, adjust=False).mean()
        long_ema = df[price_col].ewm(span=26, adjust=False).mean()
        df['price_oscillator'] = ((short_ema - long_ema) / long_ema) * 100
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volume indicators
        """
        if 'volume' not in df.columns:
            return df
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['usd_brl_ptax_close'].iloc[i] > df['usd_brl_ptax_close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['usd_brl_ptax_close'].iloc[i] < df['usd_brl_ptax_close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        df['obv_ma'] = df['obv'].rolling(window=20).mean()
        
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['usd_brl_ptax_close'].pct_change()).cumsum()
        
        # Money Flow Index (MFI) - simplified
        typical_price = df['usd_brl_ptax_close']
        raw_money_flow = typical_price * df['volume']
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        money_ratio = positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-10)
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        return df
    
    def calculate_forex_specific_indicators(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Calculate forex-specific indicators.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with forex indicators
        """
        # Pivot Points
        if all(col in df.columns for col in ['usd_brl_high', 'usd_brl_low']):
            high = df['usd_brl_high']
            low = df['usd_brl_low']
            close = df[price_col]
            
            # Standard Pivot Points
            df['pivot'] = (high + low + close) / 3
            df['pivot_r1'] = 2 * df['pivot'] - low
            df['pivot_s1'] = 2 * df['pivot'] - high
            df['pivot_r2'] = df['pivot'] + (high - low)
            df['pivot_s2'] = df['pivot'] - (high - low)
            
            # Fibonacci Pivot Points
            df['fib_pivot'] = (high + low + close) / 3
            df['fib_r1'] = df['fib_pivot'] + 0.382 * (high - low)
            df['fib_r2'] = df['fib_pivot'] + 0.618 * (high - low)
            df['fib_s1'] = df['fib_pivot'] - 0.382 * (high - low)
            df['fib_s2'] = df['fib_pivot'] - 0.618 * (high - low)
        
        # Currency Strength Index (simplified)
        returns = df[price_col].pct_change()
        df['currency_strength'] = returns.rolling(window=14).mean() * 100
        
        # Relative Vigor Index (RVI)
        if all(col in df.columns for col in ['usd_brl_high', 'usd_brl_low', 'usd_brl_ptax_close']):
            close_open = df['usd_brl_ptax_close'] - df['usd_brl_ptax_close'].shift(1)
            high_low = df['usd_brl_high'] - df['usd_brl_low']
            
            numerator = close_open.rolling(window=10).mean()
            denominator = high_low.rolling(window=10).mean()
            
            df['rvi'] = numerator / (denominator + 1e-10)
            df['rvi_signal'] = df['rvi'].rolling(window=4).mean()
        
        return df
    
    def calculate_price_patterns(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Identify common price patterns.
        
        Args:
            df: Input DataFrame
            price_col: Name of price column
            
        Returns:
            DataFrame with pattern indicators
        """
        # Simple pattern detection
        price = df[price_col]
        
        # Higher highs and higher lows (uptrend)
        high_5 = price.rolling(window=5).max()
        low_5 = price.rolling(window=5).min()
        
        df['higher_high'] = (high_5 > high_5.shift(5)).astype(int)
        df['higher_low'] = (low_5 > low_5.shift(5)).astype(int)
        df['uptrend'] = (df['higher_high'] & df['higher_low']).astype(int)
        
        # Lower highs and lower lows (downtrend)
        df['lower_high'] = (high_5 < high_5.shift(5)).astype(int)
        df['lower_low'] = (low_5 < low_5.shift(5)).astype(int)
        df['downtrend'] = (df['lower_high'] & df['lower_low']).astype(int)
        
        # Consolidation (range-bound)
        price_std = price.rolling(window=20).std()
        df['consolidation'] = (price_std < price_std.rolling(window=100).quantile(0.25)).astype(int)
        
        # Breakout detection
        resistance = price.rolling(window=20).max()
        support = price.rolling(window=20).min()
        
        df['breakout_up'] = (price > resistance.shift(1)).astype(int)
        df['breakout_down'] = (price < support.shift(1)).astype(int)
        
        # Gap detection
        if all(col in df.columns for col in ['usd_brl_high', 'usd_brl_low']):
            df['gap_up'] = (df['usd_brl_low'] > df['usd_brl_high'].shift(1)).astype(int)
            df['gap_down'] = (df['usd_brl_high'] < df['usd_brl_low'].shift(1)).astype(int)
        
        return df
    
    def _get_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify the main price column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of price column or None
        """
        price_candidates = ['usd_brl_ptax_close', 'usd_brl_close', 'close', 'Close']
        
        for col in price_candidates:
            if col in df.columns:
                return col
        
        return None
    
    def _has_volume(self, df: pd.DataFrame) -> bool:
        """
        Check if volume data is available.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if volume data exists
        """
        volume_candidates = ['volume', 'Volume', 'usd_brl_volume']
        
        for col in volume_candidates:
            if col in df.columns:
                # Check if volume has actual data (not all NaN or zero)
                if df[col].notna().sum() > 0 and (df[col] != 0).sum() > 0:
                    return True
        
        return False