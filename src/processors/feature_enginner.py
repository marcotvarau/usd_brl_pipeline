"""
Feature Engineering Module
Creates derived features for USD/BRL prediction model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Main feature engineering class for USD/BRL pipeline.
    
    Creates:
    - Economic indicators (interest rate differentials, carry trade metrics)
    - Risk sentiment scores
    - Commodity indices
    - Market microstructure features
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load feature configuration
        self.commodity_weights = config.get('features', {}).get('commodity_weights', {
            'oil': 0.162,
            'soybeans': 0.159,
            'iron_ore': 0.080,
            'food': 0.050,
            'sugar': 0.025,
            'coffee': 0.020
        })
        
        self.lag_periods = config.get('features', {}).get('lag_periods', [1, 2, 5, 10, 22, 44, 66])
        self.rolling_windows = config.get('features', {}).get('rolling_windows', [5, 10, 22, 44, 66])
        
        # Feature groups for organization
        self.feature_groups = {
            'tier1': [],
            'tier2': [],
            'tier3': [],
            'tier4': []
        }
    
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering process")
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # 1. Economic features (Tier 1)
        self.logger.info("Creating economic features...")
        df = self.create_interest_rate_features(df)
        df = self.create_volatility_features(df)
        
        # 2. Commodity and trade features (Tier 2)
        self.logger.info("Creating commodity features...")
        df = self.create_commodity_index(df)
        df = self.create_trade_balance_features(df)
        
        # 3. Sentiment and positioning features (Tier 3)
        self.logger.info("Creating sentiment features...")
        df = self.create_risk_sentiment_score(df)
        df = self.create_carry_trade_features(df)
        df = self.create_correlation_features(df)
        
        # 4. Technical and temporal features (Tier 4)
        self.logger.info("Creating technical features...")
        df = self.create_price_action_features(df)
        df = self.create_seasonal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        
        # 5. Interaction features
        self.logger.info("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # 6. Clean and validate
        df = self.clean_engineered_features(df)
        
        self.logger.info(f"Feature engineering complete. Created {len(df.columns)} features")
        
        return df
    
    def create_interest_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interest rate related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interest rate features
        """
        # Real interest rate differential
        if all(col in df.columns for col in ['selic_rate', 'fed_funds_rate', 'ipca_monthly', 'us_cpi_yoy']):
            # Calculate real rates
            df['brazil_real_rate'] = df['selic_rate'] - df['ipca_monthly']
            df['us_real_rate'] = df['fed_funds_rate'] - df['us_cpi_yoy']
            
            # Real rate differential
            df['real_interest_differential'] = df['brazil_real_rate'] - df['us_real_rate']
            
            # Moving averages of differential
            for window in [5, 22, 66]:
                df[f'real_diff_ma_{window}'] = df['real_interest_differential'].rolling(window).mean()
            
            # Rate of change of differential
            df['real_diff_change_5d'] = df['real_interest_differential'].diff(5)
            df['real_diff_change_22d'] = df['real_interest_differential'].diff(22)
            
            self.feature_groups['tier1'].extend([
                'real_interest_differential', 'brazil_real_rate', 'us_real_rate'
            ])
        
        # Yield curve features
        if 'us_10y_treasury' in df.columns and 'us_2y_treasury' in df.columns:
            df['us_yield_curve_slope'] = df['us_10y_treasury'] - df['us_2y_treasury']
            df['yield_curve_inverted'] = (df['us_yield_curve_slope'] < 0).astype(int)
            
            self.feature_groups['tier2'].extend(['us_yield_curve_slope', 'yield_curve_inverted'])
        
        # Interest rate momentum
        if 'selic_rate' in df.columns:
            df['selic_momentum_5d'] = df['selic_rate'].pct_change(5)
            df['selic_momentum_22d'] = df['selic_rate'].pct_change(22)
        
        if 'fed_funds_rate' in df.columns:
            df['fed_momentum_5d'] = df['fed_funds_rate'].pct_change(5)
            df['fed_momentum_22d'] = df['fed_funds_rate'].pct_change(22)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volatility features
        """
        if 'usd_brl_ptax_close' in df.columns:
            # Calculate returns
            df['usd_brl_returns'] = df['usd_brl_ptax_close'].pct_change()
            df['usd_brl_log_returns'] = np.log(df['usd_brl_ptax_close'] / df['usd_brl_ptax_close'].shift(1))
            
            # Realized volatility (different windows)
            for window in [5, 10, 22, 44, 66]:
                df[f'usd_brl_volatility_{window}d'] = df['usd_brl_log_returns'].rolling(window).std() * np.sqrt(252)
            
            # EWMA volatility
            df['usd_brl_ewma_vol'] = df['usd_brl_log_returns'].ewm(span=22).std() * np.sqrt(252)
            
            # Parkinson volatility estimator (if high/low available)
            if 'usd_brl_high' in df.columns and 'usd_brl_low' in df.columns:
                df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * 
                                              (np.log(df['usd_brl_high']/df['usd_brl_low'])**2).rolling(22).mean() * 252)
            
            # Volatility of volatility
            df['vol_of_vol'] = df['usd_brl_volatility_22d'].rolling(22).std()
            
            # Volatility regime
            vol_mean = df['usd_brl_volatility_22d'].rolling(252).mean()
            vol_std = df['usd_brl_volatility_22d'].rolling(252).std()
            df['volatility_zscore'] = (df['usd_brl_volatility_22d'] - vol_mean) / vol_std
            df['high_vol_regime'] = (df['volatility_zscore'] > 1).astype(int)
            
            self.feature_groups['tier1'].extend([
                'usd_brl_volatility_22d', 'usd_brl_ewma_vol', 'volatility_zscore'
            ])
        
        # Cross-asset volatility
        if 'vix' in df.columns:
            df['vix_ma_5'] = df['vix'].rolling(5).mean()
            df['vix_ma_22'] = df['vix'].rolling(22).mean()
            df['vix_above_20'] = (df['vix'] > 20).astype(int)
            df['vix_above_30'] = (df['vix'] > 30).astype(int)
            
            self.feature_groups['tier2'].append('vix')
        
        return df
    
    def create_commodity_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Brazilian export-weighted commodity index.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with commodity index
        """
        commodity_columns = {
            'oil': ['brent', 'wti'],
            'soybeans': ['soybeans'],
            'sugar': ['sugar'],
            'coffee': ['coffee'],
            'iron_ore': ['vale', 'bhp']  # Using mining stocks as proxy
        }
        
        # Find available commodity columns
        available_commodities = {}
        for commodity, cols in commodity_columns.items():
            for col in cols:
                if col in df.columns:
                    available_commodities[commodity] = col
                    break
        
        if available_commodities:
            # Normalize prices (base 100)
            normalized_prices = pd.DataFrame(index=df.index)
            total_weight = 0
            
            for commodity, col in available_commodities.items():
                if commodity in self.commodity_weights:
                    # Normalize to base 100
                    first_valid = df[col].first_valid_index()
                    if first_valid is not None:
                        normalized_prices[commodity] = (df[col] / df[col].loc[first_valid]) * 100
                        total_weight += self.commodity_weights[commodity]
            
            # Create weighted index
            if not normalized_prices.empty and total_weight > 0:
                df['brazilian_commodity_index'] = 0
                for commodity in normalized_prices.columns:
                    weight = self.commodity_weights[commodity] / total_weight  # Renormalize weights
                    df['brazilian_commodity_index'] += normalized_prices[commodity] * weight
                
                # Commodity index momentum
                df['commodity_momentum_5d'] = df['brazilian_commodity_index'].pct_change(5)
                df['commodity_momentum_22d'] = df['brazilian_commodity_index'].pct_change(22)
                
                self.feature_groups['tier2'].append('brazilian_commodity_index')
        
        return df
    
    def create_risk_sentiment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk sentiment score.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk sentiment score
        """
        risk_components = []
        weights = []
        
        # VIX component
        if 'vix' in df.columns:
            vix_normalized = self._normalize_percentile(df['vix'], window=252)
            risk_components.append(vix_normalized)
            weights.append(0.30)
        
        # Credit spreads component (if available)
        if 'credit_spreads' in df.columns:
            spreads_normalized = self._normalize_percentile(df['credit_spreads'], window=252)
            risk_components.append(spreads_normalized)
            weights.append(0.25)
        
        # FX volatility component
        if 'usd_brl_volatility_22d' in df.columns:
            fx_vol_normalized = self._normalize_percentile(df['usd_brl_volatility_22d'], window=252)
            risk_components.append(fx_vol_normalized)
            weights.append(0.20)
        
        # Commodity volatility (if available)
        if 'commodity_volatility' in df.columns:
            comm_vol_normalized = self._normalize_percentile(df['commodity_volatility'], window=252)
            risk_components.append(comm_vol_normalized)
            weights.append(0.15)
        
        # Safe haven flows (inverse - gold, USD index)
        if 'dxy_index' in df.columns:
            dxy_momentum = df['dxy_index'].pct_change(5)
            dxy_normalized = self._normalize_percentile(dxy_momentum, window=252)
            risk_components.append(dxy_normalized)
            weights.append(0.10)
        
        # Calculate weighted score
        if risk_components:
            # Normalize weights to sum to 1
            weights = np.array(weights) / sum(weights)
            
            # Create risk sentiment score (0-100 scale)
            df['risk_sentiment_score'] = sum(comp * w for comp, w in zip(risk_components, weights))
            df['risk_sentiment_score'] = df['risk_sentiment_score'].clip(0, 100)
            
            # Risk sentiment regimes
            df['risk_off'] = (df['risk_sentiment_score'] > 60).astype(int)
            df['risk_on'] = (df['risk_sentiment_score'] < 40).astype(int)
            
            self.feature_groups['tier3'].append('risk_sentiment_score')
        
        return df
    
    def create_carry_trade_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create carry trade attractiveness features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with carry trade features
        """
        if all(col in df.columns for col in ['real_interest_differential', 'usd_brl_volatility_22d']):
            # Basic carry trade return
            df['carry_return'] = df['real_interest_differential'] / 100
            
            # Risk-adjusted carry (Sharpe-like ratio)
            vol_adj = df['usd_brl_volatility_22d'].rolling(22).mean()
            df['carry_trade_attractiveness'] = df['carry_return'] / (vol_adj + 0.01)
            
            # Carry momentum
            df['carry_momentum'] = df['carry_trade_attractiveness'].diff(22)
            
            # Carry regime
            carry_ma = df['carry_trade_attractiveness'].rolling(66).mean()
            df['carry_above_average'] = (df['carry_trade_attractiveness'] > carry_ma).astype(int)
            
            self.feature_groups['tier3'].extend([
                'carry_trade_attractiveness', 'carry_momentum'
            ])
        
        return df
    
    def create_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling correlation features with other assets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with correlation features
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return df
        
        usd_brl_returns = df['usd_brl_ptax_close'].pct_change()
        
        correlation_pairs = [
            ('dxy_index', 'correlation_dxy'),
            ('sp500', 'correlation_sp500'),
            ('emerging_markets', 'correlation_em'),
            ('vix', 'correlation_vix'),
            ('brazilian_commodity_index', 'correlation_commodities')
        ]
        
        for asset, feature_name in correlation_pairs:
            if asset in df.columns:
                asset_returns = df[asset].pct_change()
                
                # Rolling correlations (different windows)
                for window in [22, 66]:
                    df[f'{feature_name}_{window}d'] = usd_brl_returns.rolling(window).corr(asset_returns)
                
                if window == 22:
                    self.feature_groups['tier3'].append(f'{feature_name}_22d')
        
        # Correlation stability (how stable are correlations)
        if 'correlation_dxy_22d' in df.columns:
            df['correlation_stability'] = df['correlation_dxy_22d'].rolling(66).std()
        
        return df
    
    def create_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price action and microstructure features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with price action features
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return df
        
        # Distance from moving averages
        for period in [5, 22, 50, 200]:
            ma = df['usd_brl_ptax_close'].rolling(period).mean()
            df[f'distance_from_ma_{period}'] = (df['usd_brl_ptax_close'] - ma) / ma * 100
        
        # Support/Resistance levels
        df['resistance_1m'] = df['usd_brl_ptax_close'].rolling(22).max()
        df['support_1m'] = df['usd_brl_ptax_close'].rolling(22).min()
        df['distance_from_resistance'] = (df['resistance_1m'] - df['usd_brl_ptax_close']) / df['usd_brl_ptax_close'] * 100
        df['distance_from_support'] = (df['usd_brl_ptax_close'] - df['support_1m']) / df['usd_brl_ptax_close'] * 100
        
        # Price momentum oscillators
        df['momentum_5d'] = df['usd_brl_ptax_close'].pct_change(5) * 100
        df['momentum_22d'] = df['usd_brl_ptax_close'].pct_change(22) * 100
        
        # Rate of change
        for period in [5, 10, 22]:
            df[f'roc_{period}d'] = ((df['usd_brl_ptax_close'] - df['usd_brl_ptax_close'].shift(period)) / 
                                    df['usd_brl_ptax_close'].shift(period)) * 100
        
        self.feature_groups['tier4'].extend([
            'distance_from_ma_22', 'momentum_22d', 'roc_22d'
        ])
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal and calendar features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with seasonal features
        """
        # Basic temporal features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        df['weekofyear'] = df.index.isocalendar().week
        
        # Cyclical encoding for periodicity
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        # Brazilian specific seasonality
        df['is_carnival'] = ((df['month'] == 2) | (df['month'] == 3)).astype(int)
        df['is_year_end'] = ((df['month'] >= 11) | (df['month'] <= 2)).astype(int)
        df['is_harvest_season'] = ((df['month'] >= 3) & (df['month'] <= 7)).astype(int)
        df['is_vacation_season'] = ((df['month'] == 1) | (df['month'] == 7) | (df['month'] == 12)).astype(int)
        
        # Trading patterns
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        df['is_mid_month'] = ((df['day'] >= 10) & (df['day'] <= 20)).astype(int)
        df['is_month_end'] = (df['day'] >= 25).astype(int)
        df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df['day'] >= 25)).astype(int)
        
        self.feature_groups['tier4'].extend([
            'month_sin', 'month_cos', 'is_harvest_season', 'is_year_end'
        ])
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for time series.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        # Key variables to lag
        lag_variables = [
            'usd_brl_ptax_close',
            'real_interest_differential',
            'risk_sentiment_score',
            'brazilian_commodity_index',
            'dxy_index',
            'vix'
        ]
        
        for var in lag_variables:
            if var in df.columns:
                for lag in self.lag_periods:
                    # Simple lag
                    df[f'{var}_lag_{lag}'] = df[var].shift(lag)
                    
                    # Lag returns
                    df[f'{var}_return_lag_{lag}'] = df[var].pct_change(lag)
                    
                    # Lag differences
                    df[f'{var}_diff_lag_{lag}'] = df[var].diff(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling features
        """
        # Variables for rolling statistics
        rolling_variables = [
            'usd_brl_ptax_close',
            'real_interest_differential',
            'risk_sentiment_score'
        ]
        
        for var in rolling_variables:
            if var in df.columns:
                for window in self.rolling_windows:
                    # Basic statistics
                    df[f'{var}_rolling_mean_{window}'] = df[var].rolling(window).mean()
                    df[f'{var}_rolling_std_{window}'] = df[var].rolling(window).std()
                    df[f'{var}_rolling_min_{window}'] = df[var].rolling(window).min()
                    df[f'{var}_rolling_max_{window}'] = df[var].rolling(window).max()
                    
                    # Z-score
                    rolling_mean = df[var].rolling(window).mean()
                    rolling_std = df[var].rolling(window).std()
                    df[f'{var}_zscore_{window}'] = (df[var] - rolling_mean) / (rolling_std + 1e-8)
                    
                    # Percentile rank
                    df[f'{var}_rank_{window}'] = df[var].rolling(window).apply(
                        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
                    )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        # Key interactions based on economic theory
        interactions = [
            ('real_interest_differential', 'risk_sentiment_score', 'multiply'),
            ('real_interest_differential', 'usd_brl_volatility_22d', 'divide'),
            ('brazilian_commodity_index', 'dxy_index', 'multiply'),
            ('carry_trade_attractiveness', 'risk_sentiment_score', 'multiply'),
            ('vix', 'dxy_index', 'multiply')
        ]
        
        for var1, var2, operation in interactions:
            if var1 in df.columns and var2 in df.columns:
                feature_name = f'{var1}_x_{var2}'
                
                if operation == 'multiply':
                    df[feature_name] = df[var1] * df[var2]
                elif operation == 'divide':
                    df[feature_name] = df[var1] / (df[var2] + 1e-8)
                elif operation == 'add':
                    df[feature_name] = df[var1] + df[var2]
                elif operation == 'subtract':
                    df[feature_name] = df[var1] - df[var2]
        
        return df
    
    def _normalize_percentile(self, series: pd.Series, window: int = 252) -> pd.Series:
        """
        Normalize series to 0-100 percentile scale.
        
        Args:
            series: Input series
            window: Rolling window for percentile calculation
            
        Returns:
            Normalized series
        """
        rolling_min = series.rolling(window, min_periods=window//4).min()
        rolling_max = series.rolling(window, min_periods=window//4).max()
        
        normalized = 100 * (series - rolling_min) / (rolling_max - rolling_min + 1e-8)
        return normalized.clip(0, 100)
    
    def clean_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate engineered features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Cleaned DataFrame
        """
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values (limited)
        df = df.fillna(method='ffill', limit=5)
        
        # Drop columns with too many missing values
        missing_threshold = 0.5
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.warning(f"Dropping columns with >50% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def get_feature_importance_estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate feature importance using correlation with target.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with feature importance estimates
        """
        if 'usd_brl_ptax_close' not in df.columns:
            return pd.DataFrame()
        
        # Calculate correlation with future returns (predictive power)
        target = df['usd_brl_ptax_close'].pct_change(5).shift(-5)  # 5-day forward returns
        
        importance = pd.DataFrame()
        importance['feature'] = df.columns
        importance['correlation'] = [abs(df[col].corr(target)) if col != 'usd_brl_ptax_close' else 0 
                                    for col in df.columns]
        importance['tier'] = 'tier4'  # Default
        
        # Assign tiers
        for tier, features in self.feature_groups.items():
            importance.loc[importance['feature'].isin(features), 'tier'] = tier
        
        # Sort by importance
        importance = importance.sort_values('correlation', ascending=False)
        
        return importance