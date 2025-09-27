"""
FRED (Federal Reserve Economic Data) Collector
Collects US economic indicators from the St. Louis Fed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import fredapi
from time import sleep

from .base_collector import BaseCollector


class FREDCollector(BaseCollector):
    """
    Collector for Federal Reserve Economic Data.
    
    Handles:
    - Fed Funds Rate
    - US Treasury Yields
    - Inflation indicators (CPI, PCE)
    - Economic indicators (GDP, Unemployment)
    - Monetary aggregates (M1, M2)
    - Dollar indices
    """
    
    def _initialize(self) -> None:
        """Initialize FRED-specific settings."""
        fred_config = self.config.get('apis', {}).get('fred', {})
        
        self.api_key = fred_config.get('api_key')
        if not self.api_key:
            raise ValueError("FRED API key not found in configuration")
        
        # Initialize FRED client
        self.fred = fredapi.Fred(api_key=self.api_key)
        
        # Series to collect
        self.series_ids = fred_config.get('series_ids', {
            # Interest rates
            'fed_funds_rate': 'FEDFUNDS',          # Federal Funds Effective Rate
            'fed_funds_target': 'DFEDTARU',        # Federal Funds Target Rate (Upper)
            'fed_funds_target_lower': 'DFEDTARL',  # Federal Funds Target Rate (Lower)
            
            # Treasury yields
            'us_3m_treasury': 'DGS3MO',            # 3-Month Treasury
            'us_2y_treasury': 'DGS2',              # 2-Year Treasury
            'us_5y_treasury': 'DGS5',              # 5-Year Treasury
            'us_10y_treasury': 'DGS10',            # 10-Year Treasury
            'us_30y_treasury': 'DGS30',            # 30-Year Treasury
            
            # Inflation
            'us_cpi': 'CPIAUCSL',                  # CPI All Urban Consumers
            'us_core_cpi': 'CPILFESL',             # Core CPI (Less Food & Energy)
            'us_pce': 'PCEPI',                     # PCE Price Index
            'us_core_pce': 'PCEPILFE',             # Core PCE
            'us_cpi_yoy': 'CPIAUCSL',              # Will calculate YoY
            'us_5y_inflation_exp': 'T5YIE',        # 5-Year Breakeven Inflation Rate
            'us_10y_inflation_exp': 'T10YIE',      # 10-Year Breakeven Inflation Rate
            
            # Economic indicators
            'us_gdp': 'GDP',                       # Gross Domestic Product
            'us_gdp_growth': 'A191RL1Q225SBEA',    # GDP Growth Rate
            'us_unemployment': 'UNRATE',           # Unemployment Rate
            'us_jobless_claims': 'ICSA',           # Initial Claims
            'us_nonfarm_payrolls': 'PAYEMS',       # Nonfarm Payrolls
            'us_industrial_production': 'INDPRO',   # Industrial Production Index
            'us_capacity_utilization': 'TCU',      # Capacity Utilization
            'us_retail_sales': 'RSXFS',            # Retail Sales
            
            # Consumer indicators
            'us_consumer_sentiment': 'UMCSENT',    # U of Michigan Consumer Sentiment
            'us_consumer_confidence': 'CSCICP03USM665S',  # Consumer Confidence Index
            
            # Housing
            'us_housing_starts': 'HOUST',          # Housing Starts
            'us_case_shiller': 'CSUSHPISA',        # Case-Shiller Home Price Index
            
            # Money supply
            'us_m1': 'M1SL',                       # M1 Money Supply
            'us_m2': 'M2SL',                       # M2 Money Supply
            'us_monetary_base': 'BOGMBASE',        # Monetary Base
            
            # Trade and current account
            'us_trade_balance': 'BOPGSTB',         # Trade Balance
            'us_current_account': 'NETFI',         # Current Account
            
            # Financial conditions
            'us_financial_conditions': 'NFCI',     # Chicago Fed Financial Conditions Index
            'us_stress_index': 'STLFSI4',          # St. Louis Fed Financial Stress Index
            
            # Dollar and forex
            'us_dollar_index': 'DTWEXBGS',         # Trade Weighted Dollar Index (Broad)
            'us_dollar_major': 'DTWEXM',           # Trade Weighted Dollar Index (Major)
            'us_dollar_emerging': 'DTWEXEMEGS',    # Trade Weighted Dollar Index (Emerging)
            
            # Credit spreads
            'us_baa_spread': 'BAA10Y',             # Moody's BAA Corporate Bond Yield Spread
            'us_high_yield_spread': 'BAMLH0A0HYM2', # High Yield Spread
            'us_term_spread': 'T10Y2Y',            # 10Y-2Y Treasury Spread
            
            # Commodities (for reference)
            'wti_oil': 'DCOILWTICO',               # WTI Oil Price
            'gold_price': 'GOLDPMGBD228NLBM',      # Gold Price
        })
        
        self.timeout = fred_config.get('timeout', 30)
        
    def collect(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        series: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect FRED data for specified period.
        
        Args:
            start_date: Start date for collection
            end_date: End date for collection
            series: List of series to collect (None = all)
            
        Returns:
            DataFrame with collected data
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
            'series': series or 'all'
        }
        cache_key = self.get_cache_key(cache_params)
        
        cached_data = self.load_from_cache(cache_key, ttl_hours=6)
        if cached_data is not None:
            self.logger.info("Using cached FRED data")
            return cached_data
        
        # Determine series to collect
        if series is None:
            series_to_collect = self.series_ids
        else:
            series_to_collect = {k: v for k, v in self.series_ids.items() if k in series}
        
        self.logger.info(f"Collecting FRED data for {len(series_to_collect)} series")
        
        # Collect data
        all_data = pd.DataFrame()
        
        for name, series_id in series_to_collect.items():
            try:
                data = self._collect_series(series_id, start_date, end_date, name)
                if data is not None and not data.empty:
                    if all_data.empty:
                        all_data = data
                    else:
                        all_data = all_data.join(data, how='outer')
                    self.logger.debug(f"✓ Collected {name}: {len(data)} observations")
            except Exception as e:
                self.logger.error(f"✗ Failed to collect {name} ({series_id}): {e}")
                self.metrics['requests_failed'] += 1
            
            # Rate limiting
            sleep(0.5)  # FRED has generous limits but be respectful
        
        # Process collected data
        if not all_data.empty:
            all_data = self._process_fred_data(all_data)
            all_data = self.standardize_dataframe(all_data)
            
            # Validate
            if self.validate_data(all_data):
                self.save_to_cache(cache_key, all_data)
            else:
                self.logger.warning("FRED data validation failed")
        
        return all_data
    
    def _collect_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
        name: str
    ) -> Optional[pd.DataFrame]:
        """
        Collect a single FRED series.
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date
            name: Friendly name for series
            
        Returns:
            DataFrame with series data
        """
        try:
            self.metrics['requests_made'] += 1
            
            # Get series data
            series_data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            if series_data is None or series_data.empty:
                self.logger.debug(f"No data returned for {series_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(series_data, columns=[name])
            
            # Get series info for metadata
            try:
                series_info = self.fred.get_series_info(series_id)
                self.logger.debug(f"Series {name}: {series_info.get('units', 'Unknown units')}, "
                               f"Frequency: {series_info.get('frequency', 'Unknown')}")
            except:
                pass
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting {series_id}: {e}")
            return None
    
    def _process_fred_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance FRED data.
        
        Args:
            data: Raw FRED data
            
        Returns:
            Processed DataFrame
        """
        # Calculate year-over-year changes for key series
        if 'us_cpi' in data.columns:
            data['us_cpi_yoy'] = data['us_cpi'].pct_change(12) * 100  # Monthly data
            data['us_cpi_mom'] = data['us_cpi'].pct_change(1) * 100
        
        if 'us_core_cpi' in data.columns:
            data['us_core_cpi_yoy'] = data['us_core_cpi'].pct_change(12) * 100
            data['us_core_cpi_mom'] = data['us_core_cpi'].pct_change(1) * 100
        
        if 'us_pce' in data.columns:
            data['us_pce_yoy'] = data['us_pce'].pct_change(12) * 100
        
        if 'us_core_pce' in data.columns:
            data['us_core_pce_yoy'] = data['us_core_pce'].pct_change(12) * 100
        
        # Calculate real rates
        if 'fed_funds_rate' in data.columns and 'us_cpi_yoy' in data.columns:
            data['us_real_rate'] = data['fed_funds_rate'] - data['us_cpi_yoy']
        
        # Calculate term spreads
        if 'us_10y_treasury' in data.columns and 'us_2y_treasury' in data.columns:
            data['us_term_spread_10y2y'] = data['us_10y_treasury'] - data['us_2y_treasury']
            data['yield_curve_inverted'] = (data['us_term_spread_10y2y'] < 0).astype(int)
        
        if 'us_10y_treasury' in data.columns and 'us_3m_treasury' in data.columns:
            data['us_term_spread_10y3m'] = data['us_10y_treasury'] - data['us_3m_treasury']
        
        if 'us_30y_treasury' in data.columns and 'us_5y_treasury' in data.columns:
            data['us_term_spread_30y5y'] = data['us_30y_treasury'] - data['us_5y_treasury']
        
        # Calculate inflation expectations
        if 'us_10y_treasury' in data.columns and 'us_10y_inflation_exp' in data.columns:
            # Real yield = Nominal - Inflation Expectation
            data['us_10y_real_yield'] = data['us_10y_treasury'] - data['us_10y_inflation_exp']
        
        # Calculate money supply growth
        if 'us_m2' in data.columns:
            data['us_m2_yoy'] = data['us_m2'].pct_change(12) * 100
            data['us_m2_3m_annualized'] = data['us_m2'].pct_change(3) * 4 * 100
        
        # Calculate employment momentum
        if 'us_nonfarm_payrolls' in data.columns:
            data['us_payrolls_change'] = data['us_nonfarm_payrolls'].diff()
            data['us_payrolls_3m_avg'] = data['us_payrolls_change'].rolling(3).mean()
        
        # GDP indicators
        if 'us_gdp' in data.columns:
            data['us_gdp_yoy'] = data['us_gdp'].pct_change(4) * 100  # Quarterly data
        
        # Financial conditions
        if 'us_financial_conditions' in data.columns:
            data['financial_conditions_tight'] = (data['us_financial_conditions'] > 0).astype(int)
        
        # Dollar strength momentum
        if 'us_dollar_index' in data.columns:
            data['dollar_momentum_1m'] = data['us_dollar_index'].pct_change(22) * 100
            data['dollar_momentum_3m'] = data['us_dollar_index'].pct_change(66) * 100
        
        # Recession indicators
        data = self._calculate_recession_indicators(data)
        
        return data
    
    def _calculate_recession_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recession probability indicators.
        
        Args:
            data: DataFrame with economic data
            
        Returns:
            DataFrame with recession indicators
        """
        recession_score = pd.Series(0, index=data.index, dtype=float)
        
        # 1. Yield curve inversion
        if 'yield_curve_inverted' in data.columns:
            recession_score += data['yield_curve_inverted'] * 0.3
        
        # 2. Unemployment rate increase (Sahm Rule simplified)
        if 'us_unemployment' in data.columns:
            unemployment_3m_avg = data['us_unemployment'].rolling(3).mean()
            unemployment_12m_min = data['us_unemployment'].rolling(12).min()
            sahm_indicator = unemployment_3m_avg - unemployment_12m_min
            recession_score += (sahm_indicator > 0.5).astype(float) * 0.2
        
        # 3. Industrial production decline
        if 'us_industrial_production' in data.columns:
            ip_yoy = data['us_industrial_production'].pct_change(12)
            recession_score += (ip_yoy < 0).astype(float) * 0.15
        
        # 4. Financial stress
        if 'us_stress_index' in data.columns:
            stress_high = data['us_stress_index'] > data['us_stress_index'].rolling(252).quantile(0.8)
            recession_score += stress_high.astype(float) * 0.15
        
        # 5. Credit spread widening
        if 'us_baa_spread' in data.columns:
            spread_high = data['us_baa_spread'] > data['us_baa_spread'].rolling(252).quantile(0.8)
            recession_score += spread_high.astype(float) * 0.1
        
        # 6. Consumer sentiment decline
        if 'us_consumer_sentiment' in data.columns:
            sentiment_low = data['us_consumer_sentiment'] < data['us_consumer_sentiment'].rolling(252).quantile(0.2)
            recession_score += sentiment_low.astype(float) * 0.1
        
        data['recession_probability_score'] = recession_score.clip(0, 1)
        data['recession_risk_high'] = (data['recession_probability_score'] > 0.5).astype(int)
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate FRED data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        validations = []
        
        # Check key series
        key_series = ['fed_funds_rate', 'us_cpi', 'us_unemployment']
        for series in key_series:
            if series in data.columns:
                # Check for reasonable values
                if series == 'fed_funds_rate':
                    valid = (data[series] >= -1) & (data[series] <= 20)
                elif series == 'us_unemployment':
                    valid = (data[series] >= 0) & (data[series] <= 30)
                elif series == 'us_cpi':
                    valid = (data[series] > 0) & (data[series] < 1000)
                
                if not valid.all():
                    invalid_count = (~valid).sum()
                    self.logger.warning(f"{series} has {invalid_count} values out of expected range")
                
                validations.append(valid.mean() > 0.95)  # Allow 5% anomalies
        
        # Check for data staleness
        if data.index.max() < datetime.now() - timedelta(days=30):
            self.logger.warning("Data appears stale (last update > 30 days ago)")
        
        # Check data coverage
        coverage = data.notna().sum() / len(data)
        low_coverage = coverage[coverage < 0.1]
        if not low_coverage.empty:
            self.logger.warning(f"Low data coverage (<10%) for: {low_coverage.index.tolist()}")
        
        return all(validations) if validations else True
    
    def get_series_metadata(self, series_id: str) -> Dict[str, Any]:
        """
        Get metadata for a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series metadata
        """
        try:
            info = self.fred.get_series_info(series_id)
            
            return {
                'id': series_id,
                'title': info.get('title'),
                'units': info.get('units'),
                'frequency': info.get('frequency'),
                'seasonal_adjustment': info.get('seasonal_adjustment'),
                'last_updated': info.get('last_updated'),
                'observation_start': info.get('observation_start'),
                'observation_end': info.get('observation_end'),
                'popularity': info.get('popularity'),
                'notes': info.get('notes')
            }
        except Exception as e:
            self.logger.error(f"Error getting metadata for {series_id}: {e}")
            return {}
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for FRED series.
        
        Args:
            search_text: Text to search
            limit: Maximum results
            
        Returns:
            List of series information
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            
            series_list = []
            for idx, row in results.iterrows():
                series_list.append({
                    'id': idx,
                    'title': row.get('title'),
                    'units': row.get('units'),
                    'frequency': row.get('frequency'),
                    'popularity': row.get('popularity')
                })
            
            return series_list
            
        except Exception as e:
            self.logger.error(f"Error searching series: {e}")
            return []
    
    def get_release_dates(self, series_id: str) -> pd.DataFrame:
        """
        Get release dates for a series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            DataFrame with release dates
        """
        try:
            releases = self.fred.get_series_all_releases(series_id)
            return releases
        except Exception as e:
            self.logger.error(f"Error getting release dates for {series_id}: {e}")
            return pd.DataFrame()