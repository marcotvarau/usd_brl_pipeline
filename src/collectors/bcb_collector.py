"""
BCB (Banco Central do Brasil) Data Collector
Collects PTAX, SELIC, IPCA and other Brazilian economic indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import requests
import json
from time import sleep

from .base_collector import BaseCollector


class BCBCollector(BaseCollector):
    """
    Collector for Banco Central do Brasil data.
    
    Handles:
    - PTAX (official USD/BRL exchange rate)
    - SELIC (Brazilian interest rate)
    - IPCA (Brazilian inflation)
    - Focus market expectations
    - Other SGS time series
    """
    
    def _initialize(self) -> None:
        """Initialize BCB-specific settings."""
        self.base_url = self.config.get('apis', {}).get('bcb', {}).get('base_url', 
                                                                       'https://olinda.bcb.gov.br/olinda/servico')
        self.sgs_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
        
        # SGS series codes
        self.sgs_series = {
            'selic_daily': 11,           # Taxa SELIC diária
            'selic_target': 432,          # Meta SELIC
            'ipca_monthly': 433,          # IPCA mensal
            'igpm_monthly': 189,          # IGP-M mensal
            'industrial_production': 21859, # Produção industrial
            'unemployment': 24369,        # Taxa de desemprego
            'current_account': 23929,     # Conta corrente
            'fdi_net': 23936,            # Investimento direto líquido
            'portfolio_net': 23937,       # Investimento em carteira líquido
            'public_debt_gdp': 4536,      # Dívida pública % PIB
            'primary_result': 5793,       # Resultado primário
        }
        
        self.timeout = self.config.get('apis', {}).get('bcb', {}).get('timeout', 30)
        self.retry_attempts = self.config.get('apis', {}).get('bcb', {}).get('retry_attempts', 3)
    
    def collect(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        series: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect all BCB data for specified period.
        
        Args:
            start_date: Start date for collection
            end_date: End date for collection
            series: List of series to collect (None = all)
            
        Returns:
            DataFrame with all collected data
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache first
        cache_params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'series': series or 'all'
        }
        cache_key = self.get_cache_key(cache_params)
        
        cached_data = self.load_from_cache(cache_key, ttl_hours=1)
        if cached_data is not None:
            self.logger.info("Using cached BCB data")
            return cached_data
        
        # Collect data
        self.logger.info(f"Collecting BCB data from {start_date} to {end_date}")
        
        all_data = pd.DataFrame()
        
        # 1. Collect PTAX data
        try:
            ptax_data = self.get_ptax_data(start_date, end_date)
            if not ptax_data.empty:
                all_data = pd.concat([all_data, ptax_data], axis=1)
        except Exception as e:
            self.logger.error(f"Error collecting PTAX: {e}")
        
        # 2. Collect SELIC data
        try:
            selic_data = self.get_selic_data(start_date, end_date)
            if not selic_data.empty:
                all_data = pd.concat([all_data, selic_data], axis=1)
        except Exception as e:
            self.logger.error(f"Error collecting SELIC: {e}")
        
        # 3. Collect other SGS series
        if series is None:
            series = list(self.sgs_series.keys())
        
        for series_name in series:
            if series_name in self.sgs_series:
                try:
                    series_data = self.get_sgs_series(
                        self.sgs_series[series_name],
                        start_date,
                        end_date,
                        series_name
                    )
                    if not series_data.empty:
                        all_data = pd.concat([all_data, series_data], axis=1)
                except Exception as e:
                    self.logger.error(f"Error collecting {series_name}: {e}")
        
        # 4. Collect Focus expectations
        try:
            focus_data = self.get_focus_expectations(start_date, end_date)
            if not focus_data.empty:
                all_data = pd.concat([all_data, focus_data], axis=1)
        except Exception as e:
            self.logger.error(f"Error collecting Focus data: {e}")
        
        # Standardize and validate
        if not all_data.empty:
            all_data = self.standardize_dataframe(all_data)
            all_data = self.handle_missing_data(all_data, method='forward_fill')
            
            if self.validate_data(all_data):
                self.save_to_cache(cache_key, all_data)
            else:
                self.logger.warning("Data validation failed")
        
        return all_data
    
    def get_ptax_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get PTAX exchange rate data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with PTAX data
        """
        endpoint = f"{self.base_url}/PTAX/versao/v1/odata/CotacaoDolarPeriodo"
        
        params = {
            '@dataInicial': f"'{start_date.strftime('%m-%d-%Y')}'",
            '@dataFinal': f"'{end_date.strftime('%m-%d-%Y')}'",
            '$format': 'json',
            '$orderby': 'dataHoraCotacao'
        }
        
        response = self.make_request(endpoint, params=params, timeout=self.timeout)
        data = response.json()['value']
        
        if not data:
            self.logger.warning("No PTAX data received")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Process PTAX data
        df['dataHoraCotacao'] = pd.to_datetime(df['dataHoraCotacao'])
        df.set_index('dataHoraCotacao', inplace=True)
        
        # Rename columns to standard names
        df = df.rename(columns={
            'cotacaoCompra': 'usd_brl_bid',
            'cotacaoVenda': 'usd_brl_ask'
        })
        
        # Calculate mid price (PTAX)
        df['usd_brl_ptax_close'] = (df['usd_brl_bid'] + df['usd_brl_ask']) / 2
        
        # Keep only end-of-day values (último boletim)
        df = df.groupby(df.index.date).last()
        df.index = pd.to_datetime(df.index)
        
        return df[['usd_brl_bid', 'usd_brl_ask', 'usd_brl_ptax_close']]
    
    def get_selic_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get SELIC interest rate data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with SELIC data
        """
        # Get SELIC daily rate
        selic_daily = self.get_sgs_series(
            11,  # SELIC daily code
            start_date,
            end_date,
            'selic_rate'
        )
        
        # Get SELIC target rate
        selic_target = self.get_sgs_series(
            432,  # SELIC target code
            start_date,
            end_date,
            'selic_target'
        )
        
        # Combine both
        selic_data = pd.concat([selic_daily, selic_target], axis=1)
        
        return selic_data
    
    def get_sgs_series(
        self,
        series_code: int,
        start_date: datetime,
        end_date: datetime,
        series_name: str
    ) -> pd.DataFrame:
        """
        Get data from BCB SGS (Sistema Gerenciador de Séries).
        
        Args:
            series_code: SGS series code
            start_date: Start date
            end_date: End date
            series_name: Name for the series column
            
        Returns:
            DataFrame with series data
        """
        url = f"{self.sgs_url}/{series_code}/dados"
        
        params = {
            'formato': 'json',
            'dataInicial': start_date.strftime('%d/%m/%Y'),
            'dataFinal': end_date.strftime('%d/%m/%Y')
        }
        
        response = self.make_request(url, params=params, timeout=self.timeout)
        data = response.json()
        
        if not data:
            self.logger.warning(f"No data received for SGS series {series_code}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df.set_index('data', inplace=True)
        df.rename(columns={'valor': series_name}, inplace=True)
        
        # Convert to float
        df[series_name] = pd.to_numeric(df[series_name], errors='coerce')
        
        return df[[series_name]]
    
    def get_focus_expectations(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get Focus market expectations data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with Focus expectations
        """
        # Focus API endpoint
        focus_url = f"{self.base_url}/Expectativas/versao/v1/odata/ExpectativasMercadoMensais"
        
        # Focus indicators to collect
        indicators = ['Taxa de câmbio', 'IPCA', 'Selic', 'PIB Total']
        
        all_focus_data = []
        
        for indicator in indicators:
            params = {
                '$filter': f"Indicador eq '{indicator}' and Data ge '{start_date.strftime('%Y-%m-%d')}' and Data le '{end_date.strftime('%Y-%m-%d')}'",
                '$orderby': 'Data',
                '$format': 'json'
            }
            
            try:
                response = self.make_request(focus_url, params=params, timeout=self.timeout)
                data = response.json()['value']
                
                if data:
                    df = pd.DataFrame(data)
                    df['Data'] = pd.to_datetime(df['Data'])
                    df.set_index('Data', inplace=True)
                    
                    # Pivot to get median expectations
                    pivot_df = df.pivot_table(
                        values='Mediana',
                        index='Data',
                        columns='Indicador',
                        aggfunc='mean'
                    )
                    
                    # Rename columns
                    pivot_df.columns = [f"focus_{col.lower().replace(' ', '_')}" for col in pivot_df.columns]
                    all_focus_data.append(pivot_df)
                    
            except Exception as e:
                self.logger.error(f"Error getting Focus data for {indicator}: {e}")
            
            # Rate limiting
            sleep(0.5)
        
        if all_focus_data:
            focus_df = pd.concat(all_focus_data, axis=1)
            return focus_df
        
        return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate BCB data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        # Check for required columns
        required_cols = ['usd_brl_ptax_close', 'selic_rate']
        missing_cols = [col for col in required_cols if col in data.columns]
        
        if len(missing_cols) < len(required_cols):
            self.logger.warning(f"Some required columns present: {missing_cols}")
        
        # Check data ranges
        validations = []
        
        if 'usd_brl_ptax_close' in data.columns:
            valid_range = (1.0 <= data['usd_brl_ptax_close']) & (data['usd_brl_ptax_close'] <= 10.0)
            validations.append(valid_range.all())
            if not valid_range.all():
                self.logger.error("USD/BRL values out of range")
        
        if 'selic_rate' in data.columns:
            valid_range = (0.0 <= data['selic_rate']) & (data['selic_rate'] <= 25.0)
            validations.append(valid_range.all())
            if not valid_range.all():
                self.logger.error("SELIC values out of range")
        
        # Check for excessive missing data
        missing_pct = (data.isnull().sum() / len(data)) * 100
        excessive_missing = missing_pct[missing_pct > 10]
        
        if not excessive_missing.empty:
            self.logger.warning(f"Excessive missing data: {excessive_missing.to_dict()}")
        
        return all(validations) if validations else True
    
    def get_brazilian_fiscal_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get Brazilian fiscal indicators.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with fiscal data
        """
        fiscal_series = {
            'public_debt_gdp': 4536,      # Dívida pública % PIB
            'primary_result': 5793,        # Resultado primário
            'nominal_result': 5795,        # Resultado nominal
            'net_debt_gdp': 4513,         # Dívida líquida % PIB
        }
        
        fiscal_data = pd.DataFrame()
        
        for name, code in fiscal_series.items():
            try:
                series = self.get_sgs_series(code, start_date, end_date, name)
                if not series.empty:
                    fiscal_data = pd.concat([fiscal_data, series], axis=1)
            except Exception as e:
                self.logger.error(f"Error getting fiscal data {name}: {e}")
        
        return fiscal_data
    
    def get_capital_flows_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get capital flows data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with capital flows
        """
        flows_series = {
            'fdi_net': 23936,              # Investimento direto líquido
            'portfolio_equity': 23938,      # Investimento em ações
            'portfolio_debt': 23939,        # Investimento em renda fixa
            'derivatives': 23940,           # Derivativos
            'other_investments': 23941,     # Outros investimentos
        }
        
        flows_data = pd.DataFrame()
        
        for name, code in flows_series.items():
            try:
                series = self.get_sgs_series(code, start_date, end_date, name)
                if not series.empty:
                    flows_data = pd.concat([flows_data, series], axis=1)
            except Exception as e:
                self.logger.error(f"Error getting flows data {name}: {e}")
        
        # Calculate net capital flows
        if not flows_data.empty:
            flows_data['capital_flows_net'] = flows_data.sum(axis=1)
        
        return flows_data