"""
Quality Validator Module
Validates data quality for USD/BRL pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
# import great_expectations as ge
# from great_expectations.dataset import PandasDataset


class QualityValidator:
    """
    Comprehensive data quality validator.
    
    Performs:
    - Completeness checks
    - Consistency validation
    - Range validation
    - Statistical anomaly detection
    - Correlation validation
    - Time series specific checks
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize quality validator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load validation rules from config
        validation_config = config.get('validation', {})
        
        self.ranges = validation_config.get('ranges', {})
        self.max_missing_pct = validation_config.get('max_missing_pct', 5.0)
        self.required_columns = validation_config.get('required_columns', [])
        self.expected_correlations = validation_config.get('expected_correlations', [])
        
        # Statistical thresholds
        self.outlier_std_threshold = 4  # Number of standard deviations for outlier
        self.min_correlation = 0.1  # Minimum expected correlation
        self.max_correlation_change = 0.3  # Maximum correlation drift
        
        # Time series specific
        self.max_gap_days = 5  # Maximum allowed gap in time series
        self.min_data_points = 100  # Minimum required data points
        
        # Validation results storage
        self.validation_results = []
        self.issues_found = []
        self.metrics = {}
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete validation suite.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting data quality validation")
        
        # Reset results
        self.validation_results = []
        self.issues_found = []
        self.metrics = {}
        
        # Run validation checks
        checks = [
            ('completeness', self.check_completeness),
            ('data_types', self.check_data_types),
            ('ranges', self.check_ranges),
            ('outliers', self.check_outliers),
            ('duplicates', self.check_duplicates),
            ('time_series', self.check_time_series_integrity),
            ('correlations', self.check_correlations),
            ('statistical', self.check_statistical_properties),
            ('business_rules', self.check_business_rules),
            ('cross_validation', self.cross_validate_sources)
        ]
        
        for check_name, check_func in checks:
            self.logger.info(f"Running {check_name} check...")
            try:
                result = check_func(data)
                self.validation_results.append({
                    'check': check_name,
                    'passed': result['passed'],
                    'details': result
                })
                
                if not result['passed']:
                    self.issues_found.extend(result.get('issues', []))
                    
            except Exception as e:
                self.logger.error(f"Check {check_name} failed with error: {e}")
                self.validation_results.append({
                    'check': check_name,
                    'passed': False,
                    'error': str(e)
                })
                self.issues_found.append(f"{check_name}: {str(e)}")
        
        # Generate summary
        summary = self._generate_summary(data)
        
        return summary
    
    def check_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data completeness.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Completeness check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check if DataFrame is empty
        if data.empty:
            result['passed'] = False
            result['issues'].append("DataFrame is empty")
            return result
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            result['passed'] = False
            result['issues'].append(f"Insufficient data points: {len(data)} < {self.min_data_points}")
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            result['passed'] = False
            result['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check missing data percentages
        missing_pct = (data.isnull().sum() / len(data)) * 100
        
        for col, pct in missing_pct.items():
            if col in self.required_columns and pct > self.max_missing_pct:
                result['passed'] = False
                result['issues'].append(f"Column {col} has {pct:.2f}% missing data (max allowed: {self.max_missing_pct}%)")
        
        # Calculate completeness metrics
        result['metrics']['total_rows'] = len(data)
        result['metrics']['total_columns'] = len(data.columns)
        result['metrics']['overall_completeness'] = 100 - (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        result['metrics']['column_completeness'] = {
            col: 100 - pct for col, pct in missing_pct.items()
        }
        
        # Check for columns that are entirely null
        completely_null = data.columns[data.isnull().all()].tolist()
        if completely_null:
            result['passed'] = False
            result['issues'].append(f"Columns with all null values: {completely_null}")
        
        return result
    
    def check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data types consistency.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Data type check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Expected data types
        expected_types = {
            'usd_brl_ptax_close': 'float',
            'selic_rate': 'float',
            'fed_funds_rate': 'float',
            'dxy_index': 'float',
            'real_interest_differential': 'float',
            'risk_sentiment_score': 'float',
            'brazilian_commodity_index': 'float'
        }
        
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                
                if expected_type == 'float' and 'float' not in actual_type and 'int' not in actual_type:
                    result['passed'] = False
                    result['issues'].append(f"Column {col} has type {actual_type}, expected numeric")
                    
                    # Try to convert
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        result['issues'].append(f"Converted {col} to numeric")
                    except:
                        pass
        
        # Check datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            result['passed'] = False
            result['issues'].append("Index is not DatetimeIndex")
        
        result['metrics']['data_types'] = {
            col: str(dtype) for col, dtype in data.dtypes.items()
        }
        
        return result
    
    def check_ranges(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if values are within expected ranges.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Range check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        for col_pattern, range_dict in self.ranges.items():
            # Find matching columns
            matching_cols = [col for col in data.columns if col_pattern in col]
            
            for col in matching_cols:
                if col in data.columns and data[col].notna().any():
                    min_val = range_dict.get('min', -np.inf)
                    max_val = range_dict.get('max', np.inf)
                    
                    # Check range
                    out_of_range = data[col][(data[col] < min_val) | (data[col] > max_val)]
                    
                    if len(out_of_range) > 0:
                        pct_out = (len(out_of_range) / len(data[col].dropna())) * 100
                        
                        if pct_out > 1:  # Allow 1% outliers
                            result['passed'] = False
                            result['issues'].append(
                                f"Column {col} has {len(out_of_range)} values ({pct_out:.2f}%) outside range [{min_val}, {max_val}]"
                            )
                        
                        result['metrics'][f'{col}_range_violations'] = len(out_of_range)
                        result['metrics'][f'{col}_min_violation'] = out_of_range.min()
                        result['metrics'][f'{col}_max_violation'] = out_of_range.max()
        
        return result
    
    def check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for statistical outliers.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Outlier check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].notna().sum() < 10:  # Skip if too few values
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers_zscore = z_scores > self.outlier_std_threshold
            n_outliers_zscore = outliers_zscore.sum()
            
            # IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = (data[col] < lower_bound) | (data[col] > upper_bound)
            n_outliers_iqr = outliers_iqr.sum()
            
            # Store metrics
            result['metrics'][f'{col}_outliers_zscore'] = int(n_outliers_zscore)
            result['metrics'][f'{col}_outliers_iqr'] = int(n_outliers_iqr)
            
            # Check if excessive outliers
            pct_outliers = (n_outliers_iqr / len(data[col].dropna())) * 100
            
            if pct_outliers > 5:  # More than 5% outliers
                result['issues'].append(
                    f"Column {col} has {n_outliers_iqr} outliers ({pct_outliers:.2f}%) by IQR method"
                )
                # Don't fail for outliers, just warn
        
        return result
    
    def check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicate rows.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Duplicate check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check for duplicate indices (dates)
        duplicate_indices = data.index.duplicated()
        n_duplicate_indices = duplicate_indices.sum()
        
        if n_duplicate_indices > 0:
            result['passed'] = False
            result['issues'].append(f"Found {n_duplicate_indices} duplicate date indices")
            result['metrics']['duplicate_dates'] = data.index[duplicate_indices].tolist()
        
        # Check for fully duplicate rows
        duplicate_rows = data.duplicated()
        n_duplicate_rows = duplicate_rows.sum()
        
        if n_duplicate_rows > 0:
            result['issues'].append(f"Found {n_duplicate_rows} fully duplicate rows")
            result['metrics']['duplicate_rows'] = n_duplicate_rows
        
        # Check for partial duplicates on key columns
        key_cols = ['usd_brl_ptax_close', 'selic_rate', 'fed_funds_rate']
        key_cols_present = [col for col in key_cols if col in data.columns]
        
        if key_cols_present:
            partial_duplicates = data.duplicated(subset=key_cols_present)
            n_partial_duplicates = partial_duplicates.sum()
            
            if n_partial_duplicates > 0:
                result['issues'].append(
                    f"Found {n_partial_duplicates} rows with duplicate key values"
                )
                result['metrics']['partial_duplicates'] = n_partial_duplicates
        
        return result
    
    def check_time_series_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check time series specific integrity.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Time series check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check if index is sorted
        if not data.index.is_monotonic_increasing:
            result['passed'] = False
            result['issues'].append("Time series index is not sorted")
            data = data.sort_index()
        
        # Check for gaps in time series
        if isinstance(data.index, pd.DatetimeIndex):
            # Calculate expected frequency
            freq_counts = pd.Series(data.index[1:] - data.index[:-1]).value_counts()
            
            if len(freq_counts) > 0:
                most_common_freq = freq_counts.index[0]
                
                # Find gaps
                gaps = []
                for i in range(1, len(data.index)):
                    diff = data.index[i] - data.index[i-1]
                    if diff > most_common_freq + timedelta(days=self.max_gap_days):
                        gaps.append({
                            'start': data.index[i-1],
                            'end': data.index[i],
                            'gap_days': diff.days
                        })
                
                if gaps:
                    result['issues'].append(f"Found {len(gaps)} gaps in time series")
                    result['metrics']['gaps'] = gaps
                    
                    # Check if gaps are excessive
                    max_gap = max(g['gap_days'] for g in gaps)
                    if max_gap > 30:  # More than a month
                        result['passed'] = False
                        result['issues'].append(f"Maximum gap of {max_gap} days exceeds threshold")
            
            # Check date range
            date_range = data.index.max() - data.index.min()
            result['metrics']['date_range_days'] = date_range.days
            result['metrics']['start_date'] = str(data.index.min())
            result['metrics']['end_date'] = str(data.index.max())
            
            # Check for weekend/holiday patterns
            weekday_counts = pd.Series(data.index.weekday).value_counts()
            if 5 in weekday_counts or 6 in weekday_counts:
                weekend_pct = (weekday_counts.get(5, 0) + weekday_counts.get(6, 0)) / len(data) * 100
                result['metrics']['weekend_data_pct'] = weekend_pct
        
        return result
    
    def check_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check expected correlations between variables.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Correlation check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Calculate correlation matrix for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            result['metrics']['message'] = "Insufficient numeric columns for correlation analysis"
            return result
        
        corr_matrix = numeric_data.corr()
        
        # Check expected correlations
        for expected_corr in self.expected_correlations:
            col1 = expected_corr.get('columns', [None, None])[0]
            col2 = expected_corr.get('columns', [None, None])[1]
            expected_range = expected_corr.get('range', [-1, 1])
            
            if col1 in corr_matrix.index and col2 in corr_matrix.columns:
                actual_corr = corr_matrix.loc[col1, col2]
                
                if not (expected_range[0] <= actual_corr <= expected_range[1]):
                    result['issues'].append(
                        f"Correlation between {col1} and {col2} is {actual_corr:.3f}, "
                        f"expected [{expected_range[0]:.2f}, {expected_range[1]:.2f}]"
                    )
                    # Don't fail for correlation issues, just warn
                
                result['metrics'][f'corr_{col1}_{col2}'] = actual_corr
        
        # Check for perfect correlations (potential data leakage)
        perfect_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.95 and not np.isnan(corr_val):
                    perfect_corrs.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if perfect_corrs:
            result['issues'].append(f"Found {len(perfect_corrs)} near-perfect correlations (|r| > 0.95)")
            result['metrics']['perfect_correlations'] = perfect_corrs
        
        # Store correlation summary
        result['metrics']['mean_abs_correlation'] = np.abs(corr_matrix).mean().mean()
        
        return result
    
    def check_statistical_properties(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check statistical properties of the data.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Statistical check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Key columns to check
        key_cols = ['usd_brl_ptax_close', 'real_interest_differential', 'risk_sentiment_score']
        
        for col in key_cols:
            if col in data.columns and data[col].notna().sum() > 30:
                # Normality test (Shapiro-Wilk)
                sample = data[col].dropna()
                if len(sample) > 5000:
                    sample = sample.sample(5000)  # Limit sample size for test
                
                try:
                    stat, p_value = stats.shapiro(sample)
                    result['metrics'][f'{col}_normality_p'] = p_value
                    
                    if p_value < 0.01:
                        # Check if returns are normal (often more important)
                        returns = sample.pct_change().dropna()
                        if len(returns) > 30:
                            ret_stat, ret_p = stats.shapiro(returns)
                            result['metrics'][f'{col}_returns_normality_p'] = ret_p
                except:
                    pass
                
                # Stationarity test (simplified - checking if mean/std change over time)
                first_half = data[col].iloc[:len(data)//2].dropna()
                second_half = data[col].iloc[len(data)//2:].dropna()
                
                if len(first_half) > 10 and len(second_half) > 10:
                    # T-test for mean difference
                    t_stat, p_value = stats.ttest_ind(first_half, second_half)
                    
                    if p_value < 0.01:
                        result['issues'].append(
                            f"Column {col} shows non-stationarity (mean shift), p-value: {p_value:.4f}"
                        )
                    
                    result['metrics'][f'{col}_stationarity_p'] = p_value
                
                # Autocorrelation check
                if len(data[col].dropna()) > 100:
                    # Simple autocorrelation at lag 1
                    autocorr = data[col].autocorr(lag=1)
                    result['metrics'][f'{col}_autocorr_lag1'] = autocorr
                    
                    if abs(autocorr) > 0.95:
                        result['issues'].append(
                            f"Column {col} has very high autocorrelation ({autocorr:.3f})"
                        )
        
        return result
    
    def check_business_rules(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check business-specific rules.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Business rule check results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Business rule 1: USD/BRL should move inversely with real interest differential
        if 'usd_brl_ptax_close' in data.columns and 'real_interest_differential' in data.columns:
            correlation = data['usd_brl_ptax_close'].corr(data['real_interest_differential'])
            
            if correlation > 0.2:  # Should be negative or near zero
                result['issues'].append(
                    f"USD/BRL and interest differential have positive correlation ({correlation:.3f}), "
                    "which violates economic theory"
                )
        
        # Business rule 2: Volatility should be positive
        vol_cols = [col for col in data.columns if 'volatility' in col.lower()]
        for col in vol_cols:
            if col in data.columns:
                negative_vols = data[col][data[col] < 0]
                if len(negative_vols) > 0:
                    result['passed'] = False
                    result['issues'].append(f"Column {col} has {len(negative_vols)} negative values")
        
        # Business rule 3: Risk scores should be bounded
        if 'risk_sentiment_score' in data.columns:
            out_of_bounds = data['risk_sentiment_score'][
                (data['risk_sentiment_score'] < 0) | (data['risk_sentiment_score'] > 100)
            ]
            if len(out_of_bounds) > 0:
                result['passed'] = False
                result['issues'].append(
                    f"Risk sentiment score has {len(out_of_bounds)} values outside [0, 100]"
                )
        
        # Business rule 4: Interest rates should be reasonable
        rate_cols = ['selic_rate', 'fed_funds_rate']
        for col in rate_cols:
            if col in data.columns:
                unreasonable = data[col][(data[col] < -1) | (data[col] > 100)]
                if len(unreasonable) > 0:
                    result['passed'] = False
                    result['issues'].append(
                        f"{col} has {len(unreasonable)} unreasonable values (<-1% or >100%)"
                    )
        
        return result
    
    def cross_validate_sources(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Cross-validate data from different sources.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Cross-validation results
        """
        result = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # If we have USD/BRL from multiple sources, compare them
        usd_brl_cols = [col for col in data.columns if 'usd_brl' in col.lower() and 'close' in col.lower()]
        
        if len(usd_brl_cols) > 1:
            # Compare pairs
            for i in range(len(usd_brl_cols)):
                for j in range(i+1, len(usd_brl_cols)):
                    col1, col2 = usd_brl_cols[i], usd_brl_cols[j]
                    
                    # Calculate difference
                    diff = (data[col1] - data[col2]).abs()
                    mean_diff_pct = (diff / data[col1] * 100).mean()
                    
                    if mean_diff_pct > 1:  # More than 1% average difference
                        result['issues'].append(
                            f"Large discrepancy between {col1} and {col2}: "
                            f"average difference of {mean_diff_pct:.2f}%"
                        )
                    
                    result['metrics'][f'diff_{col1}_{col2}'] = mean_diff_pct
        
        # Check consistency between related indicators
        if 'vix' in data.columns and 'risk_sentiment_score' in data.columns:
            # These should be positively correlated
            correlation = data['vix'].corr(data['risk_sentiment_score'])
            if correlation < 0:
                result['issues'].append(
                    f"VIX and risk sentiment have negative correlation ({correlation:.3f}), "
                    "which is unexpected"
                )
            result['metrics']['vix_risk_correlation'] = correlation
        
        return result
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate validation summary.
        
        Args:
            data: Validated DataFrame
            
        Returns:
            Summary dictionary
        """
        # Count passed/failed checks
        passed_checks = sum(1 for r in self.validation_results if r.get('passed', False))
        failed_checks = len(self.validation_results) - passed_checks
        
        # Compile all metrics
        all_metrics = {}
        for result in self.validation_results:
            if 'details' in result and 'metrics' in result['details']:
                all_metrics.update(result['details']['metrics'])
        
        self.metrics = all_metrics
        
        summary = {
            'passed': len(self.issues_found) == 0,
            'total_checks': len(self.validation_results),
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'issues': self.issues_found,
            'metrics': all_metrics,
            'validation_timestamp': datetime.now().isoformat(),
            'data_shape': list(data.shape) if not data.empty else [0, 0],
            'date_range': {
                'start': str(data.index.min()) if not data.empty else None,
                'end': str(data.index.max()) if not data.empty else None
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Log summary
        self.logger.info(f"Validation complete: {passed_checks}/{len(self.validation_results)} checks passed")
        if self.issues_found:
            self.logger.warning(f"Issues found: {len(self.issues_found)}")
            for issue in self.issues_found[:5]:  # Log first 5 issues
                self.logger.warning(f"  - {issue}")
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze issues and generate recommendations
        issue_text = ' '.join(self.issues_found)
        
        if 'missing data' in issue_text.lower():
            recommendations.append("Consider implementing data imputation strategies for missing values")
        
        if 'outlier' in issue_text.lower():
            recommendations.append("Review outlier detection and consider robust scaling methods")
        
        if 'correlation' in issue_text.lower():
            recommendations.append("Investigate changes in market regime that may affect correlations")
        
        if 'duplicate' in issue_text.lower():
            recommendations.append("Implement deduplication in data collection pipeline")
        
        if 'gap' in issue_text.lower():
            recommendations.append("Backfill missing time periods to ensure continuous data")
        
        if 'range' in issue_text.lower():
            recommendations.append("Review data source quality and consider alternative sources")
        
        if not recommendations:
            recommendations.append("Data quality is good, continue with regular monitoring")
        
        return recommendations