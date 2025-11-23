import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import jarque_bera

@st.cache_data(show_spinner="Running stationarity tests...")
def run_stationarity_tests(_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Run ADF and KPSS tests on all columns."""
    results = []
    for col in _ts_df.columns:
        series = _ts_df[col].dropna()
        adf_stat, adf_p = adfuller(series, autolag='AIC')[:2]
        try:
            kpss_stat, kpss_p = kpss(series, regression='c', nlags="auto")[:2]
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan
        
        results.append({
            "Variable": col,
            "ADF Statistic": round(adf_stat, 3),
            "ADF p-value": round(adf_p, 4),
            "KPSS Statistic": round(kpss_stat, 3) if not np.isnan(kpss_stat) else np.nan,
            "KPSS p-value": round(kpss_p, 4) if not np.isnan(kpss_p) else np.nan
        })
    return pd.DataFrame(results)

@st.cache_data(show_spinner="Testing residual normality...")
def run_residual_normality(_var_fitted) -> pd.DataFrame:
    """Run Jarque-Bera test on VAR residuals."""
    results = []
    for col in _var_fitted.resid.columns:
        stat, p_value = jarque_bera(_var_fitted.resid[col])
        results.append({
            'Variable': col,
            'JB Statistic': round(stat, 3),
            'p-value': round(p_value, 4),
            'Normal?': p_value > 0.05
        })
    return pd.DataFrame(results)
