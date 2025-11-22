import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np

def run_stationarity_tests(ts_df):
    """Run ADF and KPSS tests for each column."""
    results = []
    for col in ts_df.columns:
        series = ts_df[col].dropna()
        adf_stat, adf_p = adfuller(series, autolag='AIC')[:2]
        try:
            kpss_stat, kpss_p = kpss(series, regression='c', nlags="auto")[:2]
        except:
            kpss_stat, kpss_p = np.nan, np.nan
        results.append({
            "Variable": col,
            "ADF Statistic": round(adf_stat,3),
            "ADF p-value": round(adf_p,4),
            "KPSS Statistic": round(kpss_stat,3),
            "KPSS p-value": round(kpss_p,4)
        })
    return pd.DataFrame(results)
