import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.api import VAR

@st.cache_resource(show_spinner="Fitting VAR model...")
def fit_var_model(_ts_df: pd.DataFrame, max_lags: int):
    """Fit VAR model with optimal lag selection."""
    var_res = VAR(_ts_df)
    lag_order_results = var_res.select_order(maxlags=max_lags)
    optimal_lag = lag_order_results.selected_orders.get('aic') or 1
    return var_res.fit(optimal_lag)

@st.cache_data(show_spinner="Computing FEVD...")
def compute_fevd(_var_fitted, horizon: int):
    """Compute Forecast Error Variance Decomposition."""
    fevd = _var_fitted.fevd(horizon)
    adj = fevd.decomp.mean(axis=1)
    adj_pct = pd.DataFrame(
        (adj * 100).round(1),
        index=_var_fitted.names,
        columns=_var_fitted.names
    )
    return adj, adj_pct

@st.cache_data(show_spinner="Computing lag criteria...")
def get_lag_selection_df(_ts_df: pd.DataFrame, max_lags: int):
    """Compute AIC, BIC, HQIC for each lag."""
    results = []
    for lag in range(1, max_lags + 1):
        try:
            model = VAR(_ts_df).fit(lag)
            results.append({"Lag": lag, "AIC": model.aic, "BIC": model.bic, "HQIC": model.hqic})
        except Exception:
            continue
    if not results:
        return None
    return pd.DataFrame(results).set_index("Lag")
