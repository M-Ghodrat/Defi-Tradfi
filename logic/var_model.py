from statsmodels.tsa.api import VAR
import pandas as pd

def run_var_model(ts_df, max_lags=15):
    """Fit VAR model and compute lag selection criteria."""
    var_res = VAR(ts_df)
    lag_order = var_res.select_order(maxlags=max_lags)
    var_fitted = var_res.fit(lag_order.selected_orders['aic'] or 1)

    # Lag selection table
    lag_results = []
    for lag in range(1, max_lags+1):
        try:
            m = VAR(ts_df).fit(lag)
            lag_results.append({
                "Lag": lag,
                "AIC": m.aic,
                "BIC": m.bic,
                "HQIC": m.hqic
            })
        except:
            continue
    lag_df = pd.DataFrame(lag_results).set_index("Lag")
    return var_fitted, lag_df
