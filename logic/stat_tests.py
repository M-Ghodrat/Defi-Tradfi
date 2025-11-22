from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series):
    stat, p, *_ = adfuller(series)
    return stat, p

def kpss_test(series):
    stat, p, *_ = kpss(series, nlags="auto")
    return stat, p
