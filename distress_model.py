import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from sklearn.linear_model import LogisticRegression

@st.cache_data(show_spinner="Computing distress probability...")
def compute_distress_probability(_data: pd.DataFrame, lookahead_days: int):
    """Compute and plot biweekly distress probability."""
    data = _data.copy()
    
    # Define thresholds
    tvl_thr = data['TVL_pct'].quantile(0.05)
    stable_thr = data['Stablecoin_pct'].quantile(0.05)
    bank_thr = data['BankIndex_pct'].quantile(0.05)
    vix_thr = data['VIXC_pct'].quantile(0.95)
    bond_thr = data['BondYield_chg'].quantile(0.95)
    
    # Distress indicator
    data['Distress'] = (
        (data['TVL_pct'] <= tvl_thr) |
        (data['Stablecoin_pct'] <= stable_thr) |
        (data['BankIndex_pct'] <= bank_thr) |
        (data['VIXC_pct'] >= vix_thr) |
        (data['BondYield_chg'] >= bond_thr)
    ).astype(int)
    
    data['Target'] = (data['Distress']
                      .rolling(window=lookahead_days, min_periods=1)
                      .max()
                      .shift(-lookahead_days + 1)
                      .fillna(0)
                      .astype(int))
    
    # Create lagged features
    feature_cols = []
    for var in ['TVL_pct', 'Stablecoin_pct', 'BankIndex_pct', 'VIXC_pct', 'BondYield_chg']:
        for lag in range(1, 4):
            col = f"{var}_lag{lag}"
            data[col] = data[var].shift(lag)
            feature_cols.append(col)
    
    clf_df = data.dropna(subset=feature_cols + ['Target'])
    X, y = clf_df[feature_cols], clf_df['Target']
    
    if y.nunique() < 2 or len(y) < 10:
        st.warning("Not enough target variation — adjust lookahead or smoothing.")
        return None
    
    # Fit model
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    
    prob_series = pd.Series(y_prob, index=X.index)
    biweekly = prob_series.resample('14D').mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(biweekly.index, 0, biweekly.values, alpha=0.25)
    ax.plot(biweekly.index, biweekly.values, marker='o', linewidth=1.6, markersize=5,
            markerfacecolor='white', markeredgewidth=1.5)
    ax.axhline(0.25, linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axhline(0.5, linestyle='--', linewidth=1.2, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=10, fontweight='600')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=9)
    ax.set_title(f'Lookahead = {lookahead_days} days', fontsize=11, fontweight='700', pad=8)
    ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    return fig
