import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LogisticRegression

def calculate_features(df, smooth_window=3):
    """Compute rolling % changes and derivatives for key variables."""
    data = df.copy()
    data['TVL_pct'] = data['TVL'].pct_change().rolling(smooth_window).mean() * 100
    data['Stablecoin_pct'] = data['StablecoinIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['BankIndex_pct'] = data['BankIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['VIXC_pct'] = data['VIXC'].pct_change().rolling(smooth_window).mean() * 100
    data['BondYield_chg'] = data['BondYield'].diff()
    ts_df = data[['TVL_pct','Stablecoin_pct','BankIndex_pct','VIXC_pct','BondYield_chg']].dropna()
    return ts_df, data

def calculate_distress(data, lookahead_days=14):
    """Compute distress signal and biweekly logistic regression probabilities."""
    # Thresholds for distress
    tvl_thr = data['TVL_pct'].quantile(0.05)
    stable_thr = data['Stablecoin_pct'].quantile(0.05)
    bank_thr = data['BankIndex_pct'].quantile(0.05)
    vix_thr = data['VIXC_pct'].quantile(0.95)
    bond_thr = data['BondYield_chg'].quantile(0.95)

    data['Distress'] = (
        (data['TVL_pct'] <= tvl_thr) |
        (data['Stablecoin_pct'] <= stable_thr) |
        (data['BankIndex_pct'] <= bank_thr) |
        (data['VIXC_pct'] >= vix_thr) |
        (data['BondYield_chg'] >= bond_thr)
    ).astype(int)

    data['Target'] = data['Distress'].rolling(window=lookahead_days, min_periods=1)\
                        .max().shift(-lookahead_days+1).fillna(0).astype(int)

    feature_cols = []
    for var in ['TVL_pct','Stablecoin_pct','BankIndex_pct','VIXC_pct','BondYield_chg']:
        for lag in range(1, 4):
            col = f"{var}_lag{lag}"
            data[col] = data[var].shift(lag)
            feature_cols.append(col)

    clf_df = data.dropna(subset=feature_cols+['Target'])
    X, y = clf_df[feature_cols], clf_df['Target']

    if y.nunique() < 2 or len(y) < 10:
        print("Not enough target variation — adjust lookahead or smoothing.")
        return

    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:,1]
    prob_series = pd.Series(y_prob, index=X.index)
    biweekly = prob_series.resample('14D').mean()

    # Plot
    fig, ax = plt.subplots(figsize=(6.5,4.5))
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
    plt.show()
