import pandas as pd

def calculate_features(df, smooth_window=3):
    data = df.copy()
    data['TVL_pct'] = data['TVL'].pct_change().rolling(smooth_window).mean() * 100
    data['Stablecoin_pct'] = data['StablecoinIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['BankIndex_pct'] = data['BankIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['VIXC_pct'] = data['VIXC'].pct_change().rolling(smooth_window).mean() * 100
    data['BondYield_chg'] = data['BondYield'].diff()
    ts_df = data[['TVL_pct','Stablecoin_pct','BankIndex_pct','VIXC_pct','BondYield_chg']].dropna()
    return ts_df, data
