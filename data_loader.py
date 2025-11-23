import pandas as pd
import streamlit as st
from io import StringIO

@st.cache_data(show_spinner="Loading data...")
def load_and_process_data(file_content: str, smooth_window: int):
    """Load CSV from string content and compute smoothed percentage changes.
    
    Note: We pass file content as string (not the UploadedFile object) 
    so it can be properly cached.
    """
    df = pd.read_csv(StringIO(file_content), parse_dates=['Date'], index_col='Date')
    df = df.asfreq('D')
    
    data = df.copy()
    data['TVL_pct'] = data['TVL'].pct_change().rolling(smooth_window).mean() * 100
    data['Stablecoin_pct'] = data['StablecoinIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['BankIndex_pct'] = data['BankIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['VIXC_pct'] = data['VIXC'].pct_change().rolling(smooth_window).mean() * 100
    data['BondYield_chg'] = data['BondYield'].diff()
    
    ts_df = data[['TVL_pct', 'Stablecoin_pct', 'BankIndex_pct', 'VIXC_pct', 'BondYield_chg']].dropna()
    
    return data, ts_df
