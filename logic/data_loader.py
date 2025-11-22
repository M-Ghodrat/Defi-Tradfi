import pandas as pd

def load_csv(file):
    """Load CSV file and parse Date column."""
    df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
    df = df.asfreq('D')
    return df

