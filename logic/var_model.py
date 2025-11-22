from statsmodels.tsa.api import VAR
import numpy as np

def run_var(df):
    model = VAR(df)
    results = model.fit(maxlags=5)
    roots = np.real(results.roots)
    return results, roots
