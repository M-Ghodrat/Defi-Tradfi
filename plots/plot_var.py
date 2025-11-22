import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.stats import jarque_bera
import numpy as np

def plot_lag_selection(lag_df):
    fig, ax = plt.subplots(figsize=(6.5,3))
    ax.plot(lag_df.index, lag_df["AIC"], marker='o', label='AIC', linewidth=1.5)
    ax.plot(lag_df.index, lag_df["BIC"], marker='s', label='BIC', linewidth=1.5)
    ax.plot(lag_df.index, lag_df["HQIC"], marker='^', label='HQIC', linewidth=1.5)
    ax.set_xlabel("Lag Length")
    ax.set_ylabel("Criterion Value")
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    st.pyplot(fig)

def plot_residual_normality(var_fitted):
    jb_results = []
    for col in var_fitted.resid.columns:
        stat, p = jarque_bera(var_fitted.resid[col])
        jb_results.append({'Variable': col, 'JB Statistic': stat, 'p-value': p, 'Normal?': p>0.05})
    st.dataframe(pd.DataFrame(jb_results))

def plot_var_stability(var_fitted):
    eigvals = var_fitted.roots
    modulus = np.abs(eigvals)
    eig_df = pd.DataFrame({'Eigenvalue': eigvals, 'Modulus': modulus})
    eig_df['Stable?'] = modulus < 1
    st.dataframe(eig_df)
    if np.all(modulus < 1):
        st.success("VAR model is STABLE ✅")
    else:
        st.error("VAR model is NOT stable ❌")
