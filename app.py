import streamlit as st

st.set_page_config(page_title="DeFi & TradFi Contagion Dashboard", layout="wide")

from data_loader import load_and_process_data
from var_analysis import fit_var_model, compute_fevd, get_lag_selection_df
from network_viz import build_network, draw_enhanced_network
from distress_model import compute_distress_probability
from stationarity import run_stationarity_tests, run_residual_normality

# --- Sidebar controls ---
st.sidebar.header("📊 Model Parameters")
smooth_window = st.sidebar.slider("Rolling smoothing window (days)", 1, 10, 3)
max_lags = st.sidebar.slider("VAR model maximum number of lags", 1, 40, 15)
fevd_horizon = st.sidebar.slider("FEVD horizon (days ahead)", 5, 60, 20)

st.sidebar.header("⚙️ Network Threshold")
threshold_mode = st.sidebar.selectbox("Threshold mode", ['fixed', 'percentile'])
threshold_value = st.sidebar.slider("Fixed threshold value", 0.005, 0.1, 0.02, 0.005)
threshold_percentile = st.sidebar.slider("Percentile threshold (%)", 50, 99, 95)

st.sidebar.header("📈 Distress Prediction")
lookahead_days = st.sidebar.slider("Lookahead window (days)", 3, 30, 14)

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# --- Load & Process Data (CACHED) ---
# Read file content as string so it can be cached properly
file_content = uploaded_file.getvalue().decode("utf-8")
data, ts_df = load_and_process_data(file_content, smooth_window)
st.success("File uploaded and processed!")

# --- Tabs ---
tabs = st.tabs([
    "FEVD Table", "Stationarity Tests", "Contagion Network", 
    "Lag Selection", "Residual Normality", "VAR Stability", "Distress Probability"
])

# Only compute what's needed per tab (lazy evaluation)
with tabs[0]:  # FEVD Table
    st.subheader("FEVD Adjacency Table")
    var_fitted = fit_var_model(ts_df, max_lags)
    adj, adj_pct = compute_fevd(var_fitted, fevd_horizon)
    st.dataframe(adj_pct)

with tabs[1]:  # Stationarity
    st.subheader("Stationarity Tests (ADF & KPSS)")
    stat_df = run_stationarity_tests(ts_df)
    st.dataframe(stat_df)

with tabs[2]:  # Network
    st.subheader("Contagion Network Graph")
    var_fitted = fit_var_model(ts_df, max_lags)
    adj, _ = compute_fevd(var_fitted, fevd_horizon)
    threshold = threshold_value if threshold_mode == 'fixed' else None
    percentile = threshold_percentile if threshold_mode == 'percentile' else None
    G, title_label = build_network(ts_df.columns.tolist(), adj, threshold, percentile)
    fig = draw_enhanced_network(G, fevd_horizon, title_label)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

with tabs[3]:  # Lag Selection
    st.subheader("Lag Selection Criteria")
    lag_df = get_lag_selection_df(ts_df, max_lags)
    if lag_df is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ax.plot(lag_df.index, lag_df["AIC"], marker='o', label='AIC')
        ax.plot(lag_df.index, lag_df["BIC"], marker='s', label='BIC')
        ax.plot(lag_df.index, lag_df["HQIC"], marker='^', label='HQIC')
        ax.set_xlabel("Lag Length")
        ax.set_ylabel("Criterion Value")
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        st.pyplot(fig)

with tabs[4]:  # Residual Normality
    st.subheader("Residual Normality (Jarque–Bera)")
    var_fitted = fit_var_model(ts_df, max_lags)
    jb_df = run_residual_normality(var_fitted)
    st.dataframe(jb_df)

with tabs[5]:  # VAR Stability
    st.subheader("VAR Model Stability (Eigenvalues)")
    var_fitted = fit_var_model(ts_df, max_lags)
    import numpy as np
    import pandas as pd
    eigvals = var_fitted.roots
    modulus = np.abs(eigvals)
    eig_df = pd.DataFrame({'Eigenvalue': eigvals, 'Modulus': modulus, 'Stable?': modulus < 1})
    st.dataframe(eig_df)
    if np.all(modulus < 1):
        st.success("VAR model is STABLE ✅")
    else:
        st.error("VAR model is NOT stable ❌")

with tabs[6]:  # Distress
    st.subheader("Distress Probability (Biweekly)")
    fig = compute_distress_probability(data, lookahead_days)
    if fig:
        st.pyplot(fig)
