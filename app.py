import streamlit as st
import pandas as pd

from logic.data_loader import load_csv
from logic.stationarity import run_stationarity_tests
from logic.var_model import run_var_model
from logic.network import build_network
from plots.plot_stationarity import plot_stationarity_table
from plots.plot_var import plot_lag_selection, plot_residual_normality, plot_var_stability
from plots.plot_network import plot_network_graph
from utils.helpers import calculate_features, calculate_distress

st.title("DeFi & TradFi Contagion Risk Dashboard")

# ---- File uploader ----
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# ---- Load data ----
df = load_csv(uploaded_file)
st.success("File uploaded successfully!")
st.write(df.head())

# ---- Sidebar controls ----
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

# ---- Feature calculation ----
ts_df, data = calculate_features(df, smooth_window)

# ---- Stationarity Tests ----
stat_df = run_stationarity_tests(ts_df)

# ---- VAR Model ----
var_fitted, lag_df = run_var_model(ts_df, max_lags)

# ---- FEVD ----
adj_pct, adj = var_fitted.fevd(fevd_horizon).decomp.mean(axis=1) * 100, var_fitted.fevd(fevd_horizon).decomp.mean(axis=1)

# ---- Threshold & Network ----
G, title_label = build_network(adj, ts_df.columns.tolist(), threshold_mode, threshold_value, threshold_percentile)

# ---- Tabs ----
tabs = st.tabs([
    "FEVD Table", "Stationarity Tests", "Contagion Network",
    "Lag Selection", "Residual Normality", "VAR Stability", "Distress Probability"
])

# --- FEVD Table ---
with tabs[0]:
    st.subheader("FEVD Adjacency Table")
    df_table = pd.DataFrame(adj_pct, index=ts_df.columns, columns=ts_df.columns)
    st.dataframe(df_table)

# --- Stationarity Tests ---
with tabs[1]:
    st.subheader("Stationarity Tests (ADF & KPSS)")
    plot_stationarity_table(stat_df)

# --- Network Graph ---
with tabs[2]:
    st.subheader("Contagion Network Graph")
    plot_network_graph(G, fevd_horizon, title_label)

# --- Lag Selection ---
with tabs[3]:
    st.subheader("VAR Lag Selection Criteria")
    plot_lag_selection(lag_df)

# --- Residual Normality ---
with tabs[4]:
    st.subheader("Residual Normality (Jarque–Bera)")
    plot_residual_normality(var_fitted)

# --- VAR Stability ---
with tabs[5]:
    st.subheader("VAR Model Stability (Eigenvalues)")
    plot_var_stability(var_fitted)

# --- Distress Probability ---
with tabs[6]:
    st.subheader("Distress Probability (Biweekly)")
    calculate_distress(data, lookahead_days)
