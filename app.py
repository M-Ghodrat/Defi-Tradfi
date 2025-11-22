import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Circle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import jarque_bera
import matplotlib.dates as mdates
import streamlit as st
from sklearn.linear_model import LogisticRegression

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # ---- Load data ----
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        df = df.asfreq('D')
        st.success("File uploaded successfully!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Failed to read dataset: {e}")
        st.stop()

    # --- Helper: network visualization ---
    def draw_enhanced_network(G, fevd_horizon, threshold_label):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.axis('off')
        if len(G) == 0:
            return fig

        node_color = "#1f77b4"
        node_alpha = 0.7
        node_size = 600
        node_radius = np.sqrt(node_size)/50

        nodes_list = list(G.nodes())
        n = len(nodes_list)
        radius = 4
        angle_step = 2 * np.pi / n
        pos = {}
        for i, node in enumerate(nodes_list):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[node] = (x, y)

        def adjust_line(x1, y1, x2, y2, r):
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist == 0:
                return (x1, y1), (x2, y2)
            x1_new = x1 + dx/dist*r
            y1_new = y1 + dy/dist*r
            x2_new = x2 - dx/dist*r
            y2_new = y2 - dy/dist*r
            return (x1_new, y1_new), (x2_new, y2_new)

        for src, tgt in G.edges():
            (x1, y1), (x2, y2) = adjust_line(*pos[src], *pos[tgt], node_radius)
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='-|>',
                                    color='lightgray',
                                    linewidth=1.5,
                                    alpha=0.6,
                                    mutation_scale=12,
                                    zorder=1)
            ax.add_patch(arrow)

        for node, (x, y) in pos.items():
            circle = Circle((x, y), radius=node_radius,
                            color=node_color, alpha=node_alpha,
                            ec='black', lw=1.5, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node[0:2].upper(),
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', zorder=10)

        margin = 1.5
        ax.set_xlim(-radius-margin, radius+margin)
        ax.set_ylim(-radius-margin, radius+margin)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Contagion Network (FEVD={fevd_horizon} days)\n{threshold_label}',
                    fontsize=10, fontweight='bold', pad=10)
        return fig

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
    lookahead_days = st.sidebar.slider("Lookahead window (days) for distress calculation", 3, 30, 14)

    # --- Main dashboard ---
    st.title("DeFi & TradFi Contagion Risk Dashboard")

    # --- Feature calculation ---
    data = df.copy()
    data['TVL_pct'] = data['TVL'].pct_change().rolling(smooth_window).mean() * 100
    data['Stablecoin_pct'] = data['StablecoinIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['BankIndex_pct'] = data['BankIndex'].pct_change().rolling(smooth_window).mean() * 100
    data['VIXC_pct'] = data['VIXC'].pct_change().rolling(smooth_window).mean() * 100
    data['BondYield_chg'] = data['BondYield'].diff()
    ts_df = data[['TVL_pct','Stablecoin_pct','BankIndex_pct','VIXC_pct','BondYield_chg']].dropna()

    # --- Stationarity Tests ---
    stationarity_results = []
    for col in ts_df.columns:
        series = ts_df[col].dropna()
        adf_stat, adf_p = adfuller(series, autolag='AIC')[:2]
        try:
            kpss_stat, kpss_p = kpss(series, regression='c', nlags="auto")[:2]
        except:
            kpss_stat, kpss_p = np.nan, np.nan
        stationarity_results.append({
            "Variable": col,
            "ADF Statistic": round(adf_stat,3),
            "ADF p-value": round(adf_p,4),
            "KPSS Statistic": round(kpss_stat,3),
            "KPSS p-value": round(kpss_p,4)
        })
    stat_df = pd.DataFrame(stationarity_results)

    # --- VAR model ---
    var_res = VAR(ts_df)
    lag_order_results = var_res.select_order(maxlags=max_lags)
    var_fitted = var_res.fit(lag_order_results.selected_orders['aic'] or 1)  # fallback to 1

    # --- FEVD ---
    fevd = var_fitted.fevd(fevd_horizon)
    adj = fevd.decomp.mean(axis=1)
    adj_pct = (adj * 100).round(1)

    # --- Threshold ---
    if threshold_mode == 'percentile':
        threshold = np.percentile(adj, threshold_percentile)
        title_label = f'Threshold: Top {threshold_percentile}th Percentile'
    else:
        threshold = threshold_value
        title_label = f'Threshold: >{threshold*100:.1f}%'

    # --- Network ---
    nodes = ts_df.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for src_idx, src in enumerate(nodes):
        for tgt_idx, tgt in enumerate(nodes):
            if src_idx != tgt_idx and adj[tgt_idx, src_idx] > threshold:
                G.add_edge(src, tgt, weight=adj[tgt_idx, src_idx])

    # --- Tabs ---
    tabs = st.tabs(["FEVD Table", "Stationarity Tests", "Contagion Network", "Lag Selection", "Residual Normality", "VAR Stability", "Distress Probability"])

    # --- FEVD Table ---
    with tabs[0]:
        st.subheader("FEVD Adjacency Table")
        df_table = pd.DataFrame(adj_pct, index=ts_df.columns, columns=ts_df.columns)
        st.dataframe(df_table)

    # --- Stationarity Tests ---
    with tabs[1]:
        st.subheader("Stationarity Tests (ADF & KPSS)")
        st.dataframe(stat_df)

    # --- Network Graph ---
    with tabs[2]:
        st.subheader("Contagion Network Graph")
        fig = draw_enhanced_network(G, fevd_horizon, title_label)
        st.pyplot(fig)

    # --- VAR Lag Length Selection (AIC, BIC, HQIC) ---
    lag_order_results = []
    for lag in range(1, max_lags + 1):
        try:
            model = VAR(ts_df).fit(lag)
            lag_order_results.append({
                "Lag": lag,
                "AIC": model.aic,
                "BIC": model.bic,
                "HQIC": model.hqic
            })
        except Exception:
            continue

    if lag_order_results:
        lag_df = pd.DataFrame(lag_order_results).set_index("Lag")

        with tabs[3]:  # Assuming this is the Lag Selection tab
            st.subheader("Lag Selection Criteria")
            fig, ax = plt.subplots(figsize=(6.5, 3))
            ax.plot(lag_df.index, lag_df["AIC"], marker='o', label='AIC', linewidth=1.5)
            ax.plot(lag_df.index, lag_df["BIC"], marker='s', label='BIC', linewidth=1.5)
            ax.plot(lag_df.index, lag_df["HQIC"], marker='^', label='HQIC', linewidth=1.5)
            ax.set_xlabel("Lag Length", fontsize=9)
            ax.set_ylabel("Criterion Value", fontsize=9)
            ax.set_title("VAR Lag Selection Criteria", fontsize=10, fontweight='600')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)

    # --- Residual Normality ---
    with tabs[4]:
        st.subheader("Residual Normality (Jarque–Bera)")
        jb_results = []
        for col in var_fitted.resid.columns:
            stat, p_value = jarque_bera(var_fitted.resid[col])
            jb_results.append({'Variable': col, 'JB Statistic': stat, 'p-value': p_value, 'Normal?': p_value>0.05})
        jb_df = pd.DataFrame(jb_results)
        st.dataframe(jb_df)

    # --- VAR Stability ---
    with tabs[5]:
        st.subheader("VAR Model Stability (Eigenvalues)")
        eigvals = var_fitted.roots
        modulus = np.abs(eigvals)
        eig_df = pd.DataFrame({'Eigenvalue': eigvals, 'Modulus': modulus})
        eig_df['Stable?'] = modulus < 1
        st.dataframe(eig_df)
        if np.all(modulus < 1):
            st.success("VAR model is STABLE ✅")
        else:
            st.error("VAR model is NOT stable ❌")

    # --- Distress Probability ---
    with tabs[6]:
        st.subheader("Distress Probability (Biweekly)")
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
            st.warning("Not enough target variation — adjust lookahead or smoothing.")
        else:
            clf = LogisticRegression(solver='liblinear', max_iter=1000)
            clf.fit(X, y)
            y_prob = clf.predict_proba(X)[:,1]
            prob_series = pd.Series(y_prob, index=X.index)
            biweekly = prob_series.resample('14D').mean()

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
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()
