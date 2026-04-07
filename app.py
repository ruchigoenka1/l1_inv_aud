import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------
# 1. Page Config & Styling
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Auditor Pro")

st.markdown(
    """
    <style>
    .block-container { padding: 2rem 5rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; font-weight: bold !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 2. Sidebar & Global Settings
# ------------------------------------------------
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Participant Excel", type=["xlsx"])

st.sidebar.divider()
st.sidebar.header("Financial & Policy Params")
unit_val_input = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lead_time_manual = st.sidebar.number_input("Standard Lead Time (Days)", value=3, step=1)

# ------------------------------------------------
# 3. Audit Engine (Fixed NameError)
# ------------------------------------------------
def run_full_audit(df, lt, u_val, h_pct, o_cost):
    # Normalize Columns
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Ensure Order Placed exists (Back-calculation logic)
    if "Order Placed" not in df.columns:
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    # Financials
    daily_h_rate = (h_pct / 100) / 365
    df['HoldingCost'] = df['Closing Balance'] * u_val * daily_h_rate
    df['OrderingCost'] = np.where(df['Order Placed'] > 0, o_cost, 0)
    
    # Service Metrics
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    
    # Lead Time Window Logic (for shading)
    df['InLT'] = False
    for idx in df[df['Order Placed'] > 0].index:
        end_idx = min(idx + lt, len(df) - 1)
        df.loc[idx : end_idx, 'InLT'] = True

    # Seasonality Classification (Peak/Normal/Low)
    avg_d = df['Demand'].mean()
    std_d = df['Demand'].std()
    
    # Avoid errors if std is 0
    if std_d == 0: std_d = 1
    
    conditions = [
        (df['Demand'] > (avg_d + std_d)), 
        (df['Demand'] < (avg_d - std_d))
    ]
    df['Seasonality'] = np.select(conditions, ['Peak', 'Low'], default='Normal')
    
    return df

# ------------------------------------------------
# 4. UI Tabs
# ------------------------------------------------
if uploaded_file:
    # Use the sidebar input variable correctly here
    df = run_full_audit(pd.read_excel(uploaded_file), lead_time_manual, unit_val_input, holding_cost_pct, ordering_cost)
    
    t1, t2 = st.tabs(["📊 Performance Audit", "📈 Demand Analyzer"])

    with t1:
        # Operational KPIs
        total_d = df['Demand'].sum()
        fill_rate = (1 - (df['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Stockout Days", int(df['IsStockout'].sum()))
        k2.metric("Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inventory", f"{df['Closing Balance'].mean():.1f}")
        k4.metric("Avg WC", f"${(df['Closing Balance'].mean() * unit_val_input):,.0f}")
        k5.metric("Total Cost", f"${(df['HoldingCost'].sum() + df['OrderingCost'].sum()):,.0f}")

        # Main Chart
        st.subheader("Inventory Levels & Lead Time Windows")
        fig = go.Figure()
        
        # Shading
        fig.add_trace(go.Scatter(x=df["Date"], y=np.where(df["InLT"], df["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="LT Window", showlegend=False))
        
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
        
        # Markers
        placed = df[df["Order Placed"] > 0]
        if not placed.empty:
            fig.add_trace(go.Scatter(x=placed["Date"], y=placed["Closing Balance"], mode="markers", name="Order Placed", 
                                     marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
        
        fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with t2:
        st.header("📈 Demand Analyzer")
        
        # 1. Seasonality Classification
        st.subheader("Demand Seasonality Classification")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("**Classification Logic:**")
            st.info("Peak: > Mean + 1 SD\n\nLow: < Mean - 1 SD")
            counts = df['Seasonality'].value_counts().reset_index()
            st.table(counts)
        with c2:
            fig_s = px.scatter(df, x="Date", y="Demand", color="Seasonality", 
                               color_discrete_map={"Peak": "#F56565", "Normal": "#63B3ED", "Low": "#4FD1C5"})
            fig_s.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_s, use_container_width=True)

        st.divider()
        
        # 2. Risk & Service Level Analysis
        st.subheader("Risk & Service Level Analysis")
        window_size = st.slider("Select Analysis Window (Days)", 1, 30, 7)
        rolling_demand = df['Demand'].rolling(window=window_size).sum().dropna()
        
        if not rolling_demand.empty:
            target_sl = st.slider("Target Service Level", 0.80, 0.99, 0.95)
            sl_threshold = np.percentile(rolling_demand, target_sl * 100)
            max_val = rolling_demand.max()
            risk_gap = max_val - sl_threshold
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(f"{int(target_sl*100)}% SL Requirement", f"{int(sl_threshold)}")
            r2.metric("Maximum Observed", f"{int(max_val)}")
            r3.metric("The Risk Gap", f"{int(risk_gap)}", delta="Uncovered Units", delta_color="inverse")
            r4.metric("Risk Financials", f"${int(risk_gap * unit_val_input):,}")

            fig_risk = px.histogram(rolling_demand, nbins=20, title="Demand Probability (Rolling Window)", color_discrete_sequence=['#718096'])
            fig_risk.add_vline(x=sl_threshold, line_dash="dash", line_color="yellow", annotation_text="SL Target")
            fig_risk.add_vline(x=max_val, line_dash="dot", line_color="red", annotation_text="Absolute Max")
            fig_risk.add_vrect(x0=sl_threshold, x1=max_val, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="RISK ZONE")
            
            fig_risk.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_risk, use_container_width=True)

else:
    st.info("👋 Please upload your Excel file to begin.")
