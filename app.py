import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import io

# ------------------------------------------------
# 1. Page Config & Styling
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Inventory Auditor Pro")

st.markdown(
    """
    <style>
    .block-container { padding: 1.5rem 5rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #00CCFF !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; font-weight: bold !important; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #2E7D32 !important; color: white !important; }
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
# 3. Logic Engines
# ------------------------------------------------

def run_full_audit(df, lt, u_val, h_pct, o_cost):
    """Standardizes data and calculates core inventory metrics."""
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if "Order Placed" not in df.columns:
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    daily_h_rate = (h_pct / 100) / 365
    df['HoldingCost'] = df['Closing Balance'] * u_val * daily_h_rate
    df['OrderingCost'] = np.where(df['Order Placed'] > 0, o_cost, 0)
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    
    df['InLT'] = False
    for idx in df[df['Order Placed'] > 0].index:
        end_idx = min(idx + lt, len(df) - 1)
        df.loc[idx : end_idx, 'InLT'] = True
    
    return df

def run_prophet_and_segregate(df):
    """Uses Prophet to decompose signals and segregate data by seasonality."""
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(m_df)
    forecast = m.predict(m_df)
    
    # Extract components
    df['Trend'] = forecast['trend']
    df['Seasonal_Effect'] = forecast['additive_terms'] # Weekly + Yearly
    
    # Segregation Logic based on Seasonal Strength
    high_thresh = df['Seasonal_Effect'].quantile(0.75)
    low_thresh = df['Seasonal_Effect'].quantile(0.25)
    
    conditions = [
        (df['Seasonal_Effect'] >= high_thresh),
        (df['Seasonal_Effect'] <= low_thresh)
    ]
    df['Season_Type'] = np.select(conditions, ['High Season', 'Low Season'], default='Normal')
    
    return df, m

# ------------------------------------------------
# 4. Main UI Logic
# ------------------------------------------------
if uploaded_file:
    # 1. Initial Audit
    raw_df = pd.read_excel(uploaded_file)
    df = run_full_audit(raw_df, lead_time_manual, unit_val_input, holding_cost_pct, ordering_cost)
    
    t1, t2 = st.tabs(["📊 Performance Audit", "📈 AI Demand Analyzer"])

    # --- TAB 1: PERFORMANCE AUDIT ---
    with t1:
        total_d = df['Demand'].sum()
        fill_rate = (1 - (df['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Stockout Days", int(df['IsStockout'].sum()))
        k2.metric("Actual Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inventory", f"{df['Closing Balance'].mean():.1f}")
        k4.metric("Avg WC Investment", f"${(df['Closing Balance'].mean() * unit_val_input):,.0f}")
        k5.metric("Total Policy Cost", f"${(df['HoldingCost'].sum() + df['OrderingCost'].sum()):,.0f}")

        st.subheader("Inventory Flow & Lead Time Windows")
        fig = go.Figure()
        
        # Lead Time Shading
        fig.add_trace(go.Scatter(x=df["Date"], y=np.where(df["InLT"], df["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Lead Time Window"))
        
        # Physical Stock Line
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2.5)))
        
        # Order Placed Markers
        placed = df[df["Order Placed"] > 0]
        if not placed.empty:
            fig.add_trace(go.Scatter(x=placed["Date"], y=placed["Closing Balance"], mode="markers", name="Order Placed", 
                                     marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
        
        fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df, use_container_width=True)

    # --- TAB 2: DEMAND ANALYZER ---
    with t2:
        st.header("📈 AI Demand DNA & Seasonal Segregation")
        
        if st.button("🚀 Run Prophet AI Decomposition"):
            with st.spinner("Decomposing patterns and segregating seasons..."):
                df, model = run_prophet_and_segregate(df)
                
                # A. Seasonality Segregation Stats
                st.subheader("Inventory Metrics by Season Category")
                stats = df.groupby('Season_Type').agg({
                    'Demand': 'mean',
                    'Shortage': 'sum',
                    'IsStockout': 'sum',
                    'Closing Balance': 'mean'
                }).rename(columns={'Demand': 'Avg Demand', 'IsStockout': 'Stockout Days', 'Closing Balance': 'Avg Stock'})
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("**Segment Analysis**")
                    st.dataframe(stats.style.highlight_max(axis=0, color='#1B4D3E'))
                with c2:
                    fig_box = px.box(df, x="Season_Type", y="Demand", color="Season_Type", 
                                     color_discrete_map={"High Season": "#F56565", "Normal": "#63B3ED", "Low Season": "#4FD1C5"},
                                     title="Demand Distribution per Segment")
                    fig_box.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_box, use_container_width=True)

                st.divider()

                # B. Market DNA Components
                st.subheader("Signal Decomposition")
                fig_trend = px.line(df, x="Date", y=["Demand", "Trend"], title="Underlying Trend vs. Actuals",
                                    color_discrete_map={"Demand": "#4A5568", "Trend": "#F6AD55"})
                st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()

        # C. Risk Gap Analysis (The Request: SL vs Max)
        st.subheader("Risk & Service Level Analysis")
        window_size = st.slider("Analysis Window (Days)", 1, 30, 7)
        rolling_demand = df['Demand'].rolling(window=window_size).sum().dropna()
        
        if not rolling_demand.empty:
            target_sl = st.slider("Target Service Level", 0.80, 0.99, 0.95)
            sl_threshold = np.percentile(rolling_demand, target_sl * 100)
            max_val = rolling_demand.max()
            risk_gap = max_val - sl_threshold
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(f"{int(target_sl*100)}% SL Requirement", f"{int(sl_threshold)}")
            r2.metric("Max Observed (Worst Case)", f"{int(max_val)}")
            r3.metric("The Risk Gap", f"{int(risk_gap)} Units", delta="Uncovered", delta_color="inverse")
            r4.metric("Financial Exposure", f"${int(risk_gap * unit_val_input):,}")

            fig_risk = px.histogram(rolling_demand, nbins=20, title="Demand Probability Distribution", color_discrete_sequence=['#718096'])
            fig_risk.add_vline(x=sl_threshold, line_dash="dash", line_color="yellow", annotation_text="SL Target")
            fig_risk.add_vline(x=max_val, line_dash="dot", line_color="red", annotation_text="Absolute Max")
            fig_risk.add_vrect(x0=sl_threshold, x1=max_val, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="RISK ZONE")
            
            fig_risk.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
else:
    st.info("👋 Ready to Audit. Upload your Excel file with columns: Date, Opening Balance, Demand, Order Received, Closing Balance.")
