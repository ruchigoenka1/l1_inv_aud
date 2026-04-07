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

def run_prophet_analysis(df):
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(m_df)
    
    # Predict on existing dates to get components
    forecast = m.predict(m_df)
    
    # Merge components back to original DF
    df['Trend'] = forecast['trend']
    df['Weekly_Effect'] = forecast['weekly']
    df['Yearly_Effect'] = forecast['yearly']
    df['Seasonal_Effect'] = forecast['additive_terms']
    
    # Segregation Logic
    high_thresh = df['Seasonal_Effect'].quantile(0.75)
    low_thresh = df['Seasonal_Effect'].quantile(0.25)
    conditions = [(df['Seasonal_Effect'] >= high_thresh), (df['Seasonal_Effect'] <= low_thresh)]
    df['Season_Type'] = np.select(conditions, ['High Season', 'Low Season'], default='Normal')
    
    return df, m, forecast

# ------------------------------------------------
# 4. Main UI Logic
# ------------------------------------------------
if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df = run_full_audit(raw_df, lead_time_manual, unit_val_input, holding_cost_pct, ordering_cost)
    
    t1, t2 = st.tabs(["📊 Performance Audit", "📈 AI Demand Analyzer"])

    with t1:
        # Metrics & Inventory Chart (Same as before)
        total_d = df['Demand'].sum()
        fill_rate = (1 - (df['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Stockout Days", int(df['IsStockout'].sum()))
        k2.metric("Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inv", f"{df['Closing Balance'].mean():.1f}")
        k4.metric("Avg WC", f"${(df['Closing Balance'].mean() * unit_val_input):,.0f}")
        k5.metric("Policy Cost", f"${(df['HoldingCost'].sum() + df['OrderingCost'].sum()):,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=np.where(df["InLT"], df["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="LT Window"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2.5)))
        placed = df[df["Order Placed"] > 0]
        if not placed.empty:
            fig.add_trace(go.Scatter(x=placed["Date"], y=placed["Closing Balance"], mode="markers", name="Order Placed", 
                                     marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
        fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

    with t2:
        st.header("📈 AI Demand DNA & Seasonal Segregation")
        
        if st.button("🚀 Run Prophet AI Analysis"):
            with st.spinner("Decoding weekly and yearly signals..."):
                df, model, forecast = run_prophet_analysis(df)
                
                # A. Segment Stats Table
                st.subheader("Inventory Metrics by Season Segment")
                stats = df.groupby('Season_Type').agg({'Demand': 'mean','Shortage': 'sum','IsStockout': 'sum','Closing Balance': 'mean'
                }).rename(columns={'Demand': 'Avg Demand', 'IsStockout': 'Stockout Days', 'Closing Balance': 'Avg Stock'})
                st.table(stats)

                # B. Trend vs Actual
                st.subheader("Signal Decomposition")
                fig_trend = px.line(df, x="Date", y=["Demand", "Trend"], title="Underlying Trend vs. Actuals",
                                    color_discrete_map={"Demand": "#4A5568", "Trend": "#F6AD55"})
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- NEW: WEEKLY & YEARLY GRAPHS ---
                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("**Weekly Seasonality Profile**")
                    # Extract one week of day-of-week effects
                    # Prophet stores the weekly component per date; we map to day names
                    weekly_df = df.copy()
                    weekly_df['Day'] = weekly_df['Date'].dt.day_name()
                    # Average the effect per day to get a clean profile
                    weekly_profile = weekly_df.groupby('Day')['Weekly_Effect'].mean().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    ).reset_index()
                    
                    fig_w = px.bar(weekly_profile, x='Day', y='Weekly_Effect', 
                                   title="Avg Demand Deviation by Day",
                                   color='Weekly_Effect', color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_w, use_container_width=True)

                with c2:
                    st.write("**Yearly Seasonality Profile**")
                    # Show the yearly wave over the dataset
                    fig_y = px.line(df, x="Date", y="Yearly_Effect", 
                                    title="Annual Demand Cycle (AI Prediction)",
                                    line_shape='spline', color_discrete_sequence=['#FFCC00'])
                    st.plotly_chart(fig_y, use_container_width=True)

        st.divider()

        # Risk Analysis
        st.subheader("Risk & Service Level Analysis")
        window_size = st.slider("Analysis Window (Days)", 1, 30, 7)
        rolling_demand = df['Demand'].rolling(window=window_size).sum().dropna()
        
        if not rolling_demand.empty:
            target_sl = st.slider("Target Service Level", 0.80, 0.99, 0.95)
            sl_threshold = np.percentile(rolling_demand, target_sl * 100)
            max_val = rolling_demand.max()
            risk_gap = max_val - sl_threshold
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(f"{int(target_sl*100)}% SL Stock", f"{int(sl_threshold)}")
            r2.metric("Max Observed", f"{int(max_val)}")
            r3.metric("The Risk Gap", f"{int(risk_gap)}", delta="Uncovered Units", delta_color="inverse")
            r4.metric("Risk Financials", f"${int(risk_gap * unit_val_input):,}")

            fig_risk = px.histogram(rolling_demand, nbins=20, title="Demand Probability Distribution")
            fig_risk.add_vline(x=sl_threshold, line_dash="dash", line_color="yellow", annotation_text="SL Target")
            fig_risk.add_vline(x=max_val, line_dash="dot", line_color="red", annotation_text="Absolute Max")
            fig_risk.add_vrect(x0=sl_threshold, x1=max_val, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="RISK ZONE")
            st.plotly_chart(fig_risk, use_container_width=True)

else:
    st.info("👋 Upload your Excel to begin the Audit.")
