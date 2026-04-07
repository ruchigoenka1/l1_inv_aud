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
    
    # Initialize Prophet with Weekly, Yearly, AND Monthly (Custom) seasonality
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    m.fit(m_df)
    forecast = m.predict(m_df)
    
    df['Trend'] = forecast['trend']
    df['Weekly_Effect'] = forecast['weekly']
    df['Monthly_Effect'] = forecast['monthly'] if 'monthly' in forecast else 0
    df['Yearly_Effect'] = forecast['yearly']
    df['Seasonal_Effect'] = forecast['additive_terms']
    
    # Classification Logic
    high_thresh = df['Seasonal_Effect'].quantile(0.75)
    low_thresh = df['Seasonal_Effect'].quantile(0.25)
    df['Season_Type'] = np.select(
        [(df['Seasonal_Effect'] >= high_thresh), (df['Seasonal_Effect'] <= low_thresh)],
        ['High Season', 'Low Season'], default='Normal'
    )
    
    return df, m

# ------------------------------------------------
# 4. Main UI Logic
# ------------------------------------------------
if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df = run_full_audit(raw_df, lead_time_manual, unit_val_input, holding_cost_pct, ordering_cost)
    
    t1, t2 = st.tabs(["📊 Performance Audit", "📈 AI Demand Analyzer"])

    with t1:
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

    with t2:
        st.header("📈 Hybrid Demand DNA Analysis")
        
        # --- HYBRID THRESHOLD SETTINGS ---
        st.subheader("📡 Pattern Significance Thresholds")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            unit_thresh = st.slider("Min. Unit Swing", 1, 50, 10, help="Minimum units for a pattern to matter.")
        with col_t2:
            pct_thresh = st.slider("Min. % of Average Demand", 1, 50, 10, help="Minimum percentage impact relative to average.")
        
        if st.button("🚀 Execute Multi-Seasonality Audit"):
            with st.spinner("Decomposing Trend, Weekly, Monthly, and Yearly signals..."):
                df, model = run_prophet_analysis(df)
                
                # Significance Logic
                weekly_swing = df['Weekly_Effect'].max() - df['Weekly_Effect'].min()
                avg_demand = df['Demand'].mean()
                pct_impact = (weekly_swing / avg_demand) * 100 if avg_demand > 0 else 0
                
                # Hybrid Gate: Pattern is significant if it passes BOTH (or EITHER, depending on preference)
                # Here we use BOTH for strictness
                is_significant = (weekly_swing >= unit_thresh) and (pct_impact >= pct_thresh)

                if is_significant:
                    st.success(f"✅ **SIGNIFICANT:** Weekly swing is {weekly_swing:.1f} units ({pct_impact:.1f}% of avg). This warrants a tailored ordering schedule.")
                else:
                    st.warning(f"⚠️ **INSIGNIFICANT:** Swing of {weekly_swing:.1f} units ({pct_impact:.1f}%) is too small to adjust operations. Stick to simple averages.")

                # A. Segment Stats Table
                st.subheader("Seasonal Segment Audit")
                stats = df.groupby('Season_Type').agg({'Demand': 'mean','Shortage': 'sum','IsStockout': 'sum','Closing Balance': 'mean'
                }).rename(columns={'Demand': 'Avg Demand', 'IsStockout': 'Stockout Days', 'Closing Balance': 'Avg Stock'})
                st.table(stats)

                # B. Weekly & Monthly Profile Charts
                c1, c2 = st.columns(2)
                with c1:
                    weekly_profile = df.copy()
                    weekly_profile['Day'] = weekly_profile['Date'].dt.day_name()
                    profile = weekly_profile.groupby('Day')['Weekly_Effect'].mean().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
                    fig_w = px.bar(profile, x='Day', y='Weekly_Effect', title="Weekly Pattern Profile",
                                   color='Weekly_Effect', color_continuous_scale='RdBu_r' if is_significant else ['#4A5568']*7)
                    st.plotly_chart(fig_w, use_container_width=True)

                with c2:
                    # Monthly seasonality (day of month 1-31)
                    df['DayOfMonth'] = df['Date'].dt.day
                    monthly_profile = df.groupby('DayOfMonth')['Monthly_Effect'].mean().reset_index()
                    fig_m = px.line(monthly_profile, x='DayOfMonth', y='Monthly_Effect', title="Monthly Pay-Cycle Pattern")
                    st.plotly_chart(fig_m, use_container_width=True)

        st.divider()

        # Risk Analysis (SL vs Max)
        st.subheader("Risk & Service Level Analysis")
        window_size = st.slider("Analysis Window (Days)", 1, 30, 7, key="risk_window")
        rolling_demand = df['Demand'].rolling(window=window_size).sum().dropna()
        
        if not rolling_demand.empty:
            target_sl = st.slider("Target Service Level", 0.80, 0.99, 0.95)
            sl_threshold = np.percentile(rolling_demand, target_sl * 100)
            max_val = rolling_demand.max()
            risk_gap = max_val - sl_threshold
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(f"{int(target_sl*100)}% SL Requirement", f"{int(sl_threshold)}")
            r2.metric("Max Demand (Window)", f"{int(max_val)}")
            r3.metric("The Risk Gap", f"{int(risk_gap)}", delta="Uncovered Units", delta_color="inverse")
            r4.metric("Exposure Value", f"${int(risk_gap * unit_val_input):,}")

            fig_risk = px.histogram(rolling_demand, nbins=20, title="Windowed Demand Distribution")
            fig_risk.add_vline(x=sl_threshold, line_dash="dash", line_color="yellow", annotation_text="SL Target")
            fig_risk.add_vline(x=max_val, line_dash="dot", line_color="red", annotation_text="Max")
            fig_risk.add_vrect(x0=sl_threshold, x1=max_val, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="RISK ZONE")
            st.plotly_chart(fig_risk, use_container_width=True)

else:
    st.info("👋 Dashboard ready. Please upload your Excel audit file to continue.")
