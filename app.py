import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import io

# ------------------------------------------------
# 1. Page Config & Logic
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Stratified Auditor")

def run_full_audit(df, lt, u_val, h_pct, o_cost):
    """Calculates physical flow and financial metrics."""
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
    
    # Lead Time shading logic
    df['InLT'] = False
    for idx in df[df['Order Placed'] > 0].index:
        end_idx = min(idx + lt, len(df) - 1)
        df.loc[idx : end_idx, 'InLT'] = True
    return df

def run_prophet_stratification(df):
    """Decomposes demand and labels zones (Peak/Low/Normal)."""
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(m_df)
    forecast = m.predict(m_df)
    
    df['Trend'] = forecast['trend']
    df['Seasonal_Effect'] = forecast['additive_terms']
    
    # Stratification: Top 20% = Peak, Bottom 20% = Low
    high_v = df['Seasonal_Effect'].quantile(0.80)
    low_v = df['Seasonal_Effect'].quantile(0.20)
    
    conditions = [
        (df['Seasonal_Effect'] >= high_v),
        (df['Seasonal_Effect'] <= low_v)
    ]
    df['Zone'] = np.select(conditions, ['Peak Season', 'Low Season'], default='Normal Season')
    return df

# ------------------------------------------------
# 2. Sidebar UI
# ------------------------------------------------
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Participant Excel", type=["xlsx"])
st.sidebar.divider()
unit_val = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
order_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lt_manual = st.sidebar.number_input("Lead Time (Days)", value=3, step=1)

# ------------------------------------------------
# 3. Main UI Layout
# ------------------------------------------------
if uploaded_file:
    raw_data = pd.read_excel(uploaded_file)
    df = run_full_audit(raw_data, lt_manual, unit_val, holding_pct, order_cost)
    
    t1, t2 = st.tabs(["📊 Inventory Performance", "🕵️ Demand Stratification"])

    with t1:
        # KPI Row
        total_d = df['Demand'].sum()
        fill_rate = (1 - (df['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df['IsStockout'].sum()))
        k2.metric("Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inventory", f"{df['Closing Balance'].mean():.1f}")
        k4.metric("Policy Cost", f"${(df['HoldingCost'].sum() + df['OrderingCost'].sum()):,.0f}")

        # Main Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=np.where(df["InLT"], df["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Lead Time Window"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2.5)))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.header("🕵️ Demand Stratification Analyzer")
        
        if st.button("🚀 Run AI Stratification & Zone Analysis"):
            with st.spinner("Extracting Seasonal DNA..."):
                df = run_prophet_stratification(df)
                
                # --- 1. THE TABLE ---
                st.subheader("📋 Segmented Demand Log")
                st.dataframe(df[["Date", "Demand", "Zone"]], use_container_width=True, height=300)

                # --- 2. SEGMENTED HISTOGRAM ---
                st.divider()
                st.subheader("📊 Demand Distribution by Seasonal Zone")
                fig_hist = px.histogram(
                    df, x="Demand", color="Zone", 
                    nbins=30, barmode='overlay', opacity=0.7,
                    color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"},
                    title="Comparison of Demand Probability Across Zones"
                )
                fig_hist.update_layout(template="plotly_dark")
                st.plotly_chart(fig_hist, use_container_width=True)

                # --- 3. SAFETY STOCK BY ZONE ---
                st.divider()
                st.subheader("🛡️ Tailored Safety Stock Strategy")
                
                # Calculate stats per zone
                zone_stats = df.groupby('Zone').agg({'Demand': ['mean', 'std', 'max']}).reset_index()
                zone_stats.columns = ['Zone', 'Avg Demand', 'Volatility (StdDev)', 'Absolute Max']
                
                # Formula: 1.65 (95% SL) * StdDev * SQRT(LeadTime)
                zone_stats['Rec. Safety Stock'] = (1.65 * zone_stats['Volatility (StdDev)'] * np.sqrt(lt_manual)).round(0)
                
                st.table(zone_stats)
                st.info("💡 Insight: Notice how the Volatility (StdDev) usually increases in the Peak Season, requiring a larger buffer.")

else:
    st.info("Upload your Excel audit file to begin.")
