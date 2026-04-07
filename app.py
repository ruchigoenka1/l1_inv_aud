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

def run_prophet_stratification_smoothed(df):
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(m_df)
    forecast = m.predict(m_df)
    
    # 1. Smoothing the seasonal signal (14-day window)
    df['Raw_Seasonal'] = forecast['additive_terms']
    df['Smoothed_Seasonal'] = df['Raw_Seasonal'].rolling(window=14, center=True).mean().ffill().bfill()
    
    # 2. Zone Classification (Top/Bottom 15%)
    high_v = df['Smoothed_Seasonal'].quantile(0.85)
    low_v = df['Smoothed_Seasonal'].quantile(0.15)
    
    conditions = [
        (df['Smoothed_Seasonal'] >= high_v),
        (df['Smoothed_Seasonal'] <= low_v)
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
        total_d = df['Demand'].sum()
        fill_rate = (1 - (df['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df['IsStockout'].sum()))
        k2.metric("Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inventory", f"{df['Closing Balance'].mean():.1f}")
        k4.metric("Policy Cost", f"${(df['HoldingCost'].sum() + df['OrderingCost'].sum()):,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=np.where(df["InLT"], df["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Lead Time Window"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2.5)))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.header("🕵️ Demand Stratification & Zone DNA")
        
        if st.button("🚀 Run AI Stratification"):
            with st.spinner("Decoding seasonality and smoothing zones..."):
                df = run_prophet_stratification_smoothed(df)
                
                # --- NEW: DEMAND LINE WITH ZONE BACKGROUNDS ---
                st.subheader("Daily Demand & Seasonal Zones")
                st.info("This graph shows your actual daily demand colored by the AI-detected 'Seasonal Zone'.")
                
                fig_demand = px.line(df, x='Date', y='Demand', 
                                     title="Historical Demand Overlaid with Seasonal Zones",
                                     color_discrete_sequence=['#FFFFFF']) # White line for contrast
                
                # Update line colors based on Zone for a 'Heatmap' effect
                fig_demand = px.scatter(df, x='Date', y='Demand', color='Zone',
                                        color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"},
                                        title="Demand Distribution Over Time (By Zone)")
                fig_demand.add_trace(go.Scatter(x=df['Date'], y=df['Demand'], line=dict(color='rgba(255,255,255,0.3)', width=1), showlegend=False))
                
                fig_demand.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_demand, use_container_width=True)

                # --- THE STRATIFIED TABLE ---
                st.subheader("📋 Segmented Demand Log")
                st.dataframe(df[["Date", "Demand", "Zone"]], use_container_width=True, height=350)

                # --- SEGMENTED HISTOGRAM ---
                st.divider()
                st.subheader("📊 Probability Distribution by Zone")
                fig_hist = px.histogram(
                    df, x="Demand", color="Zone", 
                    nbins=30, barmode='overlay', opacity=0.7,
                    color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"},
                    title="Comparison of Demand Volatility Across Zones"
                )
                fig_hist.update_layout(template="plotly_dark")
                st.plotly_chart(fig_hist, use_container_width=True)

                # --- SAFETY STOCK BY ZONE ---
                st.divider()
                st.subheader("🛡️ Tailored Safety Stock Strategy")
                zone_stats = df.groupby('Zone').agg({'Demand': ['mean', 'std', 'max']}).reset_index()
                zone_stats.columns = ['Zone', 'Avg Demand', 'Volatility (StdDev)', 'Absolute Max']
                zone_stats['Rec. Safety Stock'] = (1.65 * zone_stats['Volatility (StdDev)'] * np.sqrt(lt_manual)).round(0)
                st.table(zone_stats)
else:
    st.info("Upload your Excel audit file to begin.")
