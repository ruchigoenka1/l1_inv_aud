import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm
import io

# ------------------------------------------------
# 1. Page Config & Logic
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Stratified Auditor Pro")

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
    
    df['Trend'] = forecast['trend']
    df['Raw_Seasonal'] = forecast['additive_terms']
    df['Smoothed_Seasonal'] = df['Raw_Seasonal'].rolling(window=14, center=True).mean().ffill().bfill()
    
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
    
    t1, t2 = st.tabs(["📊 Performance Audit", "🕵️ Demand DNA & Risk"])

    with t1:
        # Standard Performance Metrics
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
        st.header("🕵️ Demand DNA & Stratification")
        
        if st.button("🚀 Run AI Stratification"):
            df = run_prophet_stratification_smoothed(df)
            st.session_state['stratified_df'] = df

        if 'stratified_df' in st.session_state:
            df = st.session_state['stratified_df']

            # --- 1. TREND & ZONE GRAPH ---
            st.subheader("Daily Demand, Zones & AI Trend")
            fig_demand = px.scatter(df, x='Date', y='Demand', color='Zone',
                                    color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"})
            fig_demand.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], mode='lines', name='Growth Trend', line=dict(color='#FFA500', width=2, dash='dot')))
            fig_demand.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_demand, use_container_width=True)

            # --- 2. RISK CONTROLS ---
            st.divider()
            st.subheader("🛡️ Risk & Safety Stock Simulation")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                analysis_window = st.slider("Lead Time Window (Days)", 1, 30, lt_manual)
            with r_col2:
                target_sl = st.slider("Target Service Level (%)", 80, 99, 95) / 100

            # Calculation for Rolling Window
            df['Rolling_Demand'] = df['Demand'].rolling(window=analysis_window).sum()
            
            # --- 3. ROLLING WINDOW GRAPH ---
            st.subheader(f"Total Demand over {analysis_window}-Day Windows")
            fig_roll = px.line(df, x='Date', y='Rolling_Demand', title=f"Rolling Sum of Demand ({analysis_window} Days)", color_discrete_sequence=['#00FFCC'])
            st.plotly_chart(fig_roll, use_container_width=True)

            # --- 4. THE STRATEGY TABLE (Relocated Below Graph) ---
            st.write("### 🛡️ Tailored Strategy by Seasonal Zone")
            z_score = norm.ppf(target_sl)
            
            # Aggregating Rolling Stats
            zone_stats = df.dropna(subset=['Rolling_Demand']).groupby('Zone').agg({
                'Rolling_Demand': ['mean', 'std', 'max']
            }).reset_index()
            zone_stats.columns = ['Zone', 'Avg Window Demand', 'Volatility (StdDev)', 'Absolute Max']
            
            # Calculate ROP and Risk Gap
            zone_stats['Rec. Safety Stock'] = (z_score * zone_stats['Volatility (StdDev)']).round(0)
            zone_stats['Reorder Point (ROP)'] = (zone_stats['Avg Window Demand'] + zone_stats['Rec. Safety Stock']).round(0)
            zone_stats['The Risk Gap'] = (zone_stats['Absolute Max'] - zone_stats['Reorder Point (ROP)']).round(0)
            
            # Formatting for display
            display_table = zone_stats[['Zone', 'Avg Window Demand', 'Volatility (StdDev)', 'Reorder Point (ROP)', 'Absolute Max', 'The Risk Gap']]
            st.table(display_table)
            
            st.info(f"💡 **Interpretation:** 'The Risk Gap' shows units unprotected at your {target_sl*100:.0f}% Service Level. During Peak Season, your ROP is {zone_stats.loc[zone_stats['Zone'] == 'Peak Season', 'Reorder Point (ROP)'].values[0]:.0f} units.")

            # --- 5. PROBABILITY HISTOGRAM ---
            st.divider()
            st.subheader("📊 Probability Distribution of Window Demand")
            fig_hist = px.histogram(df.dropna(subset=['Rolling_Demand']), x="Rolling_Demand", color="Zone", 
                                    barmode='overlay', opacity=0.6,
                                    color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"})
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander("📋 View Detailed Segmented Log"):
                st.dataframe(df[["Date", "Demand", "Rolling_Demand", "Zone"]], use_container_width=True)

else:
    st.info("Upload your Excel audit file to begin.")
