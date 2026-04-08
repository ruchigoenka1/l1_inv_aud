import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm

# ------------------------------------------------
# 1. Page Config & Core Logic
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Inventory Auditor Pro")

def run_full_audit(df, lt, u_val, h_pct, o_cost):
    """Vectorized historical performance audit."""
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Vectorized order placement inference if not present
    if "Order Placed" not in df.columns:
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    # Financials (Vectorized)
    daily_h_rate = (h_pct / 100) / 365
    df['HoldingCost'] = df['Closing Balance'] * u_val * daily_h_rate
    df['OrderingCost'] = np.where(df['Order Placed'] > 0, o_cost, 0)
    
    # Audit Shortages (Vectorized)
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    
    # Vectorized Lead Time Window Shading
    # We create a boolean mask: True if any order was placed in the last 'lt' days
    df['InLT'] = df['Order Placed'].rolling(window=lt+1, min_periods=1).sum().shift(0) > 0
    
    return df

def run_prophet_dna(df):
    """Vectorized DNA Extraction & Smoothing."""
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(m_df)
    forecast = m.predict(m_df)
    
    df['Trend'] = forecast['trend'].values
    df['Raw_Seasonal'] = forecast['additive_terms'].values
    
    # Vectorized Smoothing
    df['Smoothed_Seasonal'] = df['Raw_Seasonal'].rolling(window=14, center=True).mean().ffill().bfill()
    
    # Vectorized Quantile Stratification
    high_v = df['Smoothed_Seasonal'].quantile(0.85)
    low_v = df['Smoothed_Seasonal'].quantile(0.15)
    
    df['Zone'] = 'Normal Season'
    df.loc[df['Smoothed_Seasonal'] >= high_v, 'Zone'] = 'Peak Season'
    df.loc[df['Smoothed_Seasonal'] <= low_v, 'Zone'] = 'Low Season'
    
    return df

# ------------------------------------------------
# 2. Sidebar UI
# ------------------------------------------------
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Participant Excel", type=["xlsx"])
st.sidebar.divider()
st.sidebar.header("Policy Parameters")
u_val = st.sidebar.number_input("Value Per Unit ($)", value=100)
h_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
o_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lt_manual = st.sidebar.number_input("Actual Lead Time (Days)", value=3, step=1)

# ------------------------------------------------
# 3. Main Dashboard
# ------------------------------------------------
if uploaded_file:
    raw_data = pd.read_excel(uploaded_file)
    df_audited = run_full_audit(raw_data, lt_manual, u_val, h_pct, o_cost)
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "🕵️ Demand DNA & Zones", "🎮 Strategy Simulator"])

    with tab1:
        # Vectorized Metrics
        total_d = df_audited['Demand'].sum()
        fill_rate = (1 - (df_audited['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df_audited['IsStockout'].sum()))
        k2.metric("Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inventory", f"{df_audited['Closing Balance'].mean():.1f}")
        k4.metric("Policy Cost", f"${(df_audited['HoldingCost'].sum() + df_audited['OrderingCost'].sum()):,.0f}")

        fig_audit = go.Figure()
        fig_audit.add_trace(go.Scatter(x=df_audited["Date"], y=np.where(df_audited["InLT"], df_audited["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Lead Time Window"))
        fig_audit.add_trace(go.Scatter(x=df_audited["Date"], y=df_audited["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2)))
        fig_audit.update_layout(template="plotly_dark", height=450, title="Historical Inventory Flow")
        st.plotly_chart(fig_audit, use_container_width=True)

    with tab2:
        if st.button("🚀 Run AI DNA Extraction"):
            st.session_state['dna_df'] = run_prophet_dna(df_audited)

        if 'dna_df' in st.session_state:
            df = st.session_state['dna_df']
            fig_dna = px.scatter(df, x='Date', y='Demand', color='Zone',
                                    color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"})
            fig_dna.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], mode='lines', name='Growth Trend', line=dict(color='#FFA500', width=2, dash='dot')))
            fig_dna.update_layout(template="plotly_dark")
            st.plotly_chart(fig_dna, use_container_width=True)

            st.divider()
            c_s1, c_s2 = st.columns(2)
            with c_s1: risk_window = st.slider("Lead Time Window (Days)", 1, 30, lt_manual)
            with c_s2: target_sl = st.slider("Target Service Level (%)", 80, 99, 95) / 100

            # Vectorized Rolling Window & Stats
            df['Rolling_Demand'] = df['Demand'].rolling(window=risk_window).sum()
            fig_roll = px.line(df, x='Date', y='Rolling_Demand', title=f"Rolling Sum of Demand ({risk_window} Days)", color_discrete_sequence=['#00FFCC'])
            st.plotly_chart(fig_roll, use_container_width=True)

            z_score = norm.ppf(target_sl)
            zone_stats = df.dropna(subset=['Rolling_Demand']).groupby('Zone')['Rolling_Demand'].agg(['mean', 'std', 'max']).reset_index()
            zone_stats.columns = ['Zone', 'Avg Window Demand', 'Volatility (StdDev)', 'Absolute Max']
            zone_stats['Rec. Safety Stock'] = (z_score * zone_stats['Volatility (StdDev)']).round(0)
            zone_stats['Reorder Point (ROP)'] = (zone_stats['Avg Window Demand'] + zone_stats['Rec. Safety Stock']).round(0)
            zone_stats['The Risk Gap'] = (zone_stats['Absolute Max'] - zone_stats['Reorder Point (ROP)']).round(0)
            
            st.table(zone_stats)
            st.session_state['zone_stats'] = zone_stats

    with tab3:
        st.header("🎮 Strategic Stress Test")
        if 'zone_stats' not in st.session_state:
            st.warning("Please run 'AI DNA Extraction' in Tab 2 first.")
        else:
            z_stats = st.session_state['zone_stats']
            df_full = st.session_state['dna_df']
            
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: sel_zone = st.selectbox("Season to Simulate", z_stats['Zone'].unique())
            with sc2: test_rop = st.number_input("Test ROP", value=int(z_stats.loc[z_stats['Zone']==sel_zone, 'Reorder Point (ROP)'].values[0]))
            with sc3: test_qty = st.number_input("Test Order Qty", value=int(test_rop * 1.2))
            with sc4: sim_days = st.slider("Simulation Days", 30, 180, 90)

            if st.button("▶️ Run Stress Test"):
                avg_d = df_full[df_full['Zone'] == sel_zone]['Demand'].mean()
                std_d = df_full[df_full['Zone'] == sel_zone]['Demand'].std()

                # Vectorized Demand Generation
                sim_demands = np.random.normal(avg_d, std_d, sim_days).astype(int).clip(0)
                
                # Sequential Stock Simulation (Inherently non-vectorizable for accuracy)
                stocks, shortages = [], []
                stock = test_rop + (test_qty / 2)
                p_orders = {}

                for d in range(sim_days):
                    recv = p_orders.pop(d, 0)
                    open_s = stock + recv
                    sales = min(open_s, sim_demands[d])
                    shortages.append(max(0, sim_demands[d] - open_s))
                    stock = open_s - sales
                    stocks.append(stock)
                    
                    if (stock + sum(p_orders.values())) <= test_rop:
                        p_orders[d + lt_manual] = test_qty

                sdf = pd.DataFrame({"Day": range(sim_days), "Demand": sim_demands, "Stock": stocks, "Shortage": shortages})
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Stockout Events", int((sdf['Shortage'] > 0).sum()))
                m2.metric("Total Lost Sales", int(sdf['Shortage'].sum()))
                m3.metric("Avg Sim Stock", f"{sdf['Stock'].mean():.1f}")

                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Stock"], name="Stock Level", line=dict(color="#00FFCC")))
                fig_sim.add_trace(go.Bar(x=sdf["Day"], y=sdf["Demand"], name="Demand", opacity=0.3))
                fig_sim.update_layout(template="plotly_dark", title=f"Simulation Result: {sel_zone}")
                st.plotly_chart(fig_sim, use_container_width=True)
else:
    st.info("👋 Welcome. Please upload your Excel file to begin.")
