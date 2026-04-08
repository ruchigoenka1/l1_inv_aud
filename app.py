import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm
import io

# ------------------------------------------------
# 1. Page Config & Core Logic
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Inventory Auditor Pro")

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
    df['InLT'] = df['Order Placed'].rolling(window=int(lt)+1, min_periods=1).sum() > 0
    return df

def run_prophet_dna(df):
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(m_df)
    forecast = m.predict(m_df)
    df['Trend'] = forecast['trend'].values
    df['Raw_Seasonal'] = forecast['additive_terms'].values
    df['Smoothed_Seasonal'] = df['Raw_Seasonal'].rolling(window=14, center=True).mean().ffill().bfill()
    
    high_v = df['Smoothed_Seasonal'].quantile(0.85)
    low_v = df['Smoothed_Seasonal'].quantile(0.15)
    df['Zone'] = 'Normal Season'
    df.loc[df['Smoothed_Seasonal'] >= high_v, 'Zone'] = 'Peak Season'
    df.loc[df['Smoothed_Seasonal'] <= low_v, 'Zone'] = 'Low Season'
    return df

# ------------------------------------------------
# 2. Sidebar
# ------------------------------------------------
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Participant Excel", type=["xlsx"])
st.sidebar.divider()
st.sidebar.header("Policy Parameters")
u_val = st.sidebar.number_input("Value Per Unit ($)", value=100)
h_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
o_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lt_manual = st.sidebar.number_input("Lead Time (Days)", value=3, step=1)

# ------------------------------------------------
# 3. Main Dashboard
# ------------------------------------------------
if uploaded_file:
    raw_data = pd.read_excel(uploaded_file)
    df_audited = run_full_audit(raw_data, lt_manual, u_val, h_pct, o_cost)
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "🕵️ Demand DNA", "🎮 Stress Test Simulator"])

    # --- TAB 1 & 2 remain consistent with your current best version ---
    with tab1:
        st.subheader("Historical Inventory Flow")
        fig_audit = go.Figure()
        fig_audit.add_trace(go.Scatter(x=df_audited["Date"], y=np.where(df_audited["InLT"], df_audited["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Lead Time Window"))
        fig_audit.add_trace(go.Scatter(x=df_audited["Date"], y=df_audited["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2)))
        st.plotly_chart(fig_audit, use_container_width=True)

    with tab2:
        if st.button("🚀 Run AI DNA Extraction"):
            st.session_state['dna_df'] = run_prophet_dna(df_audited)
        if 'dna_df' in st.session_state:
            df = st.session_state['dna_df']
            st.plotly_chart(px.scatter(df, x='Date', y='Demand', color='Zone', color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"}), use_container_width=True)
            
            risk_window = st.slider("Analysis Window", 1, 30, int(lt_manual))
            target_sl = st.slider("Target Service Level (%)", 80, 99, 95) / 100
            df['Rolling_Demand'] = df['Demand'].rolling(window=risk_window).sum()
            
            z_score = norm.ppf(target_sl)
            zone_stats = df.dropna(subset=['Rolling_Demand']).groupby('Zone')['Rolling_Demand'].agg(['mean', 'std', 'max']).reset_index()
            zone_stats.columns = ['Zone', 'Avg Window Demand', 'Volatility (StdDev)', 'Absolute Max']
            zone_stats['ROP'] = (zone_stats['Avg Window Demand'] + (z_score * zone_stats['Volatility (StdDev)'])).round(0)
            st.table(zone_stats)
            st.session_state['zone_stats'] = zone_stats

    # --- TAB 3: UPDATED STRESS TEST WITH KPIs ---
    with tab3:
        if 'zone_stats' not in st.session_state:
            st.warning("Please run Tab 2 first.")
        else:
            z_stats = st.session_state['zone_stats']
            df_full = st.session_state['dna_df']
            
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: sel_zone = st.selectbox("Season to Simulate", z_stats['Zone'].unique())
            with sc2: test_rop = st.number_input("Test ROP", value=int(z_stats.loc[z_stats['Zone']==sel_zone, 'ROP'].values[0]))
            with sc3: test_qty = st.number_input("Test Order Qty", value=int(test_rop * 1.2))
            with sc4: sim_days = st.slider("Simulation Days", 30, 365, 90)

            if st.button("▶️ Run Stress Test"):
                avg_d = df_full[df_full['Zone'] == sel_zone]['Demand'].mean()
                std_d = df_full[df_full['Zone'] == sel_zone]['Demand'].std()
                sim_demands = np.random.normal(avg_d, std_d, sim_days).astype(int).clip(0)
                
                stocks, shortages, orders_placed = [], [], []
                stock = test_rop + (test_qty / 2)
                p_orders = {}
                daily_h_rate = (h_pct / 100) / 365

                for d in range(sim_days):
                    recv = p_orders.pop(d, 0)
                    open_s = stock + recv
                    sales = min(open_s, sim_demands[d])
                    shrt = max(0, sim_demands[d] - open_s)
                    close_s = open_s - sales
                    
                    placed = 0
                    if (close_s + sum(p_orders.values())) <= test_rop:
                        p_orders[d + int(lt_manual)] = test_qty
                        placed = test_qty
                    
                    stocks.append(close_s)
                    shortages.append(shrt)
                    orders_placed.append(placed)
                    stock = close_s

                sdf = pd.DataFrame({"Day": range(sim_days), "Demand": sim_demands, "Stock": stocks, "Shortage": shortages, "Order_Placed": orders_placed})
                
                # --- KPI DASHBOARD (Match Screenshot) ---
                st.subheader("Inventory Operational & Service KPIs")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Stockout Days", int((sdf['Shortage'] > 0).sum()))
                fill_rate = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100
                m2.metric("Global Fill Rate", f"{fill_rate:.1f}%")
                m3.metric("Avg Inv (Units)", f"{sdf['Stock'].mean():.1f}")
                m4.metric("Max Inv (Units)", f"{sdf['Stock'].max():.1f}")

                st.subheader("Range & Working Capital Analysis")
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Min Inventory", int(sdf['Stock'].min()))
                w2.metric("Avg WC Investment", f"${(sdf['Stock'].mean() * u_val):,.0f}")
                w3.metric("Max WC Investment", f"${(sdf['Stock'].max() * u_val):,.0f}")
                w4.metric("Total Order Count", int((sdf['Order_Placed'] > 0).sum()))

                st.subheader("Financial Stress Test")
                f1, f2, f3 = st.columns(3)
                total_h = sdf['Stock'].sum() * u_val * daily_h_rate
                total_o = (sdf['Order_Placed'] > 0).sum() * o_cost
                f1.metric("Total Holding Cost", f"${total_h:,.0f}")
                f2.metric("Total Ordering Cost", f"${total_o:,.0f}")
                f3.metric("Total Policy Cost", f"${(total_h + total_o):,.0f}", delta_color="inverse")

                # Graph
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Stock"], name="Stock Level", line=dict(color="#00FFCC", width=3)))
                fig_sim.add_trace(go.Bar(x=sdf["Day"], y=sdf["Demand"], name="Demand", opacity=0.3))
                st.plotly_chart(fig_sim, use_container_width=True)

                # --- DATA TABLE AT THE END ---
                st.divider()
                st.subheader("📋 Detailed Simulation Data Logs")
                st.dataframe(sdf, use_container_width=True)
                
                # Excel Download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    sdf.to_excel(writer, index=False, sheet_name='Stress_Test_Results')
                st.download_button(label="📥 Download Simulation Data", data=buffer, file_name="stress_test_results.xlsx")
