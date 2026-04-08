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
    """Vectorized historical performance audit."""
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    if "Order Placed" not in df.columns:
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    daily_h_rate = (h_pct / 100) / 365
    df['HoldingCost'] = df['Closing Balance'] * u_val * daily_h_rate
    df['OrderingCost'] = np.where(df['Order Placed'] > 0, o_cost, 0)
    
    # Strict Stockout Calculation
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    
    # Lead Time Window Mask
    df['InLT'] = df['Order Placed'].rolling(window=int(lt)+1, min_periods=1).sum() > 0
    
    return df

def run_prophet_dna(df):
    """Extracts seasonality and smooths zones."""
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
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "🕵️ Demand DNA", "🎮 Stress Test Simulator"])

    with tab1:
        total_d = df_audited['Demand'].sum()
        global_fr = (1 - (df_audited['Shortage'].sum() / total_d)) * 100 if total_d > 0 else 100
        lt_df = df_audited[df_audited['InLT'] == True]
        lt_fr = (1 - (lt_df['Shortage'].sum() / lt_df['Demand'].sum())) * 100 if not lt_df.empty else 100

        st.subheader("Inventory Operational & Service KPIs")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df_audited['IsStockout'].sum()))
        k2.metric("Global Fill Rate", f"{global_fr:.1f}%")
        k3.metric("LT Fill Rate", f"{lt_fr:.1f}%")
        k4.metric("Total Policy Cost", f"${(df_audited['HoldingCost'].sum() + df_audited['OrderingCost'].sum()):,.0f}")

        fig_audit = go.Figure()
        fig_audit.add_trace(go.Scatter(x=df_audited["Date"], y=df_audited["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2)))
        fig_audit.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_audit, use_container_width=True)

    with tab2:
        st.header("🕵️ Demand DNA & Stratification")
        if st.button("🚀 Run AI DNA Extraction"):
            st.session_state['dna_df'] = run_prophet_dna(df_audited)

        if 'dna_df' in st.session_state:
            df = st.session_state['dna_df']
            st.plotly_chart(px.scatter(df, x='Date', y='Demand', color='Zone', color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"}), use_container_width=True)

            c_s1, c_s2 = st.columns(2)
            with c_s1: risk_window = st.slider("Lead Time Window", 1, 30, int(lt_manual))
            with c_s2: target_sl = st.slider("Target Service Level (%)", 80, 99, 95) / 100
            
            df['Rolling_Demand'] = df['Demand'].rolling(window=risk_window).sum()
            st.subheader("📊 Probability Distribution (Lead-Time Demand)")
            st.plotly_chart(px.histogram(df.dropna(subset=['Rolling_Demand']), x="Rolling_Demand", color="Zone", barmode='overlay', opacity=0.6, color_discrete_map={"Peak Season": "#F56565", "Normal Season": "#63B3ED", "Low Season": "#4FD1C5"}), use_container_width=True)

            z_score = norm.ppf(target_sl)
            zone_stats = df.dropna(subset=['Rolling_Demand']).groupby('Zone')['Rolling_Demand'].agg(['mean', 'std', 'max']).reset_index()
            zone_stats.columns = ['Zone', 'Avg Window Demand', 'Volatility (StdDev)', 'Absolute Max']
            zone_stats['ROP'] = (zone_stats['Avg Window Demand'] + (z_score * zone_stats['Volatility (StdDev)'])).round(0)
            st.table(zone_stats)
            st.session_state['zone_stats'] = zone_stats

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
            with sc4: sim_days = st.slider("Simulation Duration", 30, 365, 90)

            if st.button("▶️ Run Stress Test"):
                avg_d = df_full[df_full['Zone'] == sel_zone]['Demand'].mean()
                std_d = df_full[df_full['Zone'] == sel_zone]['Demand'].std()
                sim_demands = np.random.normal(avg_d, std_d, sim_days).astype(int).clip(0)
                
                stocks, shortages, in_lt_sim, orders_placed = [], [], [], []
                stock = test_rop + (test_qty / 2)
                p_orders = {}
                daily_h_rate = (h_pct / 100) / 365

                for d in range(sim_days):
                    recv = p_orders.pop(d, 0)
                    open_s = stock + recv
                    sales = min(open_s, sim_demands[d])
                    shrt = max(0, sim_demands[d] - open_s) # Stockout Logic
                    close_s = open_s - sales
                    
                    placed = 0
                    if (close_s + sum(p_orders.values())) <= test_rop:
                        p_orders[d + int(lt_manual)] = test_qty
                        placed = test_qty
                    
                    stocks.append(close_s); shortages.append(shrt)
                    in_lt_sim.append(len(p_orders) > 0)
                    orders_placed.append(placed)
                    stock = close_s

                sdf = pd.DataFrame({"Day": range(sim_days), "Demand": sim_demands, "Stock": stocks, "Shortage": shortages, "InLT": in_lt_sim, "Placed": orders_placed})

                # --- SIMULATION KPIs ---
                st.subheader("Inventory Operational & Service KPIs")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Stockout Days", int((sdf['Shortage'] > 0).sum()))
                fill_rate = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100 if sdf['Demand'].sum() > 0 else 100
                k2.metric("Global Fill Rate", f"{fill_rate:.1f}%")
                lt_sim_df = sdf[sdf['InLT'] == True]
                lt_sim_fr = (1 - (lt_sim_df['Shortage'].sum() / lt_sim_df['Demand'].sum())) * 100 if not lt_sim_df.empty else 100
                k3.metric("LT Fill Rate", f"{lt_sim_fr:.1f}%")
                k4.metric("Lost Revenue", f"${(sdf['Shortage'].sum() * u_val):,.0f}", delta=f"{int(sdf['Shortage'].sum())} Units", delta_color="inverse")

                # --- INVENTORY MOVEMENT GRAPH ---
                st.subheader("📈 Inventory Movement (Stock Level vs. Demand)")
                fig_stk = go.Figure()
                fig_stk.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Stock"], name="Stock Level", line=dict(color="#00FFCC", width=3)))
                fig_stk.add_trace(go.Bar(x=sdf["Day"], y=sdf["Demand"], name="Demand", opacity=0.2, marker_color="white"))
                fig_stk.add_hline(y=test_rop, line_dash="dash", line_color="orange", annotation_text="ROP")
                
                # Marking Stockout Days
                stockouts = sdf[sdf['Shortage'] > 0]
                if not stockouts.empty:
                    fig_stk.add_trace(go.Scatter(x=stockouts["Day"], y=[0]*len(stockouts), mode='markers', name="Stockout Event", marker=dict(color='red', size=10, symbol='x')))
                
                fig_stk.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_stk, use_container_width=True)

                st.subheader("💰 Financial Impact & Cash Flow")
                f1, f2, f3 = st.columns(3)
                f1.metric("Avg WC Investment", f"${(sdf['Stock'].mean() * u_val):,.0f}")
                f2.metric("Total Holding Cost", f"${(sdf['Stock'].sum() * u_val * daily_h_rate):,.0f}")
                f3.metric("Total Ordering Cost", f"${((sdf['Placed'] > 0).sum() * o_cost):,.0f}")

                st.divider()
                st.subheader("📋 Detailed Simulation Logs")
                st.dataframe(sdf, use_container_width=True)
else:
    st.info("👋 Upload your Excel audit file to begin.")
