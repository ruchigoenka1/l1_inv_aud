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
    """Performs the historical audit with strict stockout logic."""
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    if "Order Placed" not in df.columns:
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    # Prorated Daily Rates
    daily_h_rate = (h_pct / 100) / 365
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    df['InLT'] = df['Order Placed'].rolling(window=int(lt)+1, min_periods=1).sum() > 0
    return df

def run_prophet_dna(df):
    """AI Seasonal Extraction & Smoothing."""
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(m_df)
    forecast = m.predict(m_df)
    
    df['Trend'] = forecast['trend'].values
    df['Raw_Seasonal'] = forecast['additive_terms'].values
    # Smoothed for operational consistency
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

st.sidebar.header("Financial Parameters")
u_val = st.sidebar.number_input("Value Per Unit ($)", value=100)
h_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
o_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lt_manual = st.sidebar.number_input("Actual Lead Time (Days)", value=3, step=1)

st.sidebar.divider()
st.sidebar.header("Warehouse Overheads")
fixed_wh_cost = st.sidebar.number_input("Annual Fixed Storage Cost per Unit ($)", value=10.0, 
                                       help="Total annual Rent + Labor + Equip divided by total capacity")

# ------------------------------------------------
# 3. Main Dashboard
# ------------------------------------------------
if uploaded_file:
    raw_data = pd.read_excel(uploaded_file)
    df_audited = run_full_audit(raw_data, lt_manual, u_val, h_pct, o_cost)
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "🕵️ Demand DNA", "🎮 Stress Test Simulator"])

    with tab1:
        # Audit Tab Metrics
        total_d = df_audited['Demand'].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df_audited['IsStockout'].sum()))
        k2.metric("Global Fill Rate", f"{(1 - (df_audited['Shortage'].sum() / total_d)) * 100:.1f}%")
        lt_df = df_audited[df_audited['InLT'] == True]
        lt_fr = (1 - (lt_df['Shortage'].sum() / lt_df['Demand'].sum())) * 100 if not lt_df.empty else 100
        k3.metric("LT Fill Rate", f"{lt_fr:.1f}%")
        k4.metric("Avg Inventory", f"{df_audited['Closing Balance'].mean():.1f}")
        st.plotly_chart(px.line(df_audited, x="Date", y="Closing Balance", title="Historical Stock Flow", template="plotly_dark"), use_container_width=True)

    with tab2:
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
            st.plotly_chart(px.histogram(df.dropna(subset=['Rolling_Demand']), x="Rolling_Demand", color="Zone", barmode='overlay', opacity=0.6), use_container_width=True)

            z_score = norm.ppf(target_sl)
            zone_stats = df.dropna(subset=['Rolling_Demand']).groupby('Zone')['Rolling_Demand'].agg(['mean', 'std', 'max']).reset_index()
            zone_stats.columns = ['Zone', 'Avg Window Demand', 'Volatility (StdDev)', 'Absolute Max']
            zone_stats['ROP'] = (zone_stats['Avg Window Demand'] + (z_score * zone_stats['Volatility (StdDev)'])).round(0)
            st.table(zone_stats)
            st.session_state['zone_stats'] = zone_stats

    with tab3:
        if 'zone_stats' not in st.session_state:
            st.warning("Please run Tab 2 (Demand DNA) first.")
        else:
            z_stats = st.session_state['zone_stats']
            df_full = st.session_state['dna_df']
            
            # --- 1. Simulation Inputs & EOQ Logic ---
            with st.expander("🛠️ Simulation & Strategy Controls", expanded=True):
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: 
                    options = ["All Data"] + list(z_stats['Zone'].unique())
                    sel_zone = st.selectbox("Distribution Source", options)
                
                # Determine stats
                if sel_zone == "All Data":
                    avg_d, std_d = df_full['Demand'].mean(), df_full['Demand'].std()
                    default_rop = int(avg_d * lt_manual + (1.65 * std_d)) 
                else:
                    avg_d = df_full[df_full['Zone'] == sel_zone]['Demand'].mean()
                    std_d = df_full[df_full['Zone'] == sel_zone]['Demand'].std()
                    default_rop = int(z_stats.loc[z_stats['Zone']==sel_zone, 'ROP'].values[0])
    
                # EOQ Calculation
                annual_d_proj = avg_d * 365
                var_h_annual = u_val * (h_pct / 100)
                eoq_val = int(np.sqrt((2 * annual_d_proj * o_cost) / var_h_annual)) if var_h_annual > 0 else 0
    
                with sc2: test_rop = st.number_input("Test ROP", value=default_rop)
                with sc3: 
                    use_eoq = st.checkbox("Use EOQ Quantity", value=False)
                    test_qty = st.number_input("Test Order Qty (Q)", value=eoq_val if use_eoq else int(test_rop * 1.5))
                with sc4: sim_days = st.slider("Simulation Horizon (Days)", 30, 365, 365)
    
            if st.button("▶️ Run Comparative Stress Test"):
                # --- 2. Simulation Logic ---
                sim_ratio = sim_days / 365
                stocks, shortages, sim_demands, orders_placed, pipeline_history = [], [], [], [], []
                stock = test_rop + (test_qty / 2) 
                p_orders = {}
    
                for d in range(sim_days):
                    daily_d = int(max(0, np.random.normal(avg_d, std_d)))
                    recv = p_orders.pop(d, 0)
                    open_s = stock + recv
                    sales = min(open_s, daily_d)
                    shrt = int(daily_d - sales)
                    close_s = open_s - sales
                    
                    pipeline_val = sum(p_orders.values())
                    inv_pos = close_s + pipeline_val
                    
                    order_triggered = 0
                    if inv_pos <= test_rop and pipeline_val == 0:
                        p_orders[d + int(lt_manual)] = test_qty
                        order_triggered = 1
                    
                    stocks.append(close_s); shortages.append(shrt); sim_demands.append(daily_d)
                    orders_placed.append(order_triggered); pipeline_history.append(pipeline_val)
                    stock = close_s
    
                sdf = pd.DataFrame({
                    "Day": range(sim_days), "Demand": sim_demands, "Stock": stocks, 
                    "Shortage": shortages, "OrderEvent": orders_placed, "Pipeline": pipeline_history
                })
    
                # --- 3. COMPARATIVE STUDY CALCULATIONS ---
                # Case A: Original (Historical Audit Data)
                orig_days = len(df_audited)
                orig_ratio = orig_days / 365
                orig_avg_stock = df_audited['Closing Balance'].mean()
                orig_h = (orig_avg_stock * u_val * (h_pct/100) * orig_ratio) + (fixed_wh_cost * orig_ratio)
                orig_o = df_audited['Order Received'].astype(bool).sum() * o_cost # Approximation
                orig_l = df_audited['Shortage'].sum() * u_val
                orig_fr = (1 - (df_audited['Shortage'].sum() / df_audited['Demand'].sum())) * 100
    
                # Case B: Simulated (Test Policy)
                sim_avg_stock = sdf['Stock'].mean()
                sim_h = (sim_avg_stock * u_val * (h_pct/100) * sim_ratio) + (fixed_wh_cost * sim_ratio)
                sim_o = sdf['OrderEvent'].sum() * o_cost
                sim_l = sdf['Shortage'].sum() * u_val
                sim_fr = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100
    
                # --- 4. DISPLAY DASHBOARD ---
                st.subheader("💰 Comparative Financial Study")
                comp_data = {
                    "Metric": ["Avg Inventory (Units)", "Global Fill Rate (%)", "Holding Cost ($)", "Ordering Cost ($)", "Lost Sales Revenue ($)", "Total Inventory Cost ($)"],
                    "Original (Audit)": [f"{orig_avg_stock:.1f}", f"{orig_fr:.1f}%", f"{orig_h:,.0f}", f"{orig_o:,.0f}", f"{orig_l:,.0f}", f"{(orig_h + orig_o + orig_l):,.0f}"],
                    "Simulated (Test)": [f"{sim_avg_stock:.1f}", f"{sim_fr:.1f}%", f"{sim_h:,.0f}", f"{sim_o:,.0f}", f"{sim_l:,.0f}", f"{(sim_h + sim_o + sim_l):,.0f}"]
                }
                st.table(pd.DataFrame(comp_data))
    
                # --- 5. WORKING CAPITAL MAPS (Comparative) ---
                st.subheader("🏦 Working Capital Map ($ Value of On-Hand Stock)")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Original Historical Case**")
                    st.plotly_chart(px.area(df_audited, x="Date", y=df_audited["Closing Balance"]*u_val, color_discrete_sequence=['#63B3ED']), use_container_width=True)
                with c2:
                    st.write("**Simulated Strategy Case**")
                    st.plotly_chart(px.area(sdf, x="Day", y=sdf["Stock"]*u_val, color_discrete_sequence=['#FFA500']), use_container_width=True)
    
                # --- 6. PIPELINE & ROP GRAPH ---
                st.subheader("🚠 Pipeline & Stock Position Analysis")
                fig_pipe = go.Figure()
                fig_pipe.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Pipeline"], name="Pipeline Inventory (In-Transit)", line=dict(color="#FF00FF", width=2, dash='dot')))
                fig_pipe.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Stock"], name="Physical Stock (On-Hand)", fill='tozeroy', line=dict(color="#00FFCC")))
                fig_pipe.add_hline(y=test_rop, line_dash="dash", line_color="orange", annotation_text="ROP")
                fig_pipe.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_pipe, use_container_width=True)
    
                # --- 7. SIMULATED LOG DATA ---
                with st.expander("📋 View Raw Simulated Log Data"):
                    st.dataframe(sdf, use_container_width=True)
else:
    st.info("👋 Upload historical Excel file to begin.")
