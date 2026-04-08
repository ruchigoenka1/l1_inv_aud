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
            st.warning("Please run 'AI DNA Extraction' in Tab 2 first.")
        else:
            z_stats = st.session_state['zone_stats']
            df_full = st.session_state['dna_df']
            
            # --- 1. Simulation & Strategy Controls ---
            with st.expander("🛠️ Strategy & Stress Test Controls", expanded=True):
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: 
                    options = ["All Data"] + list(z_stats['Zone'].unique())
                    sel_zone = st.selectbox("Distribution Source", options)
                
                # Extract Demand Profile Stats
                if sel_zone == "All Data":
                    avg_d, std_d = df_full['Demand'].mean(), df_full['Demand'].std()
                else:
                    avg_d = df_full[df_full['Zone'] == sel_zone]['Demand'].mean()
                    std_d = df_full[df_full['Zone'] == sel_zone]['Demand'].std()
    
                # EOQ Calculation (Variable Logic:sqrt(2DS/H))
                annual_d_proj = avg_d * 365
                var_h_annual_unit = u_val * (h_pct / 100)
                eoq_val = int(np.sqrt((2 * annual_d_proj * o_cost) / var_h_annual_unit)) if var_h_annual_unit > 0 else 0
    
                with sc2: 
                    test_rop = st.number_input("Test Reorder Point (ROP)", value=int(avg_d * lt_manual * 1.2))
                with sc3: 
                    use_eoq = st.checkbox("Use EOQ Quantity", value=False)
                    test_qty = st.number_input("Test Order Qty (Q)", value=eoq_val if use_eoq else int(test_rop * 1.5))
                with sc4: 
                    sim_days = st.slider("Simulation Horizon (Days)", 30, 365, 365)
    
            if st.button("▶️ Run Comparative Study"):
                # --- 2. Simulation Logic ---
                stocks, shortages, sim_demands, orders_placed, pipeline_history = [], [], [], [], []
                stock = test_rop + (test_qty / 2) # Starting assumption
                p_orders = {}
    
                for d in range(sim_days):
                    # Demand Sync
                    daily_d = int(max(0, np.random.normal(avg_d, std_d)))
                    
                    # Arrivals
                    recv = p_orders.pop(d, 0)
                    open_s = stock + recv
                    
                    # Fulfillment
                    sales = min(open_s, daily_d)
                    shrt = int(daily_d - sales)
                    close_s = open_s - sales
                    
                    # Correct Pipeline Tracking (Sum of orders in the air)
                    pipeline_val = sum(p_orders.values())
                    inv_pos = close_s + pipeline_val
                    
                    # Reorder Logic (Only 1 order in pipeline allowed)
                    order_triggered = 0
                    if inv_pos <= test_rop and pipeline_val == 0:
                        p_orders[d + int(lt_manual)] = test_qty
                        order_triggered = 1
                        pipeline_val = test_qty # Update for current day visual
                    
                    stocks.append(close_s)
                    shortages.append(shrt)
                    sim_demands.append(daily_d)
                    orders_placed.append(order_triggered)
                    pipeline_history.append(pipeline_val)
                    stock = close_s
    
                sdf = pd.DataFrame({
                    "Day": range(sim_days), 
                    "Demand": sim_demands, 
                    "Stock": stocks, 
                    "Shortage": shortages, 
                    "OrderEvent": orders_placed, 
                    "Pipeline": pipeline_history
                })
    
                # --- 3. ANNUALIZED COMPARATIVE MATH (Apples-to-Apples) ---
                
                # Case A: Original (Audit)
                orig_n_days = len(df_audited)
                orig_annual_factor = 365 / orig_n_days
                orig_avg_stock = df_audited['Closing Balance'].mean()
                orig_avg_wc = orig_avg_stock * u_val
                
                orig_h_annual = (orig_avg_wc * (h_pct/100)) + fixed_wh_cost
                orig_o_annual = (df_audited['Order Received'].astype(bool).sum() * orig_annual_factor) * o_cost
                orig_l_annual = (df_audited['Shortage'].sum() * orig_annual_factor) * u_val
                orig_fr = (1 - (df_audited['Shortage'].sum() / df_audited['Demand'].sum())) * 100
    
                # Case B: Simulated (Test)
                sim_annual_factor = 365 / sim_days
                sim_avg_stock = sdf['Stock'].mean()
                sim_avg_wc = sim_avg_stock * u_val
                
                sim_h_annual = (sim_avg_stock * u_val * (h_pct/100)) + fixed_wh_cost
                sim_o_annual = (sdf['OrderEvent'].sum() * sim_annual_factor) * o_cost
                sim_l_annual = (sdf['Shortage'].sum() * sim_annual_factor) * u_val
                sim_fr = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100
    
                # --- 4. COMPARATIVE TABLE ---
                st.subheader("💰 Annualized Comparative Financial Study")
                
                metrics = [
                    ("Avg Inventory (Units)", orig_avg_stock, sim_avg_stock, "lower"),
                    ("Avg Working Capital ($)", orig_avg_wc, sim_avg_wc, "lower"),
                    ("Global Fill Rate (%)", orig_fr, sim_fr, "higher"),
                    ("Annual Holding Cost ($)", orig_h_annual, sim_h_annual, "lower"),
                    ("Annual Ordering Cost ($)", orig_o_annual, sim_o_annual, "lower"),
                    ("Annual Lost Sales Revenue ($)", orig_l_annual, sim_l_annual, "lower"),
                    ("Total Annual Policy Cost ($)", (orig_h_annual + orig_o_annual + orig_l_annual), (sim_h_annual + sim_o_annual + sim_l_annual), "lower")
                ]
    
                rows = []
                for label, orig, sim, direction in metrics:
                    diff = ((sim - orig) / orig * 100) if orig != 0 else 0
                    if direction == "lower":
                        color = "green" if sim <= orig else "red"
                    else:
                        color = "green" if sim >= orig else "red"
                    
                    rows.append({
                        "Metric": label,
                        "Original (Audit)": f"{orig:,.1f}" if "%" in label else f"${orig:,.0f}",
                        "Simulated (Test)": f"{sim:,.1f}" if "%" in label else f"${sim:,.0f}",
                        "% Difference": f":{color}[{diff:+.1f}%]"
                    })
    
                st.table(pd.DataFrame(rows))
    
                # --- 5. VISUALS ---
                st.subheader("🚠 Stock & Pipeline Analysis (Lead-Time Sync)")
                fig_p = go.Figure()
                # Pipeline Position (Step Line)
                fig_p.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Pipeline"], name="Pipeline (On-Order)", 
                                         line=dict(color="#FF00FF", width=2, dash='dot', shape='hv')))
                # Physical Stock
                fig_p.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Stock"], name="Physical Stock (On-Hand)", 
                                         fill='tozeroy', line=dict(color="#00FFCC", width=3), fillcolor='rgba(0, 255, 204, 0.1)'))
                # ROP Line
                fig_p.add_hline(y=test_rop, line_dash="dash", line_color="orange", annotation_text=f"ROP: {test_rop}")
                
                fig_p.update_layout(template="plotly_dark", height=450, hovermode="x unified")
                st.plotly_chart(fig_p, use_container_width=True)
    
                st.subheader("🏦 Working Capital Map ($ Value)")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Historical Audit Case (Blue)**")
                    st.plotly_chart(px.area(df_audited, x="Date", y=df_audited["Closing Balance"]*u_val, color_discrete_sequence=['#63B3ED']), use_container_width=True)
                with c2:
                    st.write("**Simulated Strategy Case (Orange)**")
                    st.plotly_chart(px.area(sdf, x="Day", y=sdf["Stock"]*u_val, color_discrete_sequence=['#FFA500']), use_container_width=True)
    
                with st.expander("📋 View Simulation Log Data"):
                    st.dataframe(sdf, use_container_width=True)
else:
    st.info("👋 Upload historical Excel file to begin.")
