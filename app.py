import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm

st.markdown(
    """
    <style>
        /* Push sidebar content down and right */
        [data-testid="stSidebarUserContent"] {
            # padding-top: 100px;
            padding-left: 50px;
            # padding-right: 20px;
        }
        
        /* Ensure the sidebar stays wide enough when zoomed */
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 450px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 1. Page Config & Core Logic
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Inventory Auditor Pro")

def run_full_audit(df, lt, u_val, h_pct, o_cost):
    """Historical Audit with Automatic Date Healing."""
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Date Healing Logic (Fills gaps in the calendar)
    full_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df = df.set_index('Date').reindex(full_range).reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    df['Demand'] = df['Demand'].fillna(0)
    df['Order Received'] = df['Order Received'].fillna(0)
    df[['Opening Balance', 'Closing Balance']] = df[['Opening Balance', 'Closing Balance']].ffill()

    if "Order Placed" not in df.columns:
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    df['InLT'] = df['Order Placed'].rolling(window=int(lt)+1, min_periods=1).sum() > 0
    return df

def run_prophet_dna(df):
    """AI Demand DNA Extraction."""
    m_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(m_df)
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

st.sidebar.header("Financial Parameters")
u_val = st.sidebar.number_input("Value Per Unit ($)", value=100)
h_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
o_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lt_manual = st.sidebar.number_input("Actual Lead Time (Days)", value=3, step=1)

st.sidebar.divider()
st.sidebar.header("Warehouse Overheads")
fixed_wh_cost = st.sidebar.number_input("Total Annual Fixed Warehouse Cost ($)", value=1000.0)

# ------------------------------------------------
# 3. Main Dashboard
# ------------------------------------------------
if uploaded_file:
    raw_data = pd.read_excel(uploaded_file)
    df_audited = run_full_audit(raw_data, lt_manual, u_val, h_pct, o_cost)
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "🕵️ Demand DNA", "🎮 Stress Test Simulator"])

    # --- TAB 1: AUDIT ---
    with tab1:
        total_d = df_audited['Demand'].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df_audited['IsStockout'].sum()))
        k2.metric("Global Fill Rate", f"{(1 - (df_audited['Shortage'].sum() / total_d)) * 100:.1f}%")
        lt_df = df_audited[df_audited['InLT'] == True]
        lt_fr = (1 - (lt_df['Shortage'].sum() / lt_df['Demand'].sum())) * 100 if not lt_df.empty else 100
        k3.metric("LT Fill Rate", f"{lt_fr:.1f}%")
        k4.metric("Avg Inventory", f"{df_audited['Closing Balance'].mean():.1f}")
        st.plotly_chart(px.line(df_audited, x="Date", y="Closing Balance", title="Historical Stock Flow", template="plotly_dark"), use_container_width=True)

    # --- TAB 2: DNA ---
    with tab2:
        if st.button("🚀 Run AI DNA Extraction"):
            st.session_state['dna_df'] = run_prophet_dna(df_audited)
        if 'dna_df' in st.session_state:
            df = st.session_state['dna_df']
            st.plotly_chart(px.scatter(df, x='Date', y='Demand', color='Zone', title="Seasonal Demand Zones"), use_container_width=True)
            
            c_s1, c_s2 = st.columns(2)
            with c_s1: risk_window = st.slider("Lead Time Window (Days)", 1, 30, int(lt_manual))
            with c_s2: target_sl = st.slider("Target Service Level (%)", 80, 99, 95) / 100
            
            df['Rolling_Demand'] = df['Demand'].rolling(window=risk_window).sum()
            st.subheader("📊 Probability Distribution (Lead-Time Demand)")
            st.plotly_chart(px.histogram(df.dropna(subset=['Rolling_Demand']), x="Rolling_Demand", color="Zone", barmode='overlay', title="Demand Distribution by DNA Zone"), use_container_width=True)

            z_score = norm.ppf(target_sl)
            zone_stats = df.dropna(subset=['Rolling_Demand']).groupby('Zone')['Rolling_Demand'].agg(['mean', 'std']).reset_index()
            zone_stats['ROP'] = (zone_stats['mean'] + (z_score * zone_stats['std'])).round(0)
            st.table(zone_stats)
            st.session_state['zone_stats'] = zone_stats

    # --- TAB 3: SIMULATOR ---
    with tab3:
        if 'zone_stats' not in st.session_state:
            st.warning("Please run Tab 2 (Demand DNA) first.")
        else:
            z_stats = st.session_state['zone_stats']
            df_full = st.session_state['dna_df']
            
            # --- 1. Simulation Controls ---
            with st.expander("🛠️ Strategy & Stress Test Controls", expanded=True):
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: 
                    sel_zone = st.selectbox("Distribution Source", ["All Data"] + list(z_stats['Zone'].unique()))
                
                avg_d, std_d = (df_full['Demand'].mean(), df_full['Demand'].std()) if sel_zone == "All Data" else (df_full[df_full['Zone'] == sel_zone]['Demand'].mean(), df_full[df_full['Zone'] == sel_zone]['Demand'].std())
    
                annual_d_proj = avg_d * 365
                var_h_unit = u_val * (h_pct / 100)
                eoq_val = int(np.sqrt((2 * annual_d_proj * o_cost) / var_h_unit)) if var_h_unit > 0 else 0
    
                with sc2: test_rop = st.number_input("Test ROP", value=int(avg_d * lt_manual * 1.2))
                with sc3: 
                    use_eoq = st.checkbox("Use EOQ Quantity")
                    test_qty = st.number_input("Test Order Quantity", value=eoq_val if use_eoq else int(test_rop * 1.5))
                with sc4: sim_days = st.slider("Simulation Horizon (Days)", 30, 365, 365)
    
            if st.button("▶️ Run Comparative Study"):
                # --- 2. Simulation Logic ---
                stocks, shortages, sim_demands, orders_placed, pipeline_history, total_inv_pos = [], [], [], [], [], []
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
                        pipeline_val = test_qty 
                    
                    stocks.append(close_s); shortages.append(shrt); sim_demands.append(daily_d)
                    orders_placed.append(order_triggered); pipeline_history.append(pipeline_val)
                    total_inv_pos.append(inv_pos); stock = close_s
    
                sdf = pd.DataFrame({
                    "Day": range(sim_days), "Demand": sim_demands, "Physical_Stock": stocks, 
                    "Shortage": shortages, "OrderEvent": orders_placed, 
                    "Pipeline": pipeline_history, "Total_Inventory": total_inv_pos
                })
    
                # --- 3. EXPANDED SIMULATION KPIs ---
                st.subheader(f"🚀 Simulation Results ({sim_days} Days)")
                
                # Row 1: Primary Metrics
                rk1, rk2, rk3, rk4 = st.columns(4)
                sim_fr_val = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100
                rk1.metric("Simulated Fill Rate", f"{sim_fr_val:.1f}%")
                rk2.metric("Avg Physical Stock", f"{sdf['Physical_Stock'].mean():.1f} Units")
                rk3.metric("Avg Working Capital", f"${sdf['Physical_Stock'].mean() * u_val:,.0f}")
                rk4.metric("Orders Placed", int(sdf['OrderEvent'].sum()))
    
                # Row 2: Operational Health
                rk5, rk6, rk7, rk8 = st.columns(4)
                rk5.metric("Stockout Days", int((sdf['Physical_Stock'] == 0).sum()))
                rk6.metric("Total Stockout Units", int(sdf['Shortage'].sum()))
                rk7.metric("Min Inventory", f"{sdf['Physical_Stock'].min():.0f} Units")
                rk8.metric("Max Inventory", f"{sdf['Physical_Stock'].max():.0f} Units")
    
                # --- 4. MOVEMENT GRAPH ---
                st.subheader("📈 Inventory Movement Detail")
                fig_mov = go.Figure()
                fig_mov.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Total_Inventory"], name="Total Position", line=dict(color="#FFD700", dash='dash')))
                fig_mov.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Physical_Stock"], name="Physical On-Hand", fill='tozeroy', line=dict(color="#00FFCC")))
                fig_mov.add_hline(y=test_rop, line_dash="dot", line_color="orange", annotation_text=f"ROP: {test_rop}")
                
                stockouts = sdf[sdf['Shortage'] > 0]
                if not stockouts.empty:
                    fig_mov.add_trace(go.Scatter(x=stockouts["Day"], y=stockouts["Physical_Stock"], mode='markers', name="Stockout Event", marker=dict(color='red', size=8, symbol='x')))
                
                fig_mov.update_layout(template="plotly_dark", height=450, hovermode="x unified")
                st.plotly_chart(fig_mov, use_container_width=True)
    
                # --- 5. UNIFIED WORKING CAPITAL MAP ---
                st.subheader("🏦 Unified Working Capital Comparison ($ Value)")
                min_days = min(len(df_audited), len(sdf))
                fig_wc = go.Figure()
                fig_wc.add_trace(go.Scatter(x=list(range(min_days)), y=(df_audited['Closing Balance']*u_val).iloc[:min_days], 
                                           name="Historical (Blue)", fill='tozeroy', line=dict(color='rgba(99, 179, 237, 0.8)', width=2),
                                           fillcolor='rgba(99, 179, 237, 0.2)'))
                fig_wc.add_trace(go.Scatter(x=list(range(min_days)), y=(sdf['Physical_Stock']*u_val).iloc[:min_days], 
                                           name="Simulated (Orange)", fill='tozeroy', line=dict(color='rgba(255, 165, 0, 0.8)', width=2),
                                           fillcolor='rgba(255, 165, 0, 0.2)'))
                fig_wc.update_layout(template="plotly_dark", height=450, hovermode="x unified")
                st.plotly_chart(fig_wc, use_container_width=True)
    
                # --- 6. ANNUALIZED COMPARATIVE MATH ---
                orig_n_days = len(df_audited)
                orig_annual_factor = 365 / orig_n_days
                orig_avg_stock = df_audited['Closing Balance'].mean()
                orig_h = (orig_avg_stock * u_val * (h_pct/100)) + fixed_wh_cost
                orig_o = (df_audited['Order Received'].astype(bool).sum() * orig_annual_factor) * o_cost
                orig_l = (df_audited['Shortage'].sum() * orig_annual_factor) * u_val
                orig_fr = (1 - (df_audited['Shortage'].sum() / df_audited['Demand'].sum())) * 100
    
                sim_annual_factor = 365 / sim_days
                sim_avg_stock = sdf['Physical_Stock'].mean()
                sim_h = (sim_avg_stock * u_val * (h_pct/100)) + fixed_wh_cost
                sim_o = (sdf['OrderEvent'].sum() * sim_annual_factor) * o_cost
                sim_l = (sdf['Shortage'].sum() * sim_annual_factor) * u_val
                sim_fr = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100
    
                # --- 7. FINANCIAL TABLE ---
                st.subheader("💰 Annualized Comparative Financial Study")
                metrics = [
                    ("Avg Inventory (Units)", orig_avg_stock, sim_avg_stock, "lower"),
                    ("Avg Working Capital ($)", orig_avg_stock * u_val, sim_avg_stock * u_val, "lower"),
                    ("Global Fill Rate (%)", orig_fr, sim_fr, "higher"),
                    ("Annual Holding Cost ($)", orig_h, sim_h, "lower"),
                    ("Annual Ordering Cost ($)", orig_o, sim_o, "lower"),
                    ("Annual Lost Sales ($)", orig_l, sim_l, "lower"),
                    ("Total Annual Policy Cost ($)", (orig_h + orig_o + orig_l), (sim_h + sim_o + sim_l), "lower")
                ]
                
                rows = []
                for label, orig, sim, direction in metrics:
                    diff = ((sim - orig) / orig * 100) if orig != 0 else 0
                    color = "green" if (direction == "lower" and sim <= orig) or (direction == "higher" and sim >= orig) else "red"
                    rows.append({"Metric": label, "Original (Audit)": f"${orig:,.0f}" if "$" in label else f"{orig:,.1f}", 
                                 "Simulated (Test)": f"${sim:,.0f}" if "$" in label else f"{sim:,.1f}", "% Difference": f":{color}[{diff:+.1f}%]"})
                st.table(pd.DataFrame(rows))
    
                # --- 8. VERTICAL DATA TABLES ---
                st.divider()
                st.subheader("📋 Detailed Data Logs")
                with st.expander("📂 View Audited Historical Data (Healed)"):
                    st.dataframe(df_audited, use_container_width=True)
                with st.expander("📂 View Simulated Strategy Data"):
                    st.dataframe(sdf, use_container_width=True)
