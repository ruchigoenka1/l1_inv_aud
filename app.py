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
            st.warning("Please run Tab 2 (DNA Extraction) first to define the distribution parameters.")
        else:
            z_stats = st.session_state['zone_stats']
            df_full = st.session_state['dna_df']
            
            # --- 1. Simulation Inputs ---
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: 
                # Choice of seasonal DNA or the complete historical distribution
                options = ["All Data"] + list(z_stats['Zone'].unique())
                sel_zone = st.selectbox("Season to Simulate (Distribution Source)", options)
            
            # Pull stats based on choice
            if sel_zone == "All Data":
                avg_d, std_d = df_full['Demand'].mean(), df_full['Demand'].std()
                default_rop = int(avg_d * lt_manual + (1.65 * std_d)) 
            else:
                avg_d = df_full[df_full['Zone'] == sel_zone]['Demand'].mean()
                std_d = df_full[df_full['Zone'] == sel_zone]['Demand'].std()
                default_rop = int(z_stats.loc[z_stats['Zone']==sel_zone, 'ROP'].values[0])
    
            with sc2: test_rop = st.number_input("Test Reorder Point (ROP)", value=default_rop)
            with sc3: test_qty = st.number_input("Test Order Quantity (Q)", value=int(test_rop * 1.5 if test_rop > 0 else 100))
            with sc4: sim_days = st.slider("Simulation Duration (Days)", 30, 365, 90)
    
            if st.button("▶️ Run Stress Test"):
                # --- 2. Prorated Financial Math ---
                # H = (Value * Financial %) + Annual Fixed Overhead
                annual_h_unit = (u_val * (h_pct/100)) + fixed_wh_cost
                daily_h_rate = annual_h_unit / 365
                
                stocks, shortages, sim_demands, in_lt_sim, orders_placed_count, inventory_positions = [], [], [], [], [], []
                stock = test_rop + (test_qty / 2) # Assume starting in a healthy state
                p_orders = {}
    
                # --- 3. Simulation Loop ---
                for d in range(sim_days):
                    # Demand generation (Forced integer for revenue sync)
                    daily_d = int(max(0, np.random.normal(avg_d, std_d)))
                    
                    # Arrival Logic
                    recv = p_orders.pop(d, 0)
                    open_s = stock + recv
                    
                    # Fulfillment Logic
                    sales = min(open_s, daily_d)
                    shrt = int(daily_d - sales) # Strict integer for revenue sync
                    close_s = open_s - sales
                    
                    # Reorder Logic (Inventory Position = On Hand + On Order)
                    pipeline_sum = sum(p_orders.values())
                    inv_pos = close_s + pipeline_sum
                    
                    # Trigger Order only if no order is already in transit
                    order_triggered = 0
                    if inv_pos <= test_rop and pipeline_sum == 0:
                        p_orders[d + int(lt_manual)] = test_qty
                        order_triggered = 1
                        inv_pos += test_qty # Position jumps immediately
                    
                    stocks.append(close_s)
                    shortages.append(shrt)
                    sim_demands.append(daily_d)
                    in_lt_sim.append(pipeline_sum > 0)
                    orders_placed_count.append(order_triggered)
                    inventory_positions.append(inv_pos)
                    stock = close_s
    
                sdf = pd.DataFrame({
                    "Day": range(sim_days),
                    "Demand": sim_demands, 
                    "Physical_Stock": stocks, 
                    "Shortage": shortages, 
                    "InLT": in_lt_sim, 
                    "OrderEvent": orders_placed_count,
                    "Inv_Position": inventory_positions
                })
    
                # --- 4. Synchronized Master KPIs ---
                # Financials
                tot_h_cost = sdf['Physical_Stock'].sum() * daily_h_rate
                tot_o_cost = sdf['OrderEvent'].sum() * o_cost
                lost_rev = sdf['Shortage'].sum() * u_val
                total_cost = tot_h_cost + tot_o_cost + lost_rev
                
                # Operational
                stockout_days = int((sdf['Shortage'] > 0).sum())
                fill_rate = (1 - (sdf['Shortage'].sum() / sdf['Demand'].sum())) * 100 if sdf['Demand'].sum() > 0 else 100
                lt_sim_df = sdf[sdf['InLT'] == True]
                lt_fr = (1 - (lt_sim_df['Shortage'].sum() / lt_sim_df['Demand'].sum())) * 100 if not lt_sim_df.empty else 100
    
                # --- 5. KPI Display Dashboard ---
                st.subheader(f"🚀 Business Impact Dashboard ({sim_days} Days)")
                
                # Row 1: The "Profit & Loss" view
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric("Total Inventory Cost", f"${total_cost:,.0f}", help="Sum of Holding + Ordering + Lost Sales")
                r1c2.metric("Total Holding Cost", f"${tot_h_cost:,.0f}", help="Financial % + Fixed WH Overhead (Prorated Daily)")
                r1c3.metric("Total Ordering Cost", f"${tot_o_cost:,.0f}", help=f"{sdf['OrderEvent'].sum()} orders placed")
                r1c4.metric("Lost Sales Revenue", f"${lost_rev:,.0f}", delta=f"{int(sdf['Shortage'].sum())} Units", delta_color="inverse")
    
                # Row 2: The Operational view
                st.divider()
                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric("Stockout Days", stockout_days)
                r2c2.metric("Global Fill Rate", f"{fill_rate:.1f}%")
                r2c3.metric("LT Fill Rate", f"{lt_fr:.1f}%", help="Fill rate while orders are in the pipeline")
                r2c4.metric("Avg WC Investment", f"${(sdf['Physical_Stock'].mean() * u_val):,.0f}")
    
                # --- 6. Enhanced Visualizations ---
                st.subheader("📈 Inventory Strategy: Physical vs. Pipeline")
                fig = go.Figure()
                # Pipeline Position
                fig.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Inv_Position"], name="Inventory Position (Inc. Pipeline)", 
                                         line=dict(color="rgba(255, 255, 255, 0.4)", dash="dot")))
                # Physical Stock
                fig.add_trace(go.Scatter(x=sdf["Day"], y=sdf["Physical_Stock"], name="Physical Stock", 
                                         line=dict(color="#00FFCC", width=3), fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.1)'))
                # Demand Bars
                fig.add_trace(go.Bar(x=sdf["Day"], y=sdf["Demand"], name="Daily Demand", opacity=0.2, marker_color="white"))
                # ROP Horizontal Line
                fig.add_hline(y=test_rop, line_dash="dash", line_color="orange", annotation_text=f"Reorder Point: {test_rop}")
                
                # Mark Stockout Events with red X
                stockouts = sdf[sdf['Shortage'] > 0]
                if not stockouts.empty:
                    fig.add_trace(go.Scatter(x=stockouts["Day"], y=[0]*len(stockouts), mode='markers', name="Stockout", marker=dict(color='red', size=10, symbol='x')))
                
                fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
    
                # Strategy Guidance
                annual_d_proj = avg_d * 365
                eoq_val = np.sqrt((2 * annual_d_proj * o_cost) / annual_h_unit)
                st.info(f"💡 AI Benchmarking: For this {sel_zone} distribution, the optimal EOQ is **{int(eoq_val)}**. Your test Q is {test_qty}.")
    
                # --- 7. Simulated Data Table ---
                st.divider()
                with st.expander("📋 Download & View Detailed Simulation Data"):
                    st.dataframe(sdf, use_container_width=True)
                    # Export option
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        sdf.to_excel(writer, index=False)
                    st.download_button("📥 Download Simulation Excel", data=buffer, file_name=f"stress_test_{sel_zone}.xlsx")
else:
    st.info("👋 Upload historical Excel file to begin.")
