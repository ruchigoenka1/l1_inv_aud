import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------
# 1. Page Config & Styling
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Auditor Pro")

st.markdown(
    """
    <style>
    .block-container { padding: 2rem 5rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    section[data-testid="stSidebar"] .stButton button {
        background-color: #2E7D32 !important; color: white !important; width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 2. Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Participant Excel", type=["xlsx"])

st.sidebar.divider()
st.sidebar.header("Audit Parameters")
unit_value = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)
lead_time_manual = st.sidebar.number_input("Standard Lead Time (Days)", value=3, step=1)

# ------------------------------------------------
# 3. Data Processing Engine
# ------------------------------------------------
def process_audit(df, lt, u_val, h_pct, o_cost):
    # Standardize Column Names (Handling case sensitivity)
    df.columns = [c.strip().title() for c in df.columns]
    
    # Fill missing columns
    if "Order Placed" not in df.columns:
        # Back-calculate: If received on Day 5 with LT 3, placed on Day 2
        df["Order Placed"] = df["Order Received"].shift(-lt).fillna(0)
    
    # Calculate Service Metrics
    df['Shortage'] = np.maximum(0, df['Demand'] - (df['Opening Balance'] + df['Order Received']))
    df['IsStockout'] = df['Shortage'] > 0
    
    # Calculate Costs
    daily_h_rate = (h_pct / 100) / 365
    df['HoldingCost'] = df['Closing Balance'] * u_val * daily_h_rate
    df['OrderingCost'] = np.where(df['Order Placed'] > 0, o_cost, 0)
    
    # Identify Lead Time Windows (Shading Logic)
    # A day is "In Lead Time" if an order has been placed but not yet fully received
    df['InLT'] = False
    for idx, row in df.iterrows():
        if row['Order Placed'] > 0:
            df.loc[idx : idx + lt - 1, 'InLT'] = True
            
    return df

# ------------------------------------------------
# 4. Main App Logic
# ------------------------------------------------
if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df = process_audit(raw_df, lead_time_manual, unit_value, holding_cost_pct, ordering_cost)
    
    # --- Tabs Integration ---
    t1, t2 = st.tabs(["📊 Performance Audit", "📈 Risk & Window Analysis"])

    with t1:
        # KPI Row
        avg_demand = df['Demand'].mean()
        total_demand = df['Demand'].sum()
        fill_rate = (1 - (df['Shortage'].sum() / total_demand)) * 100 if total_demand > 0 else 100
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", int(df['IsStockout'].sum()))
        k2.metric("Actual Fill Rate", f"{fill_rate:.1f}%")
        k3.metric("Avg Inventory", f"{df['Closing Balance'].mean():.1f}")
        k4.metric("Total Policy Cost", f"${(df['HoldingCost'].sum() + df['OrderingCost'].sum()):,.0f}")

        # EOQ Comparison Expander
        with st.expander("💰 EOQ Benchmarking", expanded=False):
            annual_d = avg_demand * 365
            annual_h = unit_value * (holding_cost_pct / 100)
            eoq_val = np.sqrt((2 * annual_d * ordering_cost) / annual_h)
            st.write(f"Based on this data, your **Economic Order Quantity (EOQ)** should be: **{int(eoq_val)} units**.")
            st.info("Compare this to the average 'Order Placed' quantity in your file.")

        # Main Chart with Shading and Markers
        st.subheader("Inventory Flow & Lead Time Windows")
        fig = go.Figure()
        
        # Lead Time Shading
        fig.add_trace(go.Scatter(
            x=df["Date"], 
            y=np.where(df["InLT"], df["Closing Balance"].max(), np.nan),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.05)', 
            line=dict(width=0), name="Lead Time Window", showlegend=True
        ))

        # Balance Line
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Stock", line=dict(color='#00CCFF', width=2)))
        
        # Event Markers
        placed = df[df["Order Placed"] > 0]
        fig.add_trace(go.Scatter(x=placed["Date"], y=placed["Closing Balance"], mode="markers", 
                                 name="Order Placed", marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
        
        shorts = df[df["IsStockout"]]
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Closing Balance"], mode="markers", 
                                 name="Stockout Event", marker=dict(color="red", size=10, symbol="x")))

        fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Audit Log")
        st.dataframe(df, use_container_width=True)

    with t2:
        st.subheader("Risk & Distribution Analysis")
        window_size = st.slider("Analysis Window (Days)", 1, 30, 5)
        rolling_demand = df['Demand'].rolling(window=window_size).sum().dropna()
        
        if not rolling_demand.empty:
            target_sl = st.slider("Target Service Level", 0.80, 0.99, 0.95)
            cutoff = np.percentile(rolling_demand, target_sl * 100)
            
            fig_risk = px.histogram(rolling_demand, nbins=20, title="Demand Volatility Window", color_discrete_sequence=['#00CC96'])
            fig_risk.add_vline(x=cutoff, line_dash="dash", line_color="yellow", annotation_text="Target SL")
            fig_risk.update_layout(template="plotly_dark")
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.metric("Required Safety Stock for Window", f"{int(cutoff)} Units")
else:
    st.info("👋 Welcome! Please upload your Excel file to begin the audit.")
    st.write("Required columns: **Date, Opening Balance, Demand, Order Received, Closing Balance**")
