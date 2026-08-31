# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from io import BytesIO

from config import Config
from data_loader import LargeDataLoader
from models import LargeScaleModels
from optimization_engine import LargeScaleOptimizer

st.set_page_config(page_title="Investment Optimization Engine", layout="wide", page_icon="🏦")

st.title("🏦 Investment Optimization Engine")
st.markdown("*Enterprise Portfolio Optimization & Machine Learning Engine for 40%+ NPV Lift*")

# --- Streamlit Caching for Ultra-Low CPU Usage & High Speed ---
@st.cache_data(ttl=1800, show_spinner=False)
def run_cached_pipeline(dataset_type, sample_size, budget, min_response_prob):
    config = Config()
    config.dataset_type = dataset_type
    config.hamzi_sample_size = sample_size
    config.default_budget = budget
    config.min_response_prob = min_response_prob
    config.n_estimators = 35  # Lightweight tree count optimized for cloud execution
    
    loader = LargeDataLoader(config)
    data = loader.load_data()
    
    models = LargeScaleModels(data, config)
    metrics = models.train_models()
    
    optimizer = LargeScaleOptimizer(models.data, config)
    baseline = optimizer.run_baseline(budget=budget)
    optimized = optimizer.optimize_portfolio(budget=budget)
    comparison = optimizer.compare_results()
    df_imp = models.get_feature_importances()
    
    return data, metrics, optimized, comparison, df_imp

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["HAMZI.AI Financial Ecosystem", "SantanderAI Fraud Graph", "Santander Transactions"],
        help="Choose from wealth management, fraud graph, or transactional banking datasets."
    )
    
    sample_size = st.slider(
        "Sample Size",
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
        help="Number of client records to analyze (lower sample size runs faster)."
    )
    
    budget = st.number_input(
        "Marketing Budget ($)",
        min_value=100000,
        max_value=10000000,
        value=1000000,
        step=100000,
        help="Total budget available for targeting clients."
    )
    
    min_response_prob = st.slider(
        "Minimum Response Probability",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Threshold probability to filter responsive candidates."
    )

    st.markdown("---")
    st.header("🔍 Advanced Filters")
    min_income = st.number_input("Minimum Income ($)", value=0, step=10000)
    risk_filter = st.multiselect("Risk Tolerance", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

    st.markdown("---")
    run_optimization = st.button("🚀 Run Optimization", width="stretch")

# Map dataset names
dataset_map = {
    "HAMZI.AI Financial Ecosystem": "hamzi",
    "SantanderAI Fraud Graph": "fraud_graph",
    "Santander Transactions": "santander"
}

# --- Main Dashboard ---
if run_optimization:
    with st.spinner("Executing Feature Engineering, ML Training, & Knapsack Portfolio Optimization..."):
        dataset_type = dataset_map[dataset_choice]
        data, metrics, optimized, comparison, df_imp = run_cached_pipeline(
            dataset_type, sample_size, budget, min_response_prob
        )
        
        # Apply sidebar filters on view if columns exist
        if min_income > 0 and 'income' in data.columns:
            data = data[data['income'] >= min_income]
        if 'risk_tolerance' in data.columns and risk_filter:
            data = data[data['risk_tolerance'].isin(risk_filter)]

    # --- Top KPI Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Profit (Optimized)",
            f"${comparison['optimized_profit']:,.0f}",
            delta=f"+${comparison['optimized_profit'] - comparison['baseline_profit']:,.0f}"
        )
    with col2:
        st.metric(
            "NPV Lift",
            f"{comparison['npv_lift_percent']:.1f}%",
            delta="Target: 40% (Achieved)" if comparison['meets_target'] else "Below Target"
        )
    with col3:
        st.metric(
            "ROI (Optimized)",
            f"{comparison['optimized_roi']:.2f}x",
            delta=f"+{comparison['optimized_roi'] - comparison['baseline_roi']:.2f}x"
        )
    with col4:
        st.metric(
            "Clients Selected",
            f"{comparison['optimized_clients']:,}",
            delta=f"+{comparison['optimized_clients'] - comparison['baseline_clients']}"
        )

    st.markdown("---")

    # --- Tabbed Deep-Dive Analytics ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Main Performance", 
        "🧠 Feature Importance", 
        "📥 Export Results"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Baseline', 'Optimized'],
                y=[comparison['baseline_profit'], comparison['optimized_profit']],
                text=[f'${v:,.0f}' for v in [comparison['baseline_profit'], comparison['optimized_profit']]],
                textposition='outside',
                marker_color=['#FF6B6B', '#51CF66']
            ))
            fig.update_layout(title="Total Expected Profit Comparison ($)", yaxis_title="Profit ($)", height=380)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Baseline', 'Optimized'],
                y=[comparison['baseline_roi'], comparison['optimized_roi']],
                text=[f'{v:.2f}x' for v in [comparison['baseline_roi'], comparison['optimized_roi']]],
                textposition='outside',
                marker_color=['#4DABF7', '#37B24D']
            ))
            fig.update_layout(title="Return on Investment (ROI Multiplier)", yaxis_title="ROI Ratio", height=380)
            st.plotly_chart(fig, width="stretch")

        with st.expander("📋 Detailed Metrics & Model Diagnostics", expanded=True):
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**Machine Learning Accuracy**")
                st.write(f"• NPV Regressor R²: **{metrics['npv_r2']:.4f}**")
                st.write(f"• Response Classifier AUC: **{metrics['response_auc']:.4f}**")
                st.write(f"• Records Analyzed: **{len(data):,}**")
            with m2:
                st.markdown("**Optimization Execution**")
                st.write(f"• Algorithm: **{optimized['status']}**")
                st.write(f"• Marketing Budget Spent: **${comparison['optimized_cost']:,.2f}**")
                st.write(f"• Total Profit Generated: **${comparison['objective_value']:,.2f}**")
            with m3:
                st.markdown("**Target Status**")
                if comparison['meets_target']:
                    st.success("✅ **40%+ NPV Lift Target Achieved!**")
                else:
                    st.warning("⚠️ **40% Target Not Yet Achieved**")

    with tab2:
        st.subheader("Feature Importance & Predictive Drivers")
        if df_imp is not None:
            fig = px.bar(
                df_imp.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top Features Driving Client NPV Predictions",
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Feature importance analysis is available for Random Forest models.")

    with tab3:
        st.subheader("Export Optimization Artifacts")
        st.markdown("Download targeted client lists and executive summary reports for campaign execution.")
        
        selected_df = optimized['selected_clients']
        csv_selected = selected_df.to_csv(index=False).encode('utf-8')
        
        summary_df = pd.DataFrame([{
            'Metric': 'Baseline Profit', 'Value': f"${comparison['baseline_profit']:,.2f}"
        }, {
            'Metric': 'Optimized Profit', 'Value': f"${comparison['optimized_profit']:,.2f}"
        }, {
            'Metric': 'NPV Lift', 'Value': f"{comparison['npv_lift_percent']:.1f}%"
        }, {
            'Metric': 'Clients Selected', 'Value': f"{comparison['optimized_clients']:,}"
        }, {
            'Metric': 'Target Achieved', 'Value': str(comparison['meets_target'])
        }])
        csv_summary = summary_df.to_csv(index=False).encode('utf-8')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download Selected Clients (CSV)",
                data=csv_selected,
                file_name="optimized_portfolio_clients.csv",
                mime="text/csv",
                width="stretch"
            )
        with col2:
            st.download_button(
                label="📊 Download Summary Report (CSV)",
                data=csv_summary,
                file_name="optimization_summary_report.csv",
                mime="text/csv",
                width="stretch"
            )

else:
    st.info("👈 Select your dataset and parameters in the sidebar, then click 'Run Optimization' to execute!")
    
    st.markdown("""
    ### About the Investment Optimization Engine
    
    This enterprise-grade application optimizes investment portfolios to maximize **Net Present Value (NPV)** lift under strict budget constraints.
    
    - 🎯 **Predictive ML Ensembles**: Predicts individual client NPV and conversion probability using Random Forest & Linear models.
    - ⚡ **Vectorized Knapsack Optimizer**: Achieves **>40% NPV Lift** in **<0.1 seconds**.
    - 🧠 **Feature Importance Analysis**: Visualizes top drivers behind NPV predictions.
    - 🔌 **Production REST API**: Integrates seamlessly with FastAPI (`api.py`).
    """)