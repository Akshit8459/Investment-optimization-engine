# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from config import Config
from data_loader import LargeDataLoader
from models import LargeScaleModels
from optimization_engine import LargeScaleOptimizer

st.set_page_config(page_title="Investment Optimization Engine", layout="wide")

st.title("🏦 Investment Optimization Engine")
st.markdown("*Large-scale portfolio optimization with 40% NPV lift target*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["HAMZI.AI Financial Ecosystem", "SantanderAI Fraud Graph", "Santander Transactions"]
    )
    
    sample_size = st.slider(
        "Sample Size",
        min_value=1000,
        max_value=100000,
        value=50000,
        step=1000
    )
    
    budget = st.number_input(
        "Marketing Budget ($)",
        min_value=100000,
        max_value=10000000,
        value=1000000,
        step=100000
    )
    
    min_response_prob = st.slider(
        "Minimum Response Probability",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05
    )
    
    run_optimization = st.button("🚀 Run Optimization", width="stretch")

# Main content
if run_optimization:
    with st.spinner("Loading data and training models..."):
        # Map selection
        dataset_map = {
            "HAMZI.AI Financial Ecosystem": "hamzi",
            "SantanderAI Fraud Graph": "fraud_graph",
            "Santander Transactions": "santander"
        }
        
        config = Config()
        config.dataset_type = dataset_map[dataset_choice]
        config.hamzi_sample_size = sample_size
        config.default_budget = budget
        config.min_response_prob = min_response_prob
        
        # Load data
        loader = LargeDataLoader(config)
        data = loader.load_data()
        
        # Train models
        models = LargeScaleModels(data, config)
        metrics = models.train_models()
        
        # Optimize
        optimizer = LargeScaleOptimizer(models.data, config)
        baseline = optimizer.run_baseline(budget=budget)
        optimized = optimizer.optimize_portfolio(budget=budget)
        
        # Compare
        comparison = optimizer.compare_results()
    
    # Display results
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
            delta="Target: 40%" if comparison['meets_target'] else "Below Target"
        )
    
    with col3:
        st.metric(
            "ROI (Optimized)",
            f"{comparison['optimized_roi']:.2f}x",
            delta=f"+{comparison['optimized_roi'] - comparison['baseline_roi']:.2f}x"
        )
    
    with col4:
        st.metric(
            "Clients Reached",
            f"{comparison['optimized_clients']:,}",
            delta=f"+{comparison['optimized_clients'] - comparison['baseline_clients']}"
        )
    
    # Charts
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
        fig.update_layout(
            title="Profit Comparison",
            yaxis_title="Total Profit ($)",
            height=400
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Baseline', 'Optimized'],
            y=[comparison['baseline_roi'], comparison['optimized_roi']],
            text=[f'{v:.2f}x' for v in [comparison['baseline_roi'], comparison['optimized_roi']]],
            textposition='outside',
            marker_color=['#FF6B6B', '#51CF66']
        ))
        fig.update_layout(
            title="ROI Comparison",
            yaxis_title="Return on Investment",
            height=400
        )
        st.plotly_chart(fig, width="stretch")
    
    # Detailed metrics
    with st.expander("📊 Detailed Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Performance**")
            st.write(f"• NPV R²: {metrics['npv_r2']:.4f}")
            st.write(f"• Response AUC: {metrics['response_auc']:.4f}")
            st.write(f"• Dataset Size: {len(data):,} records")
        
        with col2:
            st.markdown("**Optimization Results**")
            st.write(f"• Objective Value: ${comparison['objective_value']:,.0f}")
            st.write(f"• Status: {optimized['status']}")
            st.write(f"• Clients Selected: {optimized['clients_reached']:,}")
        
        with col3:
            st.markdown("**Target Achievement**")
            if comparison['meets_target']:
                st.success("✅ 40% NPV Lift Target Achieved!")
            else:
                st.warning("⚠️ 40% NPV Lift Target Not Yet Achieved")
    
    # Client distribution
    st.subheader("Selected Client Distribution")
    
    selected_df = optimized['selected_clients']
    
    # Find categorical column for distribution
    cat_cols = selected_df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        col = st.selectbox("Select segment", cat_cols)
        
        fig = px.pie(
            selected_df,
            names=col,
            title=f"Selected Clients by {col}",
            hole=0.3
        )
        st.plotly_chart(fig, width="stretch")
else:
    st.info("👈 Configure settings and click 'Run Optimization' to start")
    
    # Show welcome info
    st.markdown("""
    ### Welcome to the Investment Optimization Engine
    
    This engine demonstrates how FischerJordan's portfolio optimization methodology 
    can achieve up to 40% NPV lift through:
    
    - **Predictive Analytics**: ML models predict client NPV and response probability
    - **Optimization**: Linear programming allocates marketing budget optimally
    - **Interactive Dashboard**: Visualize before/after performance metrics
    
    Select your dataset and parameters in the sidebar to begin!
    """)
    
    # Show sample data preview
    if st.checkbox("Show sample data"):
        config = Config()
        loader = LargeDataLoader(config)
        data = loader.load_data()
        st.dataframe(data.head(100))