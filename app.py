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
from insights_engine import InsightsGenerator
from report_generator import OptimizationReportGenerator

st.set_page_config(page_title="Investment Optimization Engine", layout="wide", page_icon="🏦")

st.title("🏦 Investment Optimization Engine")
st.markdown("*Enterprise Portfolio Optimization & Machine Learning Engine for 40%+ NPV Lift*")

# --- Streamlit Caching for High Speed Performance ---
@st.cache_data(ttl=1800, show_spinner=False)
def run_cached_pipeline(dataset_type, sample_size, budget, min_response_prob):
    config = Config()
    config.dataset_type = dataset_type
    config.hamzi_sample_size = sample_size
    config.default_budget = budget
    config.min_response_prob = min_response_prob
    config.n_estimators = 35  # Lightweight tree count for web execution
    
    loader = LargeDataLoader(config)
    data = loader.load_data()
    
    models = LargeScaleModels(data, config)
    metrics = models.train_models()
    
    optimizer = LargeScaleOptimizer(models.data, config)
    baseline = optimizer.run_baseline(budget=budget)
    optimized = optimizer.optimize_portfolio(budget=budget)
    comparison = optimizer.compare_results()
    df_imp = models.get_feature_importances()
    
    return data, metrics, optimized, comparison, df_imp, optimizer

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
        help="Number of client records to analyze."
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
        data, metrics, optimized, comparison, df_imp, optimizer = run_cached_pipeline(
            dataset_type, sample_size, budget, min_response_prob
        )
        
        selected_df = optimized['selected_clients'].copy()
        
        # Apply sidebar filters on view if columns exist
        if min_income > 0 and 'income' in selected_df.columns:
            selected_df = selected_df[selected_df['income'] >= min_income]
        if 'risk_tolerance' in selected_df.columns and risk_filter:
            selected_df = selected_df[selected_df['risk_tolerance'].isin(risk_filter)]

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Main Performance", 
        "📈 Risk-Return & Pareto", 
        "🗺️ Segment Heatmap",
        "🔍 Client Explorer",
        "🧠 Feature Importance", 
        "📥 Export Reports"
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

        # --- AI-Powered Automated Insights & Recommendations ---
        with st.expander("🧠 AI-Powered Business Insights & Strategic Recommendations", expanded=True):
            generator = InsightsGenerator(optimized, comparison)
            insights = generator.generate_all_insights()
            
            ic1, ic2 = st.columns(2)
            for i, insight in enumerate(insights):
                col_target = ic1 if i % 2 == 0 else ic2
                icon = {'success': '✅', 'warning': '⚠️', 'info': 'ℹ️', 'recommendation': '💡'}.get(insight['type'], '📌')
                with col_target:
                    st.markdown(f"**{icon} {insight['title']}**")
                    st.write(insight['description'])
                    st.caption(f"**Action:** {insight['action']}")
                    st.markdown("---")

    with tab2:
        st.subheader("Portfolio Efficient Frontier & Pareto Curve")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**1. Budget Sensitivity Frontier**")
            scenarios = optimizer.scenario_analysis(budgets=[250000, 500000, 1000000, 2500000, 5000000])
            fig1 = px.scatter(
                scenarios,
                x='total_cost',
                y='total_profit',
                size='clients_reached',
                color='avg_roi',
                title="Profit vs Marketing Spend Scale",
                labels={'total_cost': 'Budget Spent ($)', 'total_profit': 'Expected Return ($)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig1, width="stretch")
            
        with col2:
            st.markdown("**2. Cumulative Profit Pareto Curve (80/20 Analysis)**")
            if not selected_df.empty and 'expected_profit' in selected_df.columns:
                sorted_pareto = selected_df.sort_values('expected_profit', ascending=False).copy()
                sorted_pareto['cum_profit'] = sorted_pareto['expected_profit'].cumsum()
                sorted_pareto['client_rank'] = range(1, len(sorted_pareto) + 1)
                
                fig2 = px.line(
                    sorted_pareto,
                    x='client_rank',
                    y='cum_profit',
                    title="Cumulative Profit Accumulation"
                )
                tot_p = sorted_pareto['cum_profit'].max()
                fig2.add_hline(y=tot_p * 0.8, line_dash="dash", line_color="red", annotation_text="80% Profit Threshold")
                st.plotly_chart(fig2, width="stretch")

    with tab3:
        st.subheader("Segment Performance Heatmap")
        cat_cols = selected_df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) >= 2:
            row_cat = cat_cols[0]
            col_cat = cat_cols[1]
            pivot = selected_df.pivot_table(
                index=row_cat, 
                columns=col_cat, 
                values='expected_profit', 
                aggfunc='sum', 
                fill_value=0
            )
            fig = px.imshow(
                pivot,
                title=f"Total Profit Heatmap: {row_cat.replace('_',' ').title()} vs {col_cat.replace('_',' ').title()}",
                labels=dict(x=col_cat.title(), y=row_cat.title(), color="Total Profit ($)"),
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, width="stretch")
        elif len(cat_cols) == 1:
            fig = px.pie(selected_df, names=cat_cols[0], values='expected_profit', title=f"Profit Distribution by {cat_cols[0].title()}")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Segment heatmap requires categorical attributes.")

    with tab4:
        st.subheader("🔍 Interactive Client-Level Explorer")
        st.markdown("Filter and inspect targeted individual clients in real time.")
        
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            max_p_val = int(selected_df['predicted_npv'].max()) if 'predicted_npv' in selected_df.columns and len(selected_df) > 0 else 100000
            min_npv_val = st.slider("Min Predicted NPV ($)", 0, max_p_val, 0)
        with fc2:
            min_prob_val = st.slider("Min Response Probability", 0.0, 1.0, 0.3)
        with fc3:
            cat_filter_cols = selected_df.select_dtypes(include=['object', 'category']).columns
            selected_seg = None
            if len(cat_filter_cols) > 0:
                seg_col = cat_filter_cols[0]
                selected_seg = st.multiselect(f"Filter by {seg_col.title()}", selected_df[seg_col].unique(), default=selected_df[seg_col].unique())

        explorer_df = selected_df.copy()
        if 'predicted_npv' in explorer_df.columns:
            explorer_df = explorer_df[explorer_df['predicted_npv'] >= min_npv_val]
        if 'response_probability' in explorer_df.columns:
            explorer_df = explorer_df[explorer_df['response_probability'] >= min_prob_val]
        if selected_seg is not None and len(cat_filter_cols) > 0:
            explorer_df = explorer_df[explorer_df[cat_filter_cols[0]].isin(selected_seg)]

        st.dataframe(explorer_df.head(200), use_container_width=True)
        
        csv_filtered = explorer_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Filtered Client Subset ({len(explorer_df):,} clients)",
            data=csv_filtered,
            file_name="filtered_target_clients.csv",
            mime="text/csv"
        )

    with tab5:
        st.subheader("Feature Importance & Predictive Drivers")
        if df_imp is not None:
            fig = px.bar(
                df_imp.head(12),
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

    with tab6:
        st.subheader("Export Optimization Artifacts")
        st.markdown("Download targeted client lists, summary metrics, and executive multi-sheet Excel workbooks.")
        
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
        
        # Generate Excel workbook
        report_gen = OptimizationReportGenerator(optimized, comparison)
        excel_bytes = report_gen.export_excel_bytes()

        col1, col2, col3 = st.columns(3)
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
        with col3:
            st.download_button(
                label="📗 Download Executive Excel Workbook (.xlsx)",
                data=excel_bytes,
                file_name="optimization_results_workbook.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )

else:
    st.info("👈 Select your dataset and parameters in the sidebar, then click 'Run Optimization' to execute!")
    
    st.markdown("""
    ### About the Investment Optimization Engine
    
    This enterprise-grade application optimizes investment portfolios to maximize **Net Present Value (NPV)** lift under strict budget constraints.
    
    - 🎯 **Predictive ML Ensembles**: Predicts individual client NPV and conversion probability using Random Forest & Linear models.
    - ⚡ **Vectorized Knapsack Optimizer**: Achieves **>40% NPV Lift** in **<0.1 seconds**.
    - 🧠 **AI-Powered Insights**: Generates automated strategic business recommendations.
    - 📈 **Risk-Return Efficient Frontier**: Visualizes portfolio budget scale sensitivity & Pareto profit curves.
    - 📥 **Multi-Sheet Excel Reports**: Exports complete executive workbooks (`.xlsx`).
    - 🔌 **Production REST API**: Integrates seamlessly with FastAPI (`api.py`).
    """)