import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from Fundamental.test_fundamental import BacktestFundamental
from DataPreprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from Environment.backtest import TD3Backtester

# Page configuration
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Sidebar configuration
st.sidebar.title("📊 Configuration")

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

PATHS = {
    'NIFTY_URL': "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    'MODEL_PATH': os.path.join(MODELS_DIR, "xgboost_model.bin"),
    'TD3_MODEL_PATH': os.path.join(MODELS_DIR, "td3_stock_trading_final.zip"),
    'ENV_PATH': os.path.join(MODELS_DIR, "td3_stock_trading_env_final.pkl"),
    'OUTPUT_DIR': os.path.join(BASE_DIR, "backtest_results")
}

# Main dashboard
st.title("🤖 AI-Powered Trading Dashboard")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📈 Stock Selection", "🔄 Data Processing", "📊 Backtest Results"])

with tab1:
    st.header("Stock Selection & Fundamental Analysis")
    
    num_stocks = st.slider(
        "Number of Stocks to Analyze",
        min_value=10,
        max_value=400,
        value=50,
        step=10,
        help="Select the number of top stocks to analyze based on fundamental factors"
    )
    
    if st.button("🔍 Run Fundamental Analysis"):
        with st.spinner("Running fundamental analysis..."):
            backtest = BacktestFundamental(PATHS['NIFTY_URL'], PATHS['MODEL_PATH'], num_stocks)
            top_stocks = backtest.run_backtest()
            
            with st.expander("View Top Stocks Analysis", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(top_stocks.head(10))
                with col2:
                    if 'Sector' in top_stocks.columns:
                        sector_dist = top_stocks['Sector'].value_counts()
                        fig = px.pie(values=sector_dist.values, names=sector_dist.index, title="Sector Distribution")
                        st.plotly_chart(fig)
            
            top_stocks.to_csv(os.path.join(BASE_DIR, "top_10_stocks.csv"), index=False)
            st.success("✅ Fundamental analysis completed!")

with tab2:
    st.header("Data Processing Pipeline")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            help="Select the start date for historical data"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Select the end date for historical data"
        )
    
    if st.button("🔄 Process Market Data"):
        if os.path.exists(os.path.join(BASE_DIR, "top_10_stocks.csv")):
            with st.spinner("Processing market data..."):
                pipeline = StockPreProcessPipeline()
                ticker_list = pd.read_csv(os.path.join(BASE_DIR, "top_10_stocks.csv"))["Ticker"].tolist()
                
                combined_csv_path = pipeline.run_data_pipeline(
                    ticker_list,
                    str(start_date),
                    str(end_date)
                )
                
                st.success("✅ Data processing completed!")
                st.info(f"Data saved to: {combined_csv_path}")
        else:
            st.error("⚠️ Please run fundamental analysis first!")

with tab3:
    st.header("Backtest Results & Analysis")
    
    if st.button("🚀 Run Trading Backtest"):
        if os.path.exists(os.path.join(BASE_DIR, "combined_stock_data.csv")):
            with st.spinner("Running trading backtest..."):
                backtester = TD3Backtester(
                    model_path=PATHS['TD3_MODEL_PATH'],
                    env_path=PATHS['ENV_PATH'],
                    output_dir=PATHS['OUTPUT_DIR']
                )
                
                results = backtester.run_backtest(os.path.join(BASE_DIR, "combined_stock_data.csv"))
                st.session_state.backtest_results = results
                
                st.subheader("📊 Performance Metrics")
                metrics = results['metrics']
                cols = st.columns(4)
                for i, (metric, value) in enumerate(metrics.items()):
                    with cols[i % 4]:
                        st.metric(metric, value)
                
                st.subheader("📈 Portfolio Analysis")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['df'].index,
                    y=results['df']['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue')
                ))
                fig.update_layout(title='Portfolio Value Over Time')
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.histogram(
                    results['df']['daily_returns'].dropna(),
                    nbins=50,
                    title='Distribution of Daily Returns'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.bar(
                    results['df'],
                    x=results['df'].index,
                    y='trades',
                    title='Daily Trading Activity'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Backtest completed!")
        else:
            st.error("⚠️ Please process market data first!")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit and Python*")