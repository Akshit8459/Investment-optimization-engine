@echo off
echo Investment Optimization Engine
echo ================================

REM Check for data
if not exist "data" (
    mkdir data
    echo Please download dataset and place in data/ directory
    echo For HAMZI: financial_ecosystem.csv
    echo For Santander: train.csv
)

REM Install dependencies
pip install -r requirements.txt

REM Run with streamlit
streamlit run app.py --server.port=8501
