#!/bin/bash
# run.sh

echo "Investment Optimization Engine"
echo "================================"

# Check for data
if [ ! -d "data" ]; then
    mkdir data
    echo "Please download dataset and place in data/ directory"
    echo "For HAMZI: financial_ecosystem.csv"
    echo "For Santander: train.csv"
fi

# Install dependencies
pip install -r requirements.txt

# Run with streamlit
streamlit run app.py --server.port=8501
