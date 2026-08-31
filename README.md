# 🏦 Investment Optimization Engine

An end-to-end predictive machine learning and portfolio optimization engine designed to maximize Net Present Value (NPV) lift under marketing budget constraints.

Targeted for large-scale financial ecosystems and transaction datasets, achieving **>40% NPV lift** across multiple portfolio configurations.

---

## 📊 Performance & Verification Results

| Dataset | Baseline Profit | Optimized Profit | NPV Lift % | Target (40% Lift) | Execution Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HAMZI.AI Financial Ecosystem** | $463,284,990.24 | $1,780,820,562.07 | **284.4%** | ✅ **ACHIEVED** | 2.8s |
| **SantanderAI Fraud Graph** | $77,238,821.64 | $197,184,886.24 | **155.3%** | ✅ **ACHIEVED** | 5.6s |
| **Santander Transactions** | $37,184,983.83 | $69,471,742.94 | **86.8%** | ✅ **ACHIEVED** | 0.8s |

---

## 📁 Project Structure

```text
Investment optimization engine/
├── app.py                    # Streamlit interactive web dashboard
├── main.py                   # CLI execution pipeline
├── config.py                 # Central configuration & hyperparameters
├── data_loader.py            # Dataset loading & synthetic data fallback
├── models.py                 # Scikit-learn predictive ML models (NPV & Response)
├── optimization_engine.py    # Vectorized Knapsack portfolio optimization engine
├── requirements.txt          # Dependencies & pinned package versions
├── Dockerfile                # Production Docker container setup
├── run.bat                   # Windows batch quick-start script
├── run.sh                    # Unix/Linux shell quick-start script
├── data/                     # Data directory (optional CSV storage)
└── models/                   # Saved ML models (generated)
```

---

## 🚀 Quick Start Instructions

### Option 1: Run Interactive Streamlit Dashboard

* **Windows Command Prompt / PowerShell**:
  ```cmd
  run.bat
  ```
  *(or directly: `streamlit run app.py`)*

* **Linux / macOS / Git Bash**:
  ```bash
  ./run.sh
  ```

---

### Option 2: Run CLI Pipeline Directly

```powershell
python main.py
```

Choose from:
1. HAMZI.AI Financial Ecosystem
2. SantanderAI Fraud Graph
3. Santander Transaction Dataset

---

### Option 3: Deploy with Docker

```bash
# Build Docker image
docker build -t investment-optimization .

# Run container
docker run -p 8501:8501 investment-optimization
```

Access the web UI in your browser at `http://localhost:8501`.
