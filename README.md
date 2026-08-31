# 🏦 Investment Optimization Engine

An enterprise predictive machine learning and portfolio optimization engine designed to maximize Net Present Value (NPV) lift under marketing budget constraints.

Targeted for large-scale financial ecosystems and transaction datasets, achieving **>40% NPV lift** across multiple portfolio configurations.

🔗 **Live Interactive App**: [https://optimizeinvest.streamlit.app/](https://optimizeinvest.streamlit.app/)

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
├── api.py                    # Production FastAPI REST backend endpoint
├── config.py                 # Central configuration & hyperparameters
├── data_loader.py            # Dataset handling, imputation & feature engineering
├── models.py                 # ML models (Random Forest, Linear & Feature Importances)
├── optimization_engine.py    # Vectorized Knapsack portfolio optimization engine
├── requirements.txt          # Dependencies & pinned package versions
├── Dockerfile                # Production Docker container setup
├── render.yaml               # Render Cloud deployment blueprint
├── fly.toml                  # Fly.io deployment config
├── railway.json              # Railway deployment config
├── run.bat                   # Windows batch quick-start script
├── run.sh                    # Unix/Linux shell quick-start script
├── data/                     # Data directory (optional CSV storage)
└── models/                   # Saved ML models (generated)
```

---

## 🚀 Quick Start Instructions

### 1. Run Interactive Streamlit Dashboard

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

### 2. Run CLI Pipeline

```powershell
python main.py
```

---

### 3. Run FastAPI Production REST Server

```powershell
uvicorn api:app --reload --port 8000
```

---

## ☁️ Cloud Deployment Options

### A. Deploy to Render (Free Cloud)
1. Push code to your GitHub repository: `https://github.com/Akshit8459/Investment-optimization-engine`
2. Connect your repo at [Render.com](https://render.com).
3. Render automatically reads [`render.yaml`](file:///c:/Users/akshi/Desktop/Investment%20optimization%20engine/render.yaml) and deploys your Docker container.

### B. Deploy to Streamlit Community Cloud
1. Push your repository to GitHub.
2. Visit [share.streamlit.io/deploy](https://share.streamlit.io/deploy).
3. Select your repo, branch (`main`), and set Main file path to `app.py`.

### C. Deploy to Railway
```bash
railway init
railway up
```

### D. Deploy to Fly.io
```bash
flyctl launch
flyctl deploy
```

---

### E. Deploy with Local Docker

```bash
# Build Docker image
docker build -t investment-optimization .

# Run container
docker run -p 8501:8501 investment-optimization
```

Access the web UI in your browser at `http://localhost:8501`.
