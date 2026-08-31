# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from config import Config
from data_loader import LargeDataLoader
from models import LargeScaleModels
from optimization_engine import LargeScaleOptimizer

app = FastAPI(
    title="Investment Optimization Engine API",
    description="Production REST API for portfolio optimization & NPV lift targeting",
    version="1.0.0"
)

class OptimizationRequest(BaseModel):
    dataset: str = "hamzi"
    budget: float = 1000000.0
    min_response_prob: float = 0.3
    sample_size: int = 50000

class OptimizationResponse(BaseModel):
    baseline_profit: float
    optimized_profit: float
    npv_lift_percent: float
    clients_reached: int
    meets_target: bool

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Investment Optimization Engine API v1.0",
        "target": "40%+ NPV Lift"
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio_endpoint(request: OptimizationRequest):
    try:
        config = Config()
        config.dataset_type = request.dataset
        config.default_budget = request.budget
        config.min_response_prob = request.min_response_prob
        config.hamzi_sample_size = request.sample_size
        
        loader = LargeDataLoader(config)
        data = loader.load_data()
        
        models = LargeScaleModels(data, config)
        models.train_models()
        
        optimizer = LargeScaleOptimizer(models.data, config)
        optimizer.run_baseline(budget=request.budget)
        optimizer.optimize_portfolio(budget=request.budget)
        comparison = optimizer.compare_results()
        
        return OptimizationResponse(
            baseline_profit=comparison['baseline_profit'],
            optimized_profit=comparison['optimized_profit'],
            npv_lift_percent=comparison['npv_lift_percent'],
            clients_reached=comparison['optimized_clients'],
            meets_target=comparison['meets_target']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
