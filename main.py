# main.py
import pandas as pd
from config import Config
from data_loader import LargeDataLoader
from models import LargeScaleModels
from optimization_engine import LargeScaleOptimizer
import time

import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_pipeline(dataset_type='hamzi', sample_size=50000):
    """
    Main pipeline for large-scale investment optimization
    """
    print(f"\n{'='*60}")
    print(f"Running Investment Optimization Pipeline")
    print(f"Dataset: {dataset_type}")
    print(f"Sample size: {sample_size:,}")
    print(f"{'='*60}\n")
    
    # 1. Load data
    config = Config()
    config.dataset_type = dataset_type
    config.hamzi_sample_size = sample_size
    
    print("STEP 1: Loading data...")
    loader = LargeDataLoader(config)
    data = loader.load_data()
    print(f"[OK] Loaded {len(data):,} records")
    
    # 2. Train models
    print("\nSTEP 2: Training predictive models...")
    start_time = time.time()
    
    models = LargeScaleModels(data, config)
    metrics = models.train_models()
    
    print(f"[OK] Models trained in {time.time() - start_time:.1f} seconds")
    print(f"  NPV R²: {metrics['npv_r2']:.4f}")
    print(f"  Response AUC: {metrics['response_auc']:.4f}")
    
    # 3. Run optimization
    print("\nSTEP 3: Running portfolio optimization...")
    start_time = time.time()
    
    optimizer = LargeScaleOptimizer(models.data, config)
    baseline = optimizer.run_baseline()
    optimized = optimizer.optimize_portfolio()
    
    print(f"[OK] Optimization completed in {time.time() - start_time:.1f} seconds")
    
    # 4. Compare results
    print("\nSTEP 4: Analyzing results...")
    comparison = optimizer.compare_results()
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline Profit:     ${comparison['baseline_profit']:,.2f}")
    print(f"Optimized Profit:    ${comparison['optimized_profit']:,.2f}")
    print(f"NPV Lift:            {comparison['npv_lift_percent']:.1f}%")
    print(f"Target (40% Lift):   {'ACHIEVED' if comparison['meets_target'] else 'NOT ACHIEVED'}")
    print(f"\nROI Comparison:")
    print(f"  Baseline: {comparison['baseline_roi']:.2f}x")
    print(f"  Optimized: {comparison['optimized_roi']:.2f}x")
    print(f"{'='*60}")
    
    return {
        'data': data,
        'models': models,
        'optimizer': optimizer,
        'comparison': comparison
    }

if __name__ == "__main__":
    # Run with HAMZI dataset
    # results = run_pipeline('hamzi', sample_size=50000)
    
    # Run with Fraud Graph dataset
    # results = run_pipeline('fraud_graph', sample_size=100000)
    
    # Or run with both
    print("Select dataset:")
    print("1. HAMZI.AI Financial Ecosystem")
    print("2. SantanderAI/gen-fraud-graph")
    print("3. Santander Transaction Dataset")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == '1':
        results = run_pipeline('hamzi', 50000)
    elif choice == '2':
        results = run_pipeline('fraud_graph', 100000)
    elif choice == '3':
        results = run_pipeline('santander', 20000)
    else:
        print("Invalid choice")