# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Dataset selection
    dataset_type: str = "hamzi"  # Options: "hamzi", "fraud_graph", "santander"
    
    # File paths
    data_path: str = "data/"
    models_path: str = "models/"
    results_path: str = "results/"
    
    # Dataset-specific settings
    hamzi_sample_size: Optional[int] = 50000  # Use subset for prototyping
    fraud_graph_sample_size: Optional[int] = 100000
    
    # Model settings
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    
    # Optimization settings
    default_budget: float = 1000000
    min_response_prob: float = 0.3
    
    # Performance settings
    use_dask: bool = True  # Enable Dask for large datasets
    batch_size: int = 10000  # For chunked processing

config = Config()