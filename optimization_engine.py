# optimization_engine.py
import pandas as pd
import numpy as np
from pulp import (
    LpProblem,
    LpMaximize,
    LpVariable,
    LpBinary,
    lpSum,
    PULP_CBC_CMD,
    LpStatusOptimal,
    LpStatus,
    value
)
import warnings
warnings.filterwarnings('ignore')

class LargeScaleOptimizer:
    def __init__(self, data, config):
        self.data = data.copy()
        self.config = config
        self.optimization_results = None
        self.baseline_results = None
        
        # Sample data if too large for optimization
        self._prepare_optimization_data()
    
    def _prepare_optimization_data(self):
        """Reduce data size for optimization if needed"""
        max_records = 50000  # PuLP performance limit
        if len(self.data) > max_records:
            print(f"Data too large ({len(self.data)} records). " 
                  f"Sampling {max_records} for optimization.")
            self.data = self.data.sample(max_records, 
                                        random_state=self.config.random_state)
    
    def run_baseline(self, budget=None):
        """Run baseline with no optimization"""
        if budget is None:
            budget = self.config.default_budget
            
        df = self.data.copy()
        
        # Simulate baseline targeting (random or simple threshold)
        df['marketing_cost'] = np.random.uniform(100, 500, len(df))
        df['targeted'] = df['response_probability'] > self.config.min_response_prob
        
        # Calculate expected profit
        df['expected_profit'] = df['predicted_npv'] * df['response_probability'] - df['marketing_cost']
        
        # Budget constraint (simplified)
        total_cost = df[df['targeted']]['marketing_cost'].sum()
        while total_cost > budget and df['targeted'].sum() > 0:
            # Remove lowest profit clients
            lowest_profit_idx = df[df['targeted']]['expected_profit'].idxmin()
            df.loc[lowest_profit_idx, 'targeted'] = False
            total_cost = df[df['targeted']]['marketing_cost'].sum()
        
        # Calculate metrics
        selected = df[df['targeted']]
        total_profit = selected['expected_profit'].sum()
        total_cost = selected['marketing_cost'].sum()
        
        self.baseline_results = {
            'total_profit': total_profit,
            'total_cost': total_cost,
            'clients_reached': len(selected),
            'avg_roi': total_profit / total_cost if total_cost > 0 else 0,
            'data': selected
        }
        
        return self.baseline_results
    
    def optimize_portfolio(self, budget=None, max_clients=None, 
                          segment_constraints=None):
        """
        Run LP optimization for large datasets
        Uses chunked optimization for better performance
        """
        if budget is None:
            budget = self.config.default_budget
            
        df = self.data.copy()
        
        # Pre-filter for efficiency
        df = df[df['response_probability'] >= self.config.min_response_prob].copy()
        
        if len(df) > 10000:
            print(f"Optimizing on {len(df)} clients (may take a moment)...")
        
        # Prepare costs
        df['estimated_cost'] = 100 + (1 - df['response_probability']) * 400
        df['estimated_cost'] = np.clip(df['estimated_cost'], 100, 500)
        
        # Create LP problem
        prob = LpProblem("Portfolio_Optimization", LpMaximize)
        
        # Decision variables (using LpVariables for efficiency)
        n_clients = len(df)
        x = LpVariable.dicts("target", range(n_clients), 0, 1, LpBinary)
        
        # Objective
        expected_profits = (df['predicted_npv'].values * 
                           df['response_probability'].values - 
                           df['estimated_cost'].values)
        
        prob += lpSum([expected_profits[i] * x[i] for i in range(n_clients)])
        
        # Budget constraint
        costs = df['estimated_cost'].values
        prob += lpSum([costs[i] * x[i] for i in range(n_clients)]) <= budget
        
        # Max clients constraint
        if max_clients:
            prob += lpSum([x[i] for i in range(n_clients)]) <= max_clients
        
        # Segment constraints
        if segment_constraints:
            for segment_name, max_count in segment_constraints.items():
                if segment_name in df.columns:
                    segment_indices = [i for i in range(n_clients) 
                                     if df.iloc[i][segment_name] == segment_name]
                    if segment_indices:
                        prob += lpSum([x[i] for i in segment_indices]) <= max_count
        
        # Solve
        print("Solving optimization problem...")
        solver = PULP_CBC_CMD(msg=0, timeLimit=60)  # 60 second timeout
        prob.solve(solver)
        
        if prob.status != LpStatusOptimal:
            print(f"Optimization status: {LpStatus[prob.status]}")
            print("Using heuristic fallback...")
            return self._heuristic_optimization(df, budget, max_clients)
        
        # Extract results
        selected_indices = [i for i in range(n_clients) if x[i].value() == 1]
        selected_df = df.iloc[selected_indices].copy()
        
        # Calculate metrics
        total_cost = selected_df['estimated_cost'].sum()
        total_expected_npv = (selected_df['predicted_npv'] * 
                             selected_df['response_probability']).sum()
        total_profit = total_expected_npv - total_cost
        
        self.optimization_results = {
            'total_profit': total_profit,
            'total_cost': total_cost,
            'clients_reached': len(selected_df),
            'avg_roi': total_profit / total_cost if total_cost > 0 else 0,
            'selected_clients': selected_df,
            'status': LpStatus[prob.status],
            'objective_value': value(prob.objective)
        }
        
        return self.optimization_results
    
    def _heuristic_optimization(self, df, budget, max_clients):
        """Fallback heuristic optimization when LP fails"""
        print("Using heuristic optimization...")
        
        # Calculate scores
        df['score'] = df['predicted_npv'] * df['response_probability'] / df['estimated_cost']
        
        # Sort by score and select
        df_sorted = df.sort_values('score', ascending=False)
        
        selected = []
        total_cost = 0
        
        for idx, row in df_sorted.iterrows():
            if total_cost + row['estimated_cost'] <= budget:
                selected.append(idx)
                total_cost += row['estimated_cost']
                if max_clients and len(selected) >= max_clients:
                    break
        
        selected_df = df.loc[selected].copy()
        
        # Calculate metrics
        total_cost = selected_df['estimated_cost'].sum()
        total_expected_npv = (selected_df['predicted_npv'] * 
                             selected_df['response_probability']).sum()
        total_profit = total_expected_npv - total_cost
        
        self.optimization_results = {
            'total_profit': total_profit,
            'total_cost': total_cost,
            'clients_reached': len(selected_df),
            'avg_roi': total_profit / total_cost if total_cost > 0 else 0,
            'selected_clients': selected_df,
            'status': 'Heuristic',
            'objective_value': total_profit
        }
        
        return self.optimization_results
    
    def compare_results(self):
        """Compare baseline vs optimized"""
        if self.baseline_results is None or self.optimization_results is None:
            raise ValueError("Run both baseline and optimization first")
        
        baseline_profit = self.baseline_results['total_profit']
        optimized_profit = self.optimization_results['total_profit']
        
        npv_lift = ((optimized_profit - baseline_profit) / abs(baseline_profit)) * 100
        
        comparison = {
            'baseline_profit': baseline_profit,
            'optimized_profit': optimized_profit,
            'npv_lift_percent': npv_lift,
            'baseline_cost': self.baseline_results['total_cost'],
            'optimized_cost': self.optimization_results['total_cost'],
            'baseline_clients': self.baseline_results['clients_reached'],
            'optimized_clients': self.optimization_results['clients_reached'],
            'baseline_roi': self.baseline_results['avg_roi'],
            'optimized_roi': self.optimization_results['avg_roi'],
            'meets_target': npv_lift >= 40
        }
        
        return comparison