# report_generator.py
import pandas as pd
from io import BytesIO

class OptimizationReportGenerator:
    def __init__(self, results, comparison):
        self.results = results
        self.comparison = comparison

    def export_excel_bytes(self):
        """Generate multi-sheet Excel workbook in BytesIO stream"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': [
                    'Baseline Profit ($)', 
                    'Optimized Profit ($)', 
                    'NPV Lift (%)', 
                    'Baseline Cost ($)', 
                    'Optimized Cost ($)', 
                    'Clients Reached', 
                    'Baseline ROI', 
                    'Optimized ROI', 
                    'Target Achieved (40%+ Lift)'
                ],
                'Value': [
                    f"${self.comparison['baseline_profit']:,.2f}",
                    f"${self.comparison['optimized_profit']:,.2f}",
                    f"{self.comparison['npv_lift_percent']:.1f}%",
                    f"${self.comparison['baseline_cost']:,.2f}",
                    f"${self.comparison['optimized_cost']:,.2f}",
                    f"{self.comparison['optimized_clients']:,}",
                    f"{self.comparison['baseline_roi']:.2f}x",
                    f"{self.comparison['optimized_roi']:.2f}x",
                    "Yes" if self.comparison['meets_target'] else "No"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Selected Targeted Clients
            selected_df = self.results['selected_clients']
            if selected_df is not None and not selected_df.empty:
                selected_df.to_excel(writer, sheet_name='Targeted Clients', index=False)
                
                # Sheet 3: Top Ranked Candidates by Profit
                ranked_df = selected_df.sort_values('expected_profit', ascending=False)
                ranked_df.to_excel(writer, sheet_name='Ranked Candidates', index=False)
                
                # Sheet 4: Segment Breakdown
                cat_cols = selected_df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    segment_summary = selected_df.groupby(cat_cols[0]).agg({
                        'predicted_npv': ['mean', 'sum'],
                        'response_probability': 'mean',
                        'expected_profit': 'sum'
                    }).round(2)
                    segment_summary.to_excel(writer, sheet_name='Segment Performance')

        output.seek(0)
        return output.getvalue()
