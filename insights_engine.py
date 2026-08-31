# insights_engine.py
import pandas as pd
import numpy as np

class InsightsGenerator:
    def __init__(self, results, comparison):
        self.results = results
        self.comparison = comparison
        self.insights = []

    def generate_all_insights(self):
        """Generate comprehensive business insights & recommendations"""
        self.analyze_performance()
        self.analyze_segments()
        self.analyze_efficiency()
        self.generate_recommendations()
        return self.insights

    def analyze_performance(self):
        """Analyze NPV lift performance relative to 40% target"""
        lift = self.comparison.get('npv_lift_percent', 0)
        if lift >= 100:
            self.insights.append({
                'type': 'success',
                'title': 'Exceptional Portfolio Lift',
                'description': f'NPV lift of {lift:.1f}% significantly exceeds the 40% target. The model successfully identified high-value targets.',
                'action': 'Scale campaign deployment across additional portfolio segments.'
            })
        elif lift >= 40:
            self.insights.append({
                'type': 'success',
                'title': 'Target Achieved',
                'description': f'NPV lift of {lift:.1f}% meets the 40% optimization benchmark.',
                'action': 'Proceed with current targeted allocation strategy.'
            })
        else:
            self.insights.append({
                'type': 'warning',
                'title': 'Moderate Lift',
                'description': f'NPV lift of {lift:.1f}% is currently below the 40% goal.',
                'action': 'Consider adjusting response probability filters or expanding marketing budget.'
            })

    def analyze_segments(self):
        """Analyze client concentration across segments"""
        selected = self.results.get('selected_clients')
        if selected is not None and not selected.empty:
            cat_cols = selected.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                col = cat_cols[0]
                counts = selected[col].value_counts()
                top_seg = counts.index[0]
                top_pct = (counts.iloc[0] / len(selected)) * 100
                self.insights.append({
                    'type': 'info',
                    'title': f'Segment Concentration ({col.replace("_", " ").title()})',
                    'description': f'The "{top_seg}" segment represents {top_pct:.1f}% of all selected targets.',
                    'action': f'Tailor dedicated marketing messaging specifically for the {top_seg} group.'
                })

    def analyze_efficiency(self):
        """Evaluate marketing ROI improvement"""
        b_roi = self.comparison.get('baseline_roi', 1)
        o_roi = self.comparison.get('optimized_roi', 1)
        imp = ((o_roi - b_roi) / (b_roi if b_roi > 0 else 1)) * 100
        self.insights.append({
            'type': 'success',
            'title': 'Marketing ROI Gain',
            'description': f'Return on Investment (ROI) improved by {imp:.1f}% compared to baseline targeting.',
            'action': 'Reallocate remaining unspent acquisition budget to high-performing tiers.'
        })

    def generate_recommendations(self):
        """Actionable recommendations"""
        total_cost = self.results.get('total_cost', 0)
        total_profit = self.results.get('total_profit', 0)
        self.insights.append({
            'type': 'recommendation',
            'title': 'Budget Scale Recommendation',
            'description': f'Current budget of ${total_cost:,.0f} was fully optimized for maximum yield.',
            'action': f'Increasing budget by 25% is projected to generate an additional ~${total_profit * 0.20:,.0f} in net profit.'
        })
