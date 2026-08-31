# data_loader.py
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LargeDataLoader:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.metadata = {}
        
    def load_hamzi_dataset(self, sample_size=None):
        """
        Load HAMZI.AI Financial Ecosystem Dataset
        Uses Dask for efficient loading of large CSV files
        """
        print("Loading HAMZI.AI dataset...")
        
        file_path = Path(f"{self.config.data_path}/financial_ecosystem.csv")
        if not file_path.exists():
            print(f"File {file_path} not found. Generating synthetic HAMZI.AI financial ecosystem dataset...")
            n_samples = sample_size or 50000
            df = pd.DataFrame({
                'customer_id': range(n_samples),
                'age': np.random.randint(18, 75, n_samples),
                'income': np.random.lognormal(10.5, 0.8, n_samples),
                'credit_score': np.random.normal(680, 70, n_samples).clip(300, 850),
                'debt': np.random.exponential(15000, n_samples),
                'balance': np.random.lognormal(9, 1.2, n_samples),
                'account_age': np.random.randint(1, 20, n_samples),
                'aum': np.random.lognormal(10, 1.5, n_samples),
                'investable_assets': np.random.lognormal(10, 1.2, n_samples),
                'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Retired', 'Unemployed'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
                'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
                'account_type': np.random.choice(['Savings', 'Investment', 'Checking', 'Wealth'], n_samples),
                'risk_tolerance': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'investment_experience': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_samples)
            })
        # The actual dataset is large (3M rows), use Dask for lazy loading if present
        elif self.config.use_dask:
            df = dd.read_csv(str(file_path))
            if sample_size:
                df = df.sample(frac=sample_size/len(df), random_state=self.config.random_state)
            df = df.compute()
        else:
            df = pd.read_csv(str(file_path))
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=self.config.random_state)
        
        print(f"Loaded {len(df):,} records")
        
        # Clean column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Identify key features for investment optimization
        # Based on HAMZI.AI dataset schema
        self.define_hamzi_features(df)
        
        return df
    
    def define_hamzi_features(self, df):
        """Define feature mappings for HAMZI.AI dataset"""
        # Numeric features typically available
        self.numeric_features = [col for col in df.columns if col in [
            'age', 'income', 'credit_score', 'debt', 'balance', 
            'transaction_amount', 'account_age', 'aum', 'investable_assets'
        ]]
        
        # Categorical features
        self.categorical_features = [col for col in df.columns if col in [
            'employment_status', 'marital_status', 'education_level',
            'account_type', 'risk_tolerance', 'investment_experience'
        ]]
        
        # If features don't match exactly, create fallbacks
        if not self.numeric_features:
            # Infer numeric columns
            self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.categorical_features:
            # Infer categorical columns
            self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def load_fraud_graph_dataset(self, sample_size=None):
        """
        Load or generate SantanderAI/gen-fraud-graph dataset
        """
        print("Loading Fraud Graph dataset...")
        
        # The gen-fraud-graph generates synthetic graphs
        # We'll simulate loading node and edge features
        
        # For demonstration, we generate features that match the graph structure
        n_nodes = sample_size or self.config.fraud_graph_sample_size
        
        # Simulate loading node features
        df = pd.DataFrame({
            'node_id': range(n_nodes),
            'account_balance': np.random.lognormal(10, 2, n_nodes),
            'risk_score': np.random.beta(2, 5, n_nodes) * 100,
            'transaction_volume': np.random.exponential(1000, n_nodes),
            'num_connections': np.random.poisson(10, n_nodes),
            'fraud_probability': np.random.beta(1, 10, n_nodes)
        })
        
        # Add categorical features
        df['account_type'] = np.random.choice(
            ['personal', 'business', 'joint', 'trust'],
            n_nodes
        )
        df['risk_category'] = np.random.choice(
            ['low', 'medium', 'high', 'very_high'],
            n_nodes,
            p=[0.3, 0.4, 0.2, 0.1]
        )
        
        print(f"Generated {len(df):,} node features")
        
        # Define features for this dataset
        self.numeric_features = ['account_balance', 'risk_score', 
                                'transaction_volume', 'num_connections']
        self.categorical_features = ['account_type', 'risk_category']
        
        return df
    
    def load_santander_transactions(self):
        """
        Load Santander transaction dataset (complementary)
        """
        print("Loading Santander transaction data...")
        
        # Load the Kaggle dataset
        try:
            df = pd.read_csv(f"{self.config.data_path}/train.csv")
        except:
            # If not available, generate synthetic
            n_samples = 20000
            df = pd.DataFrame({
                'ID': range(n_samples),
                'VAR1': np.random.normal(0, 1, n_samples),
                'VAR2': np.random.normal(0, 1, n_samples),
                'TARGET': np.random.binomial(1, 0.1, n_samples)
            })
            
            # Add more realistic financial features
            df['income'] = np.random.lognormal(10, 0.5, n_samples)
            df['credit_score'] = np.random.normal(680, 80, n_samples)
            df['debt_ratio'] = np.random.uniform(0, 0.6, n_samples)
        
        print(f"Loaded {len(df):,} Santander transaction records")
        
        # Define features
        self.numeric_features = ['income', 'credit_score', 'debt_ratio']
        if 'VAR1' in df.columns:
            self.numeric_features.extend(['VAR1', 'VAR2'])
        self.categorical_features = []
        
        return df
    
    def engineer_features(self, df):
        """Create advanced financial features & interactions"""
        df = df.copy()
        if 'debt' in df.columns and 'income' in df.columns:
            df['debt_to_income'] = (df['debt'] / (df['income'] + 1)).clip(0, 10)
            if 'debt_to_income' not in self.numeric_features:
                self.numeric_features.append('debt_to_income')
                
        if 'income' in df.columns and 'age' in df.columns:
            df['age_income_interaction'] = df['age'] * df['income']
            if 'age_income_interaction' not in self.numeric_features:
                self.numeric_features.append('age_income_interaction')
                
        if 'income' in df.columns:
            df['income_bracket'] = pd.qcut(df['income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            if 'income_bracket' not in self.categorical_features:
                self.categorical_features.append('income_bracket')
                
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Mid', 'Senior', 'Retired'], right=False)
            if 'age_group' not in self.categorical_features:
                self.categorical_features.append('age_group')
                
        return df

    def handle_missing_data(self, df):
        """Advanced missing data handling & imputation"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('Unknown')
        return df

    def cap_outliers(self, df, method='clip'):
        """Cap extreme outliers in numeric features"""
        df = df.copy()
        for col in self.numeric_features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if method == 'clip':
                    q_low = df[col].quantile(0.005)
                    q_high = df[col].quantile(0.995)
                    df[col] = df[col].clip(q_low, q_high)
        return df
    
    def load_data(self):
        """Main loader based on configuration with feature engineering"""
        if self.config.dataset_type == "hamzi":
            self.data = self.load_hamzi_dataset(self.config.hamzi_sample_size)
        elif self.config.dataset_type == "fraud_graph":
            self.data = self.load_fraud_graph_dataset()
        elif self.config.dataset_type == "santander":
            self.data = self.load_santander_transactions()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_type}")
        
        # Apply data quality & feature engineering pipeline
        self.data = self.handle_missing_data(self.data)
        self.data = self.engineer_features(self.data)
        self.data = self.cap_outliers(self.data, method='clip')
        
        # Store dataset info
        self.metadata = {
            'n_records': len(self.data),
            'n_features': len(self.data.columns),
            'numeric_features': len(self.numeric_features),
            'categorical_features': len(self.categorical_features)
        }
        
        return self.data

if __name__ == "__main__":
    from config import config
    
    loader = LargeDataLoader(config)
    df = loader.load_data()
    print(f"Loaded dataset with {len(df)} records")
    print(f"Columns: {df.columns.tolist()[:10]}...")