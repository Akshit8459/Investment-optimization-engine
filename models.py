# models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import joblib
import os

class LargeScaleModels:
    def __init__(self, data, config):
        """
        Initialize models with large-scale data handling
        """
        self.data = data
        self.config = config
        self.npv_model = None
        self.response_model = None
        self.preprocessor = None
        
        # Auto-detect features
        self._detect_features()
        
    def _detect_features(self):
        """Auto-detect features based on data types"""
        # Numeric features
        self.numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns
        self.numeric_features = [f for f in self.numeric_features if 'id' not in f.lower()]
        
        # Categorical features
        self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # If too many features, use a subset
        if len(self.numeric_features) > 20:
            # Use top features based on correlation with target
            # For now, just take first 15
            self.numeric_features = self.numeric_features[:15]
        
        print(f"Detected {len(self.numeric_features)} numeric features")
        print(f"Detected {len(self.categorical_features)} categorical features")
    
    def create_target_variables(self):
        """
        Create NPV and response targets based on available data
        """
        df = self.data.copy()
        
        # Try to find existing NPV-like variable
        possible_npv_columns = ['npv', 'lifetime_value', 'profit', 'aum', 'balance']
        npv_col = None
        for col in possible_npv_columns:
            if col in df.columns:
                npv_col = col
                break
        
        if npv_col:
            df['target_npv'] = df[npv_col]
        else:
            # Create synthetic NPV from available financial features
            income_col = next((col for col in df.columns if 'income' in col.lower()), None)
            credit_col = next((col for col in df.columns if 'credit' in col.lower()), None)
            
            if income_col:
                df['target_npv'] = df[income_col] * 0.5 + np.random.normal(0, 1000, len(df))
                if credit_col:
                    df['target_npv'] += (df[credit_col] - 600) * 50
            else:
                # Fallback: random NPV
                df['target_npv'] = np.random.lognormal(8, 2, len(df))
        
        # Create response target
        if 'response' in df.columns:
            df['target_response'] = df['response']
        else:
            # Generate response based on features
            prob = 1 / (1 + np.exp(-(
                df['target_npv'] / df['target_npv'].mean() * 0.3 +
                np.random.normal(0, 0.5, len(df))
            )))
            df['target_response'] = np.random.binomial(1, prob)
        
        return df
    
    def prepare_data(self):
        """Prepare data for modeling with memory optimization"""
        df = self.create_target_variables()
        
        # Select features for modeling
        X = df[self.numeric_features + self.categorical_features]
        y_npv = df['target_npv']
        y_response = df['target_response']
        
        # Store for later use
        self.data = df
        
        # Split data with memory efficiency
        indices = np.random.permutation(len(df))
        split_idx = int(len(indices) * (1 - self.config.test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_npv_train = y_npv.iloc[train_indices]
        y_npv_test = y_npv.iloc[test_indices]
        y_response_train = y_response.iloc[train_indices]
        y_response_test = y_response.iloc[test_indices]
        
        return X_train, X_test, y_npv_train, y_npv_test, y_response_train, y_response_test
    
    def create_preprocessor(self):
        """Create preprocessing pipeline with handling for large datasets"""
        # For large datasets, use simpler preprocessing
        numeric_transformer = StandardScaler()
        
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False  # Use dense array for compatibility
        )
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor
    
    def train_models(self, use_random_forest=True):
        """Train models with appropriate scaling for large data"""
        print("Preparing data...")
        X_train, X_test, y_npv_train, y_npv_test, y_response_train, y_response_test = self.prepare_data()
        
        print(f"Training set size: {len(X_train):,}")
        print(f"Test set size: {len(X_test):,}")
        
        # Create preprocessor
        self.create_preprocessor()
        
        # Train NPV model
        print("Training NPV prediction model (this may take a while)...")
        if use_random_forest:
            # For very large datasets, reduce n_estimators
            n_estimators = min(100, 50 if len(X_train) > 100000 else 100)
            self.npv_model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(
                    n_estimators=n_estimators,
                    n_jobs=-1,  # Use all cores
                    random_state=self.config.random_state,
                    max_depth=15  # Limit depth for large data
                ))
            ])
        else:
            self.npv_model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression(n_jobs=-1))
            ])
        
        self.npv_model.fit(X_train, y_npv_train)
        
        # Train response model
        print("Training response prediction model...")
        n_estimators = min(100, 50 if len(X_train) > 100000 else 100)
        self.response_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=self.config.random_state,
                max_depth=15
            ))
        ])
        self.response_model.fit(X_train, y_response_train)
        
        # Evaluate models
        print("Evaluating models...")
        y_npv_pred = self.npv_model.predict(X_test)
        y_response_pred = self.response_model.predict_proba(X_test)[:, 1]
        
        npv_rmse = np.sqrt(mean_squared_error(y_npv_test, y_npv_pred))
        npv_r2 = r2_score(y_npv_test, y_npv_pred)
        response_auc = roc_auc_score(y_response_test, y_response_pred)
        
        print(f"NPV Model - RMSE: ${npv_rmse:,.2f}, R²: {npv_r2:.4f}")
        print(f"Response Model - AUC: {response_auc:.4f}")
        
        # Store predictions
        X = self.data[self.numeric_features + self.categorical_features]
        self.data['predicted_npv'] = self.npv_model.predict(X)
        self.data['response_probability'] = self.response_model.predict_proba(X)[:, 1]
        
        return {
            'npv_rmse': npv_rmse,
            'npv_r2': npv_r2,
            'response_auc': response_auc
        }
    
    def predict_npv(self, X):
        """Predict NPV"""
        return self.npv_model.predict(X)
    
    def predict_response(self, X):
        """Predict response probability"""
        return self.response_model.predict_proba(X)[:, 1]
    
    def save_models(self):
        """Save models using joblib (more efficient for large models)"""
        os.makedirs(self.config.models_path, exist_ok=True)
        joblib.dump(self.npv_model, f'{self.config.models_path}/npv_model.joblib')
        joblib.dump(self.response_model, f'{self.config.models_path}/response_model.joblib')
        joblib.dump(self.preprocessor, f'{self.config.models_path}/preprocessor.joblib')
        print(f"Models saved to {self.config.models_path}")
    
    def load_models(self):
        """Load models"""
        self.npv_model = joblib.load(f'{self.config.models_path}/npv_model.joblib')
        self.response_model = joblib.load(f'{self.config.models_path}/response_model.joblib')
        self.preprocessor = joblib.load(f'{self.config.models_path}/preprocessor.joblib')
        print(f"Models loaded from {self.config.models_path}")