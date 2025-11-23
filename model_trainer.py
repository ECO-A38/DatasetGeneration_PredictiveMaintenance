import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import the data configuration module
from data_config import DatasetConfig

class LGBMTrainer:
    """
    LightGBM model trainer with Optuna hyperparameter optimization.
    Works with any preprocessed dataset from DatasetConfig.
    """
    
    def __init__(self, data_config, n_trials=100, early_stopping_rounds=50):
        """
        Initialize trainer.
        
        Args:
            data_config: DatasetConfig object with prepared data
            n_trials: Number of Optuna optimization trials
            early_stopping_rounds: Early stopping rounds for training
        """
        self.config = data_config
        self.n_trials = n_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.best_params = None
        self.model = None
        self.study = None
        
    def _create_lgb_datasets(self, data):
        """Create LightGBM datasets from prepared data"""
        train_data = lgb.Dataset(
            data['X_train'], 
            label=data['y_train'],
            feature_name=data['feature_names']
        )
        
        val_data = None
        if data['X_val'] is not None:
            val_data = lgb.Dataset(
                data['X_val'], 
                label=data['y_val'],
                reference=train_data,
                feature_name=data['feature_names']
            )
        
        return train_data, val_data
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, feature_names):
        """Optuna objective function for hyperparameter optimization"""
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
            'max_bin': trial.suggest_int('max_bin', 200, 300),
        }
        
        # Create datasets with current max_bin parameter
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return rmse
    
    def optimize_hyperparameters(self, data):
        """Run Optuna optimization"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        if data['X_val'] is None:
            raise ValueError("Validation data required for hyperparameter optimization")
        
        print(f"Running {self.n_trials} optimization trials...")
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            study_name='lightgbm_optimization'
        )
        
        # Optimize - pass raw data instead of datasets
        self.study.optimize(
            lambda trial: self._objective(
                trial, 
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                data['feature_names']
            ),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        print(f"\nOptimization complete!")
        print(f"Best validation RMSE: {self.study.best_value:.4f}")
        print(f"\nBest parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_params
    
    def train_final_model(self, data, params=None):
        """Train final model with best parameters"""
        print("\n" + "="*60)
        print("TRAINING FINAL MODEL")
        print("="*60)
        
        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided. Run optimize_hyperparameters first.")
            params = self.best_params.copy()
        
        # Add required parameters
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt'
        })
        
        train_data, val_data = self._create_lgb_datasets(data)
        
        # Train with more rounds for final model
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)] if val_data else []
        valid_sets = [val_data] if val_data else None
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        
        print("\n✓ Final model training complete!")
        return self.model
    
    def evaluate(self, X, y, split_name="Data"):
        """Evaluate model on a dataset"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\n{split_name} Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'predictions': y_pred
        }
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.config.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:<30} {row['importance']:>10.2f}")
        print("="*60)
        
        return importance_df
    
    def save_model(self, output_dir='models'):
        """Save trained model and results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save LightGBM model
        self.model.save_model(f"{output_dir}/lightgbm_model.txt")
        
        # Save training results
        results = {
            'best_params': self.best_params,
            'best_iteration': int(self.model.best_iteration),
            'n_features': len(self.config.feature_cols),
            'feature_names': self.config.feature_cols
        }
        
        if self.study is not None:
            results['optimization_trials'] = self.n_trials
            results['best_validation_rmse'] = float(self.study.best_value)
        
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Model saved to {output_dir}/lightgbm_model.txt")
        print(f"✓ Results saved to {output_dir}/training_results.json")
    
    @staticmethod
    def load_model(model_path='models/lightgbm_model.txt'):
        """Load a trained model"""
        return lgb.Booster(model_file=model_path)