import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
import joblib

class DatasetConfig:
    """Simple data loader and encoder for properly formatted, clean datasets.
        requires: 
            data set is clean (this is not a DS cleaner)
            the training, validation, and test path exist 
            
    """
    
    def __init__(self, train_path, val_path, test_path, target_col):
        """
        Load and prepare pre-split datasets.
        
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV
            test_path: Path to test CSV
            target_col: Name of target column
        """
        self.target_col = target_col
        self.label_encoders = {}
        
        # Load data
        print("Loading data")
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"  Train: {len(self.train_df)} rows")
        print(f"  Val: {len(self.val_df)} rows")
        print(f"  Test: {len(self.test_df)} rows")
        
        # Detect categorical columns (object dtype or few unique values)
        all_cols = [c for c in self.train_df.columns if c != target_col]
        self.categorical_cols = [c for c in all_cols 
                                if self.train_df[c].dtype == 'object' or 
                                self.train_df[c].nunique() < 20]
        self.numerical_cols = [c for c in all_cols if c not in self.categorical_cols]
        
        print(f"\nCategorical columns: {self.categorical_cols}")
        print(f"Numerical columns: {self.numerical_cols}")
    def prepare_data(self):
        """Encode categoricals and return feature matrices."""
        print("\nEncoding categorical vars")
        
        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            
            # Fit on ALL unique values across all splits
            all_values = pd.concat([
                self.train_df[col], 
                self.val_df[col], 
                self.test_df[col]
            ]).unique()
            
            le.fit(all_values)
            
            # Transform all splits
            self.train_df[f"{col}_enc"] = le.transform(self.train_df[col])
            self.val_df[f"{col}_enc"] = le.transform(self.val_df[col])
            self.test_df[f"{col}_enc"] = le.transform(self.test_df[col])
            self.label_encoders[col] = le
        
    # Rest of the code stays the same...
        
        # Transform all splits
        self.train_df[f"{col}_enc"] = le.transform(self.train_df[col])
        self.val_df[f"{col}_enc"] = le.transform(self.val_df[col])
        self.test_df[f"{col}_enc"] = le.transform(self.test_df[col])
        self.label_encoders[col] = le
    
    # Rest of the code stays the same...
        
        # Feature column names
        encoded_cols = [f"{col}_enc" for col in self.categorical_cols]
        self.feature_cols = self.numerical_cols + encoded_cols
        
        print(f"\nFeatures ({len(self.feature_cols)}): {self.feature_cols}")
        
        # Extract X and y
        X_train = self.train_df[self.feature_cols].values
        y_train = self.train_df[self.target_col].values
        X_val = self.val_df[self.feature_cols].values
        y_val = self.val_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].values
        y_test = self.test_df[self.target_col].values
        
        print(f"\nData ready - {X_train.shape[1]} features, {len(X_train)} train samples")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': self.feature_cols
        }
    
    def save_config(self, output_dir='models'):
        """Save encoders and config."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save encoders
        for col, encoder in self.label_encoders.items():
            joblib.dump(encoder, f"{output_dir}/encoder_{col}.pkl")
        
        # Save config
        config = {
            'target_col': self.target_col,
            'feature_cols': self.feature_cols,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols
        }
        
        with open(f"{output_dir}/data_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved to {output_dir}/")

