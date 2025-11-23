"""
Main pipeline for training machine time-to-failure prediction model.
Orchestrates data loading and model training.
"""

from data_config import DatasetConfig
from model_trainer import LGBMTrainer
import os

def main():
    """Run complete training pipeline."""
    
    print("\n" + "="*70)
    print(" MACHINE TIME-TO-FAILURE PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    TRAIN_PATH = 'data/train.csv'
    VAL_PATH = 'data/validation.csv'
    TEST_PATH = 'data/test.csv'
    TARGET_COL = 'time_to_failure'
    OUTPUT_DIR = 'models'
    N_TRIALS = 100  # Optuna optimization trials
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load and prepare data
    # ========================================================================
    print("\n[STEP 1/4] Loading and preparing data...")
    print("-" * 70)
    
    config = DatasetConfig(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=TEST_PATH,
        target_col=TARGET_COL
    )
    
    data = config.prepare_data()
    config.save_config(OUTPUT_DIR)
    
    # ========================================================================
    # STEP 2: Optimize hyperparameters
    # ========================================================================
    print("\n[STEP 2/4] Optimizing hyperparameters...")
    print("-" * 70)
    
    trainer = LGBMTrainer(
        data_config=config,
        n_trials=N_TRIALS,
        early_stopping_rounds=50
    )
    
    best_params = trainer.optimize_hyperparameters(data)
    
    # ========================================================================
    # STEP 3: Train final model
    # ========================================================================
    print("\n[STEP 3/4] Training final model...")
    print("-" * 70)
    
    model = trainer.train_final_model(data)
    
    # ========================================================================
    # STEP 4: Evaluate and save
    # ========================================================================
    print("\n[STEP 4/4] Evaluating model...")
    print("-" * 70)
    
    train_metrics = trainer.evaluate(data['X_train'], data['y_train'], "Train")
    val_metrics = trainer.evaluate(data['X_val'], data['y_val'], "Validation")
    test_metrics = trainer.evaluate(data['X_test'], data['y_test'], "Test")
    
    # Feature importance
    feature_importance = trainer.get_feature_importance()
    feature_importance.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
    
    # Save model and results
    trainer.save_model(OUTPUT_DIR)
    
    # Save test predictions
    import pandas as pd
    test_predictions = pd.DataFrame({
        'actual': data['y_test'],
        'predicted': test_metrics['predictions'],
        'error': data['y_test'] - test_metrics['predictions']
    })
    test_predictions.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False)
    
    # Final summary
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    
    print(f"\nSaved files in '{OUTPUT_DIR}/':")
    print(f"  ✓ lightgbm_model.txt")
    print(f"  ✓ data_config.json")
    print(f"  ✓ training_results.json")
    print(f"  ✓ feature_importance.csv")
    print(f"  ✓ test_predictions.csv")
    print(f"  ✓ encoder_*.pkl (for each categorical column)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()