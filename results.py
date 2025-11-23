"""
Professional visualization of model training results.
Generates publication-ready plots from saved model outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_results(results_dir='models'):
    """Load all result files."""
    predictions = pd.read_csv(f'{results_dir}/test_predictions.csv')
    feature_imp = pd.read_csv(f'{results_dir}/feature_importance.csv')
    
    with open(f'{results_dir}/training_results.json', 'r') as f:
        training_results = json.load(f)
    
    return predictions, feature_imp, training_results

def plot_predictions(predictions, output_dir='results'):
    """Create actual vs predicted plot with residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(predictions['actual'], predictions['predicted'], 
               alpha=0.5, s=20, color='#2E86AB')
    
    # Perfect prediction line
    min_val = min(predictions['actual'].min(), predictions['predicted'].min())
    max_val = max(predictions['actual'].max(), predictions['predicted'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Time to Failure')
    ax1.set_ylabel('Predicted Time to Failure')
    ax1.set_title('Actual vs Predicted Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = axes[1]
    ax2.scatter(predictions['predicted'], predictions['error'], 
               alpha=0.5, s=20, color='#A23B72')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Time to Failure')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/prediction_analysis.png")
    plt.close()

def plot_feature_importance(feature_imp, output_dir='results', top_n=15):
    """Create feature importance bar chart."""
    top_features = feature_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(top_features))
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/feature_importance.png")
    plt.close()

def plot_error_distribution(predictions, output_dir='results'):
    """Plot distribution of prediction errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(predictions['error'], bins=50, color='#F18F01', 
            alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Errors')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    box = ax2.boxplot(predictions['error'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#C73E1D', alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Error Distribution Summary')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/error_distribution.png")
    plt.close()

def create_summary_report(predictions, training_results, output_dir='results'):
    """Create text summary report."""
    rmse = (predictions['error']**2).mean()**0.5
    mae = predictions['error'].abs().mean()
    
    report = f"""
MODEL PERFORMANCE SUMMARY
{'='*60}

Test Set Metrics:
  RMSE:  {rmse:.4f}
  MAE:   {mae:.4f}

Model Configuration:
  Features: {training_results['n_features']}
  Best Iteration: {training_results['best_iteration']}
  Optimization Trials: {training_results.get('optimization_trials', 'N/A')}

Error Statistics:
  Mean Error: {predictions['error'].mean():.4f}
  Std Error:  {predictions['error'].std():.4f}
  Min Error:  {predictions['error'].min():.4f}
  Max Error:  {predictions['error'].max():.4f}

{'='*60}
"""
    
    with open(f'{output_dir}/summary_report.txt', 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {output_dir}/summary_report.txt")
    print(report)

def main():
    """Generate all visualizations."""
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING RESULTS VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load results
    print("Loading results...")
    predictions, feature_imp, training_results = load_results()
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_predictions(predictions, output_dir)
    plot_feature_importance(feature_imp, output_dir)
    plot_error_distribution(predictions, output_dir)
    
    # Create summary
    print("\nGenerating summary report...")
    create_summary_report(predictions, training_results, output_dir)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to '{output_dir}/' directory")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()