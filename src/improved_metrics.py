import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns  # For heatmap visualization
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(predictions, true_values, stock_name=None, plot=False):
    """
    Comprehensive regression evaluation for stock predictions.
    
    Args:
        predictions: Model's predictions
        true_values: Ground truth values
        stock_name: Name of stock (for plotting)
        plot: Whether to generate diagnostic plots
        
    Returns:
        Dictionary of metrics
    """
    # Ensure we have numpy arrays
    preds = np.array(predictions)
    labels = np.array(true_values)
    
    # Calculate regression metrics
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    
    # Calculate R² only if there's variance in the data
    if np.var(labels) > 0:
        r2 = r2_score(labels, preds)
    else:
        r2 = 0.0  # Default value when no variance
    
    # Calculate additional statistics
    mean_error = np.mean(preds - labels)
    median_error = np.median(preds - labels)
    abs_errors = np.abs(preds - labels)
    max_error = np.max(abs_errors)
    
    # Calculate normalized RMSE (as percentage of the range)
    range_labels = np.max(labels) - np.min(labels)
    if range_labels > 0:
        nrmse = (rmse / range_labels) * 100  # as percentage
    else:
        nrmse = 0.0
    
    # Initialize metrics dictionary
    metrics = {
        'samples': len(preds),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_error': mean_error,
        'median_error': median_error,
        'max_error': max_error,
        'nrmse_percent': nrmse
    }
    
    # Create diagnostic plot if requested
    if plot and stock_name:
        plot_regression_diagnostics(preds, labels, stock_name)
    
    return metrics

def plot_regression_diagnostics(predictions, true_values, stock_name):
    """
    Generate comprehensive diagnostic plots for regression analysis.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Predictions vs True Values
    plt.subplot(2, 2, 1)
    plt.scatter(true_values, predictions, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add regression line
    if len(true_values) > 1:
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(true_values, predictions)
        plt.plot(true_values, intercept + slope*np.array(true_values), 'b-', 
                label=f'Regression Line (r={r_value:.2f})')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual Values for {stock_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Errors
    plt.subplot(2, 2, 2)
    errors = predictions - true_values
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.axvline(x=np.mean(errors), color='g', linestyle='-', 
                label=f'Mean Error: {np.mean(errors):.3f}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Prediction Errors for {stock_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals vs Predicted Values
    plt.subplot(2, 2, 3)
    plt.scatter(predictions, errors, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values for {stock_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Prediction and True Value Distributions
    plt.subplot(2, 2, 4)
    plt.hist(true_values, bins=20, alpha=0.5, label='True Values')
    plt.hist(predictions, bins=20, alpha=0.5, label='Predictions')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Values for {stock_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    img_path = f'img/regression_diagnostics_{stock_name}.png'
    plt.savefig(img_path)
    plt.close()
    print(f"Regression diagnostic plot saved to {img_path}")
    
    return img_path

def create_metrics_heatmap(metrics_df, metric_columns, title, output_path):
    """
    Create a heatmap visualization for regression metrics.
    """
    # Create a copy with just the columns we need
    plot_df = metrics_df[['Stock'] + metric_columns].copy()
    plot_df.set_index('Stock', inplace=True)
    
    plt.figure(figsize=(10, max(8, len(plot_df) * 0.3)))
    
    try:
        # Try using seaborn for a nicer heatmap if available
        cmap = 'coolwarm'  # Seaborn accepts colormap names directly
        
        # Create the heatmap
        sns.heatmap(plot_df, annot=True, fmt=".3f", cmap=cmap, linewidths=.5)
    except ImportError:
        # Fallback to matplotlib if seaborn is not available
        plt.imshow(plot_df.values, cmap='coolwarm')
        plt.colorbar()
        
        # Add text annotations
        for i in range(len(plot_df)):
            for j in range(len(metric_columns)):
                plt.text(j, i, f"{plot_df.iloc[i, j]:.3f}", 
                         ha="center", va="center", color="black")
        
        plt.xticks(range(len(metric_columns)), metric_columns, rotation=45)
        plt.yticks(range(len(plot_df)), plot_df.index)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to {output_path}")
    
    return output_path

def aggregate_stock_metrics(metrics_list, stocks):
    """
    Aggregate regression metrics across multiple stocks.
    """
    # Create a robust function that can handle different key formats
    def safe_get(d, keys_to_try, default=0):
        """Try multiple possible keys and return first match or default"""
        for key in keys_to_try:
            if key in d:
                return d[key]
        return default
    
    # Create DataFrame with flexible key handling
    df = pd.DataFrame({
        'Stock': stocks,
        'Samples': [safe_get(m, ['Samples', 'samples']) for m in metrics_list],
        'MSE': [safe_get(m, ['MSE', 'mse']) for m in metrics_list],
        'RMSE': [safe_get(m, ['RMSE', 'rmse']) for m in metrics_list],
        'MAE': [safe_get(m, ['MAE', 'mae']) for m in metrics_list],
        'R2': [safe_get(m, ['R2', 'r2']) for m in metrics_list],
    })
    
    # Calculate weighted averages based on sample size
    weights = df['Samples']
    if weights.sum() > 0:
        weighted_metrics = {
            'Weighted_MSE': np.average(df['MSE'], weights=weights),
            'Weighted_RMSE': np.average(df['RMSE'], weights=weights),
            'Weighted_MAE': np.average(df['MAE'], weights=weights),
            'Weighted_R2': np.average(df['R2'], weights=weights)
        }
    else:
        weighted_metrics = {
            'Weighted_MSE': df['MSE'].mean(),
            'Weighted_RMSE': df['RMSE'].mean(),
            'Weighted_MAE': df['MAE'].mean(),
            'Weighted_R2': df['R2'].mean()
        }
    
    return df, weighted_metrics