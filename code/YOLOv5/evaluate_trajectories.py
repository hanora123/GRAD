import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def calculate_ade(predicted_trajectory, actual_trajectory):
    """Calculate Average Displacement Error"""
    if len(predicted_trajectory) != len(actual_trajectory):
        # Use only the common length
        min_len = min(len(predicted_trajectory), len(actual_trajectory))
        predicted_trajectory = predicted_trajectory[:min_len]
        actual_trajectory = actual_trajectory[:min_len]
    
    if len(predicted_trajectory) == 0:
        return float('inf')
    
    total_error = 0
    for i in range(len(predicted_trajectory)):
        pred_x, pred_y = predicted_trajectory[i]
        actual_x, actual_y = actual_trajectory[i]
        error = np.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)
        total_error += error
    
    return total_error / len(predicted_trajectory)

def calculate_fde(predicted_trajectory, actual_trajectory):
    """Calculate Final Displacement Error"""
    if len(predicted_trajectory) == 0 or len(actual_trajectory) == 0:
        return float('inf')
    
    # Use the last point of the shorter trajectory
    min_len = min(len(predicted_trajectory), len(actual_trajectory))
    pred_final = predicted_trajectory[min_len-1]
    actual_final = actual_trajectory[min_len-1]
    
    return np.sqrt((pred_final[0] - actual_final[0])**2 + (pred_final[1] - actual_final[1])**2)

def evaluate_predictions(predictions_csv):
    # Load predictions
    df = pd.read_csv(predictions_csv)
    
    # Filter only predictions with ground truth
    df_with_gt = df[df['ground_truth_available'] == True]
    
    if len(df_with_gt) == 0:
        print("No predictions with ground truth found in the CSV file.")
        return None
    
    # Calculate metrics for each prediction
    results = []
    for _, row in df_with_gt.iterrows():
        predicted_trajectory = json.loads(row['trajectory'])
        actual_trajectory = json.loads(row['ground_truth_trajectory'])
        
        # Skip if either trajectory is empty
        if len(predicted_trajectory) == 0 or len(actual_trajectory) == 0:
            print(f"Skipping empty trajectory for class {row['class']}, object ID {row['object_id']} in frame {row['frame']}")
            continue
        
        ade = calculate_ade(predicted_trajectory, actual_trajectory)
        fde = calculate_fde(predicted_trajectory, actual_trajectory)
        
        results.append({
            'frame': row['frame'],
            'object_id': row['object_id'],
            'class': row['class'],
            'probability': row['probability'],
            'ade': ade,
            'fde': fde
        })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for evaluation results')
    args = parser.parse_args()
    
    # Evaluate predictions
    results_df = evaluate_predictions(args.csv)
    
    if results_df is None:
        return
    
    # Calculate overall metrics
    overall_ade = results_df['ade'].mean()
    overall_fde = results_df['fde'].mean()
    
    print(f"Overall metrics:")
    print(f"  Average Displacement Error (ADE): {overall_ade:.2f} pixels")
    print(f"  Final Displacement Error (FDE): {overall_fde:.2f} pixels")
    
    # Calculate metrics by class
    class_metrics = results_df.groupby('class').agg({'ade': 'mean', 'fde': 'mean'})
    print("\nMetrics by class:")
    print(class_metrics)
    
    # Create output directory if specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save results to CSV
        results_df.to_csv(output_dir / 'trajectory_evaluation_results.csv', index=False)
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        
        # ADE distribution
        plt.subplot(2, 2, 1)
        plt.hist(results_df['ade'], bins=20)
        plt.title('Average Displacement Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        
        # FDE distribution
        plt.subplot(2, 2, 2)
        plt.hist(results_df['fde'], bins=20)
        plt.title('Final Displacement Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        
        # ADE by class
        plt.subplot(2, 2, 3)
        class_metrics['ade'].plot(kind='bar')
        plt.title('Average Displacement Error by Class')
        plt.ylabel('Error (pixels)')
        
        # FDE by class
        plt.subplot(2, 2, 4)
        class_metrics['fde'].plot(kind='bar')
        plt.title('Final Displacement Error by Class')
        plt.ylabel('Error (pixels)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_evaluation_plots.png')
        print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()