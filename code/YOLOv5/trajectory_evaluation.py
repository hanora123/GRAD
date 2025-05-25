import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

def calculate_ade(predicted_trajectory, actual_trajectory):
    """Calculate Average Displacement Error"""
    if len(predicted_trajectory) != len(actual_trajectory):
        raise ValueError("Trajectories must have the same length")
    
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
    
    pred_final = predicted_trajectory[-1]
    actual_final = actual_trajectory[-1]
    
    return np.sqrt((pred_final[0] - actual_final[0])**2 + (pred_final[1] - actual_final[1])**2)

def evaluate_predictions(predictions_csv):
    # Load predictions
    df = pd.read_csv(predictions_csv)
    
    # Group by frame and object ID
    results = {}
    for (frame, obj_id), group in df.groupby(['frame_number', 'object_id']):
        # Extract predicted trajectories
        for _, row in group.iterrows():
            if row['data_type'] == 'prediction':
                predicted_trajectory = json.loads(row['trajectory'])
                
                # Extract actual trajectory from columns
                actual_trajectory = []
                for i in range(len(predicted_trajectory)):
                    if pd.notna(row[f'actual_future_x{i+1}']) and pd.notna(row[f'actual_future_y{i+1}']):
                        actual_trajectory.append((row[f'actual_future_x{i+1}'], row[f'actual_future_y{i+1}']))
                    else:
                        break
                
                # Only evaluate if we have actual data
                if len(actual_trajectory) > 0:
                    ade = calculate_ade(predicted_trajectory[:len(actual_trajectory)], actual_trajectory)
                    fde = calculate_fde(predicted_trajectory[:len(actual_trajectory)], actual_trajectory)
                    
                    if frame not in results:
                        results[frame] = {}
                    if obj_id not in results[frame]:
                        results[frame][obj_id] = []
                    
                    results[frame][obj_id].append({
                        'probability': row['probability'],
                        'ade': ade,
                        'fde': fde
                    })
    
    return results

def main():
    results = evaluate_predictions('path/to/predictions.csv')
    
    # Calculate overall metrics
    all_ade = []
    all_fde = []
    
    for frame in results:
        for obj_id in results[frame]:
            for pred in results[frame][obj_id]:
                all_ade.append(pred['ade'])
                all_fde.append(pred['fde'])
    
    print(f"Average ADE: {np.mean(all_ade):.2f} pixels")
    print(f"Average FDE: {np.mean(all_fde):.2f} pixels")
    
    # Plot error distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_ade, bins=20)
    plt.title('Average Displacement Error Distribution')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_fde, bins=20)
    plt.title('Final Displacement Error Distribution')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('trajectory_error_distribution.png')
    plt.show()

if __name__ == "__main__":
    main()