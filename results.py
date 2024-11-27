import os
import numpy as np
import pandas as pd
from measurement import calculate_angles_and_distances

# Define file paths for the two CSV files
auto_csv_path = 'val_predictions_keypoints.csv'
manual_csv_path = 'val_keypoints.csv'

# Define output directories for results
auto_measurements_path = 'auto_measurements'
manual_measurements_path = 'manual_measurements'

os.makedirs(auto_measurements_path, exist_ok=True)
os.makedirs(manual_measurements_path, exist_ok=True)

# Load CSV files
auto_data = pd.read_csv(auto_csv_path)
manual_data = pd.read_csv(manual_csv_path)

# Loop over each patient (row) in the CSV
for index, (auto_row, manual_row) in enumerate(zip(auto_data.iterrows(), manual_data.iterrows())):
    patient_id = 1 + index  
    patient_id_str = str(patient_id).zfill(3)
    
    # Extract coordinates (x0, y0 to x18, y18)
    auto_coordinates = np.array([(auto_row[1][f'x{i}'], auto_row[1][f'y{i}']) for i in range(19)])
    manual_coordinates = np.array([(manual_row[1][f'x{i}'], manual_row[1][f'y{i}']) for i in range(19)])

    # Calculate angles and distances for auto and manual coordinates
    auto_eastman = calculate_angles_and_distances(auto_coordinates)
    manual_eastman = calculate_angles_and_distances(manual_coordinates)

    # Save results for auto measurements
    auto_output_csv = os.path.join(auto_measurements_path, f'{patient_id_str}_auto_eastman.csv')
    with open(auto_output_csv, 'w') as f:
        for angle in auto_eastman:
            f.write(f"{angle}\n")
        
    # Save results for manual measurements
    manual_output_csv = os.path.join(manual_measurements_path, f'{patient_id_str}_manual_eastman.csv')
    with open(manual_output_csv, 'w') as f:
        for angle in manual_eastman:
            f.write(f"{angle}\n")
    
    print(f"Results saved for patient {patient_id_str} in {auto_output_csv} and {manual_output_csv}.")
