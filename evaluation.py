import os
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

auto_path = pd.read_csv('val_predictions_keypoints.csv')
manual_path = pd.read_csv('val_keypoints.csv')
auto_measurements_path = 'auto_measurements'
manual_measurements_path = 'manual_measurements'

# Preallocate matrices for storing radial errors and evaluation metrics
num_patients = 80
R = np.zeros((num_patients, 19))  # Radial errors
MRE = np.zeros(19)  # Mean Radial Error
SD = np.zeros(19)  # Standard Deviation
manual_errors = np.zeros((num_patients, 19))  # Manual landmark errors for paired t-test

# Prepare to write results to a file
results_file_path = 'StatisticResult.txt'
with open(results_file_path, 'w') as fid2:
    fid2.write('Landmark ID, MRE(mm), SD(mm), Successful detection rates with accuracy of less than 2.0mm, 2.5mm, 3.0mm, and 4.0mm\n')

    for y, (auto_row, manual_row) in enumerate(zip(auto_path.iterrows(), manual_path.iterrows())):
        print(f'Processing patient {y + 1}')

        # Extract coordinates (x0, y0 to x18, y18)
        auto_coords = np.array([(auto_row[1][f'x{i}'], auto_row[1][f'y{i}']) for i in range(19)])
        manual_coords = np.array([(manual_row[1][f'x{i}'], manual_row[1][f'y{i}']) for i in range(19)])

        # Compute radial errors for each landmark (19 landmarks)
        for x in range(19):
            R[y, x] = (np.sqrt((manual_coords[x, 0] - auto_coords[x, 0])**2 + (manual_coords[x, 1] - auto_coords[x, 1])**2)) * 0.1  # Convert to mm

    # Calculate Mean Radial Error (MRE) and Standard Deviation (SD)
    for x in range(19):
        MRE[x] = np.mean(R[:, x])
        SD[x] = np.sqrt(np.sum((R[:, x] - MRE[x])**2) / num_patients)

    # SDR with various accuracy thresholds (2.0mm, 2.5mm, 3.0mm, and 4.0mm)
    numhit = np.zeros((19, 4))
    for x in range(19):
        for i, accur_mm in enumerate([2.0, 2.5, 3.0, 4.0]):
            numhit[x, i] = np.sum(R[:, x] <= accur_mm)  

        # Write results for each landmark to file
        fid2.write(f'L{x+1}, {MRE[x]:.3f}, {SD[x]:.3f}, '
                   f'{(numhit[x, 0] / num_patients) * 100:.2f}%, '
                   f'{(numhit[x, 1] / num_patients) * 100:.2f}%, '
                   f'{(numhit[x, 2] / num_patients) * 100:.2f}%, '
                   f'{(numhit[x, 3] / num_patients) * 100:.2f}%\n')

    # Write average MRE and SD across all landmarks
    fid2.write(f'AVERAGE, {np.mean(MRE):.3f}, {np.mean(SD):.3f}, '
               f'{(np.mean(numhit[:, 0]) / num_patients) * 100:.2f}%, '
               f'{(np.mean(numhit[:, 1]) / num_patients) * 100:.2f}%, '
               f'{(np.mean(numhit[:, 2]) / num_patients) * 100:.2f}%, '
               f'{(np.mean(numhit[:, 3]) / num_patients) * 100:.2f}%\n')

# Paired T-Test and Bland-Altman Plots
eastman_auto = []
eastman_manual = []
distances_auto = []
distances_manual = []

for patient_id in range(81):
    patient_id_str = str(patient_id).zfill(3)
    
    auto_file = os.path.join(auto_measurements_path, f'{patient_id_str}_auto_eastman.csv')
    manual_file = os.path.join(manual_measurements_path, f'{patient_id_str}_manual_eastman.csv')

    if not os.path.exists(auto_file) or not os.path.exists(manual_file):
        continue

    auto_path = pd.read_csv(auto_file, header=None, skip_blank_lines=False).squeeze()
    manual_path = pd.read_csv(manual_file, header=None, skip_blank_lines=False).squeeze()

    eastman_auto.append(auto_path.iloc[:7].values)
    eastman_manual.append(manual_path.iloc[:7].values)

eastman_auto = np.array(eastman_auto)
eastman_manual = np.array(eastman_manual)

# Paired t-test 
eastman_ttest = stats.ttest_rel(eastman_auto, eastman_manual)

with open(results_file_path, 'a') as fid2:
    fid2.write("\nPaired T-Test Results for Eastman Analysis:\n")
    for i in range(eastman_auto.shape[1]):
        fid2.write(f'Measurement {i+1}: t-statistic = {eastman_ttest.statistic[i]:.3f}, p-value = {eastman_ttest.pvalue[i]:.3f}\n')

# Bland-Altman plots 
def bland_altman_plot(data1, data2, title, save_path):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, color='blue', s=20)
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Diff')
    plt.axhline(mean_diff + 1.96 * std_diff, color='green', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='green', linestyle='--', label='-1.96 SD')
    plt.title(title)
    plt.xlabel('Mean of Auto and Manual')
    plt.ylabel('Difference (Auto - Manual)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

bland_altman_dir = 'bland_altman_plots'
os.makedirs(bland_altman_dir, exist_ok=True)

for i in range(eastman_auto.shape[1]):
    bland_altman_plot(eastman_auto[:, i], eastman_manual[:, i], f'Bland-Altman Plot for Eastman Analysis {i+1}', 
                      os.path.join(bland_altman_dir, f'bland_altman_angle_{i+1}.png'))

print("Results saved successfully.")
