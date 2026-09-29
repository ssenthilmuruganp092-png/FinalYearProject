import pandas as pd
import numpy as np

# Generate Normal Data (230V, 50Hz, 480kW, 120kVAR)
normal_data = []
for _ in range(5000):
    normal_data.append([
        230.15 + np.random.normal(0, 1), # Bus Voltage
        50.02 + np.random.normal(0, 0.1), # Grid Frequency
        480.00 + np.random.normal(0, 5), # Active Power
        120.00 + np.random.normal(0, 2), # Reactive Power
        0 # Label: 0 = Normal
    ])

# Generate FDI Attack Data (75% Injection)
attack_data = []
for _ in range(5000):
    attack_data.append([
        230.15 * 1.75, # 402.76V
        50.02 * 1.75,  # 87.54Hz
        480.00 * 1.75, # 840.00kW
        120.00 * 1.75, # 210.00kVAR
        1 # Label: 1 = Attack
    ])

# Combine and Save
df = pd.DataFrame(normal_data + attack_data, columns=['Bus_Voltage', 'Grid_Frequency', 'Active_Power', 'Reactive_Power', 'Label'])
df.to_csv('dataset/smart_grid_telemetry.csv', index=False)
print("Dataset created: dataset/smart_grid_telemetry.csv")