import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# General DS variables  
seed = 314
n_machines = 50  # number of different machines to simulate
readings_per_machine = 100  # readings per machine
n_samples = n_machines * readings_per_machine  # total rows
max_hours = 1000  # maximum machine runtime
np.random.seed(seed)

# Independent var baselines
base_temperature = 50
base_vibration = 5
base_pressure = 0.1
base_load = 70

# Categorical variables
machine_ids = ['M001', 'M002', 'M003', 'M004', 'M005']
shifts = ['day', 'night']

# Machine-specific degradation rates (some machines age faster)
machine_degradation = {
    'M001': 1.0,   # baseline
    'M002': 1.2,   # ages 20% faster
    'M003': 0.9,   # ages 10% slower
    'M004': 1.1,   # ages 10% faster
    'M005': 0.95   # ages 5% slower
}

# Shift effects (Shifts often indicate special cause variation)
shift_temp_modifier = {'day': 0, 'night': -2}  # night is cooler
shift_load_modifier = {'day': 0, 'night': 2}   # night shift runs harder

# Initialize lists to store all data
all_machine_ids = []
all_shifts = []
all_runtimes = []
all_temperatures = []
all_vibrations = []
all_pressures = []
all_loads = []
all_maintenance = []
all_time_to_failure = []

# Generate data for each machine
for machine_num in range(n_machines):
    # Randomly assign a machine type to this machine instance
    machine_type = np.random.choice(machine_ids)
    machine_factor_val = machine_degradation[machine_type]
    
    # Generate random runtime readings for this machine (sorted)
    runtimes = np.sort(np.random.uniform(0, max_hours, readings_per_machine))
    
    # Generate shift assignments for each reading
    machine_shifts = np.random.choice(shifts, readings_per_machine)
    
    # Get shift modifiers for this machine's readings
    shift_temp_mods = np.array([shift_temp_modifier[s] for s in machine_shifts])
    shift_load_mods = np.array([shift_load_modifier[s] for s in machine_shifts])
    
    # Calculate load with daily cycle
    load = base_load + np.random.normal(0, 3, readings_per_machine) + shift_load_mods
    load = load + 5 * np.sin(2 * np.pi * runtimes / 24)
    
    # Temperature rises with runtime and load, plus daily variation and shift effects
    temperature = (
        base_temperature
        + 0.015 * runtimes * machine_factor_val
        + 0.05 * (load - base_load)
        + 3 * np.sin(2 * np.pi * runtimes / 24)
        + shift_temp_mods
        + np.random.normal(0, 1, readings_per_machine)
    )
    
    # Vibration increases exponentially with age and temperature
    vibration = (
        base_vibration
        + 0.001 * runtimes * machine_factor_val
        + 0.00003 * runtimes**2
        + 0.02 * (temperature - base_temperature)
        + np.random.normal(0, 0.02, readings_per_machine)
    )
    
    # Pressure reacts to load, vibration, and temperature
    pressure = (
        base_pressure
        + 0.003 * runtimes
        + 0.004 * (load - base_load)
        + 0.3 * (vibration - base_vibration)
        + np.random.normal(0, 0.05, readings_per_machine)
    )
    
    # Maintenance events (10% chance at each time point)
    maintenance = np.random.choice([0, 1], readings_per_machine, p=[0.9, 0.1])
    
    # Time to failure with interaction effects and categorical influences
    time_to_failure = (
        (max_hours - runtimes)
        - 1.5 * (temperature - base_temperature)
        - 25 * (vibration - base_vibration)
        - 0.5 * (load - base_load)
        - 0.05 * (temperature - base_temperature) * (vibration - base_vibration)
        - 50 * (machine_factor_val - 1.0)
        + maintenance * 100
        + np.random.normal(0, 10, readings_per_machine)
    )
    
    # Ensure non-negative time to failure
    time_to_failure = np.maximum(time_to_failure, 0)
    
    # Store this machine's data
    all_machine_ids.extend([machine_type] * readings_per_machine)
    all_shifts.extend(machine_shifts)
    all_runtimes.extend(runtimes)
    all_temperatures.extend(temperature)
    all_vibrations.extend(vibration)
    all_pressures.extend(pressure)
    all_loads.extend(load)
    all_maintenance.extend(maintenance)
    all_time_to_failure.extend(time_to_failure)

# Create DataFrame
df = pd.DataFrame({
    "machine_id": all_machine_ids,
    "shift": all_shifts,
    "runtime": all_runtimes,
    'temperature': all_temperatures,
    'vibration': all_vibrations,
    'pressure': all_pressures,
    'load': all_loads,
    'maintenance': all_maintenance,
    'time_to_failure': all_time_to_failure
})

df = df.round(5)

# Train/val/test split by machine
unique_machines = df['machine_id'].unique()
np.random.shuffle(unique_machines)

n_train = int(0.6 * len(unique_machines))
n_val = int(0.2 * len(unique_machines))

train_machines = unique_machines[:n_train]
val_machines = unique_machines[n_train:n_train+n_val]
test_machines = unique_machines[n_train+n_val:]

train_df = df[df['machine_id'].isin(train_machines)]
val_df = df[df['machine_id'].isin(val_machines)]
test_df = df[df['machine_id'].isin(test_machines)]

# Save files
os.makedirs('data', exist_ok=True)

df.to_csv('data/synthetic_data.csv', index=False)
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Save data dictionary
data_dict = {
    'Feature': ['machine_id', 'shift', 'runtime', 'temperature', 'vibration', 
                'pressure', 'load', 'maintenance', 'time_to_failure'],
    'Type': ['categorical', 'categorical', 'numerical', 'numerical', 'numerical',
             'numerical', 'numerical', 'binary', 'numerical (target)'],
    'Description': [
        'Machine identifier (M001-M005)',
        'Work shift (day/night)',
        'Hours of operation since start',
        'Operating temperature (degrees C)',
        'Vibration level (mm/s)',
        'System pressure (bar)',
        'Operating load (percent)',
        'Maintenance performed (0=no, 1=yes)',
        'Predicted hours until failure'
    ],
    'Range': [
        'M001-M005',
        'day, night',
        f'0-{max_hours}',
        f'{df["temperature"].min():.1f}-{df["temperature"].max():.1f}',
        f'{df["vibration"].min():.1f}-{df["vibration"].max():.1f}',
        f'{df["pressure"].min():.2f}-{df["pressure"].max():.2f}',
        f'{df["load"].min():.1f}-{df["load"].max():.1f}',
        '0, 1',
        f'0-{df["time_to_failure"].max():.1f}'
    ]
}

dict_df = pd.DataFrame(data_dict)
dict_df.to_csv('data/data_dictionary.csv', index=False)

# Save metadata
metadata = {
    'generation_date': datetime.now().isoformat(),
    'seed': seed,
    'n_samples': n_samples,
    'n_machines': n_machines,
    'readings_per_machine': readings_per_machine,
    'max_hours': max_hours,
    'machine_types': machine_ids,
    'machine_degradation_rates': machine_degradation,
    'base_values': {
        'temperature': base_temperature,
        'vibration': base_vibration,
        'pressure': base_pressure,
        'load': base_load
    },
    'train_machines': train_machines.tolist(),
    'val_machines': val_machines.tolist(),
    'test_machines': test_machines.tolist()
}

with open('data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
    
print("data generation complete")