import pandas as pd
import numpy as np
import os
#general DS variables  
seed = 314 #seed for random generation 
n_samples = 100 #the number of rows
hours = 1000 #machine run time 
np.random.seed(seed)

#independent var baselines 
base_temperature = 50
base_vibration = 5
base_pressure = .1
base_load = 70

# independent variables
runtime = np.linspace(0, hours, n_samples)
load = base_load + np.random.normal(0, 3, n_samples)

# temperature rises with runtime and load
temperature = (
    base_temperature
    + 0.015 * runtime
    + 0.05 * (load - base_load)
    + np.random.normal(0, 1, n_samples)
)

# vibration increases exponentially as machine ages
vibration = (
    base_vibration
    + 0.001 * runtime
    + 0.00002 * runtime**2
    + 0.02 * (temperature - base_temperature)
    + np.random.normal(0, 0.02, n_samples)
)

# pressure reacts to both load and vibration
pressure = (
    base_pressure
    + 0.003 * runtime
    + 0.4 * vibration
    + np.random.normal(0, 0.05, n_samples)
)

#dependent variable
time_to_failure = (
    (hours - runtime)
    - 1.5 * (temperature - base_temperature)
    - 25 * (vibration - base_vibration)
    - 0.5 * (load - base_load)
    + np.random.normal(0, 10, n_samples)
)





df = pd.DataFrame({
    "runtime":runtime,
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'load': load,
    'time_to_failure': time_to_failure
})
df.round(5)


os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_data.csv', index=False)


