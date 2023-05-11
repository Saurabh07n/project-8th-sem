import numpy as np
import pandas as pd

np.random.seed(42)  # Set the random seed for reproducibility

# Define the modulation classes and their corresponding signals
modulations = {
    'AM-DSB': np.sin,
    'FM': np.cos,
    'PSK': lambda x: np.sign(np.sin(x)),
    'AM-SSB': lambda x: np.sin(2 * np.pi * 3 * x) + np.sin(2 * np.pi * 7 * x),
    'QAM': lambda x: np.sin(2 * np.pi * 5 * x) * np.cos(2 * np.pi * 10 * x),
}

# Generate the dataset
samples_per_modulation = 60
snrs = np.random.choice(range(-20, 21), size=samples_per_modulation * len(modulations), replace=True)
samples = np.random.uniform(size=samples_per_modulation * len(modulations))
modulation_list = []
data = []
for modulation, signal_func in modulations.items():
    signals = [signal_func(2 * np.pi * snr * samples[i]) for i, snr in enumerate(snrs[:samples_per_modulation])]
    modulation_list.extend([modulation] * samples_per_modulation)
    data.extend(zip(samples[:samples_per_modulation], snrs[:samples_per_modulation], signals))
    snrs = snrs[samples_per_modulation:]
    samples = samples[samples_per_modulation:]

# Create the DataFrame
df = pd.DataFrame(data, columns=['sample', 'snr', 'signal'])
df['modulation'] = modulation_list

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Save the DataFrame to a CSV file
df.to_csv('dataset/dataset.csv', index=False)
