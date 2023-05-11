import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from dataset.dataset import Dataset
from models.model1 import Classifier1
# Define the paths to the dataset and model
dataset_path = 'dataset/dataset.csv'

# Create an instance of the Dataset class
dataset = Dataset(dataset_path)

# Load the dataset
data = dataset.load_data()

# Split the data into features and labels
X = data[['sample', 'snr']]
y = data['modulation']

print(X, y)

# Create an instance of the Classifier class
classifier = Classifier1()

# Train the classifier
classifier.train(X, y)

# Create a sample test data
test_data = pd.DataFrame({'sample': [0.58, 0.76, 0.18, 0.62], 'snr': [-20, -5, 9, -14]})

# Make predictions
predictions = classifier.predict(test_data)

# Print the predictions
print(f'Predictions: {predictions}')

# Get unique classes of modulation
modulation_classes = data['modulation'].unique()
print(modulation_classes)

# Plot for each class of modulation
for modulation_class in modulation_classes:
    # Get indices of data points with the current modulation class
    indices = data[data['modulation'] == modulation_class].index

    # Plot time series for the current modulation class
    plt.figure()
    for index in indices:
        plt.plot(data.loc[index, 'sample'], data.loc[index, 'snr'], marker='o', label=f'Sample {index}')
    plt.xlabel('Sample')
    plt.ylabel('SNR')
    plt.title(f'Time Series for Modulation Class: {modulation_class}')
    plt.legend()
    plt.show()
