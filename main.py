from dataset.dataset import Dataset
from models.model import Classifier
import pandas as pd
import matplotlib.pyplot as plt
# Define the paths to the dataset and model
dataset_path = 'dataset/dataset.csv'

# Create an instance of the Dataset class
dataset = Dataset(dataset_path)

# Load the dataset
data = dataset.load_data()
# Split the data into features and labels
X = data.drop('modulation', axis=1)
y = data['modulation']
print(X,y)

# Create an instance of the Classifier class
classifier = Classifier()

# Train the classifier
classifier.train(X, y)

# Create a sample test data
test_data = pd.DataFrame({'sample': [0.5, 0.2, 0.8, 6], 'snr': [7, 10, 12, 9]})

# Make predictions
# Make predictions
predictions = classifier.predict(test_data)

# Print the predictions
print(predictions)

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