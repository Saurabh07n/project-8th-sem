import pandas as pd


class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # Load the dataset from the CSV file
        data = pd.read_csv(self.data_path)

        # Perform any necessary data preprocessing steps

        # Return the preprocessed data
        return data
