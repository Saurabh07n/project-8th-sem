from dataset.dataset import Dataset
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

class Classifier1:
    def __init__(self):
        self.model = SVC()  # Initialize the SVC classifier

    def train(self, X_train, y_train):
        # Train the classifier using the training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions on the test data
        y_pred = self.model.predict(X_test)
        return y_pred
