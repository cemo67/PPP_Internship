import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 'weight' column multiplied with 10 and turnd into integer !!

# Data from Kagle 'Disease'
# https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
'''
Age | Objective Feature | age | int (days)
Height | Objective Feature | height | int (cm) |
Weight | Objective Feature | weight | float (kg) |
Gender | Objective Feature | gender | categorical code |
Systolic blood pressure | Examination Feature | ap_hi | int |
Diastolic blood pressure | Examination Feature | ap_lo | int |
Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
Smoking | Subjective Feature | smoke | binary |
Alcohol intake | Subjective Feature | alco | binary |
Physical activity | Subjective Feature | active | binary |
Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
'''

class heart:
    def __init__(self, path='Data/cardio_train.csv', delimiter=';', test_size=0.2):
        self.path = path
        self.delimiter = delimiter
        self.test_size = test_size
        self.df = pd.read_csv(self.path, self.delimiter)
        self.header = self.df.columns
        self.numerical_column = [0, 2, 3, 4, 5]
        self.categorical_column = [1, 6, 7, 8, 9, 10]

    def get_train(self):
        self.df['weight'] = self.df['weight'] * 10
        self.df['weight'] = self.df['weight'].astype(np.int64)

        X_train = self.df.to_numpy()

        # Delete ID-Column
        X_train = np.delete(X_train, 0, 1)

        # Get Target values
        target = X_train.shape[1] - 1
        Y_train = X_train[:, target]

        # Delete Target Values from X
        X_train = np.delete(X_train, target, 1)

        X_train[:, 0].astype(int)
        return X_train, Y_train

    def get_train_test(self):
        self.df['weight'] = self.df['weight'] * 10
        self.df['weight'] = self.df['weight'].astype(np.int64)

        X_train = self.df.to_numpy()

        # Delete ID-Column
        X_train = np.delete(X_train, 0, 1)

        # Get Target values
        target = X_train.shape[1] - 1
        Y_train = X_train[:, target]

        # Delete Target Values from X
        X_train = np.delete(X_train, target, 1)

        X_train[:, 0].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=self.test_size)

        return X_train, X_test, y_train, y_test