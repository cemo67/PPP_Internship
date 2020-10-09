from Pertubations.Pertubation import Pertubation
import numpy as np
import random

# Swapp Columns
class Column(Pertubation):

    def __init__(self, X_train, numerical_column, categorical_column):
        super(Column, self).__init__(numerical_column, categorical_column)
        self.X_train_copy = np.copy(X_train)

    def perturbe(self):
        temp_list = self.numerical_column + self.categorical_column
        self.num_col_1 = random.choice(temp_list)
        self.num_col_2 = random.choice(temp_list)

        while self.num_col_1 == self.num_col_2:
            self.num_col_2 = random.choice(temp_list)

        X_train_copy_temp = np.copy(self.X_train_copy[:, self.num_col_1])
        self.X_train_copy[:, self.num_col_1] = self.X_train_copy[:, self.num_col_2]
        self.X_train_copy[:, self.num_col_2] = X_train_copy_temp[:]
        return self.X_train_copy

    def name(self):
        return 'Swap Columns ' + str(self.num_col_1) + ' & ' + str(self.num_col_2)

class Scale(Pertubation):

    def __init__(self, X_train, numerical_column, categorical_column, fraction=0.3, scaler=10):
        super(Scale, self).__init__(numerical_column, categorical_column)
        self.fraction = fraction
        self.X_train_copy = np.copy(X_train)
        self.scaler = scaler

    def perturbe(self):
        self.num_column = random.choice(self.numerical_column)
        for i in range(len(self.X_train_copy)):
            if random.random() < self.fraction:
                self.X_train_copy[i, self.num_column] *= self.scaler
        return self.X_train_copy


    def name(self):
        return 'Scale Column ' + str(self.num_column) + ' with ' + str(self.scaler)

# Swapp Columns
class NoPertubation(Pertubation):

    def __init__(self, X_train):
        self.X_train_copy = (X_train)

    def perturbe(self):
        return self.X_train_copy


    def name(self):
        return 'NO Petrubation!'
