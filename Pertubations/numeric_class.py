from Pertubations.Pertubation_class import Pertubation_class
import numpy as np
import random

# Swapp Columns
class Column(Pertubation_class):

    def __init__(self, X_train, fraction, numerical_column, categorical_column=None):
        super(Column, self).__init__(numerical_column, categorical_column)
        self.X_train_copy = np.copy(X_train)
        self.fraction = fraction

    def perturbe(self):
        random.seed(0)
        if self.categorical_column != None:
            temp_list = self.numerical_column + self.categorical_column
        temp_list = self.numerical_column

        if len(temp_list) == 1:
            return self.X_train_copy

        self.num_col_1 = random.choice(temp_list)
        self.num_col_2 = random.choice(temp_list)

        while self.num_col_1 == self.num_col_2:
            self.num_col_2 = random.choice(temp_list)

        for i in range(len(self.X_train_copy)):
            if random.random() < self.fraction:
                temp_value = self.X_train_copy[i, self.num_col_1]
                self.X_train_copy[i, self.num_col_1] = self.X_train_copy[i, self.num_col_2]
                self.X_train_copy[i, self.num_col_2] = temp_value
        return self.X_train_copy

    def name(self):
        return 'Swap Columns ' + str(self.num_col_1) + ' & ' + str(self.num_col_2) + ' with probability ' + str(self.fraction)

class Scale(Pertubation_class):

    def __init__(self, X_train, fraction, scaler, numerical_column, categorical_column=None):
        super(Scale, self).__init__(numerical_column, categorical_column)
        self.X_train_copy = np.copy(X_train)
        self.scaler = scaler
        self.fraction = fraction

    def perturbe(self):
        random.seed(0)
        self.num_column = random.choice(self.numerical_column)

        for i in range(len(self.X_train_copy)):
            if random.random() < self.fraction:
                self.X_train_copy[i, self.num_column] *= self.scaler
        return self.X_train_copy


    def name(self):
        return 'Scale Column ' + str(self.num_column) + ' with ' + str(self.scaler)

# Swapp Columns
class NoPertubation(Pertubation_class):

    def __init__(self, X_train):
        self.X_train_copy = (X_train)

    def perturbe(self):
        return self.X_train_copy

    def name(self):
        return 'NO Petrubation!'
