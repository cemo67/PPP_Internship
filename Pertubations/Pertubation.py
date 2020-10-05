from abc import ABCMeta, abstractmethod
import numpy as np
import random

class Pertubation(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, numerical_column, categorical_column):
        self.numerical_column = numerical_column
        self.categorical_column = categorical_column


    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def perturbe(self):
        pass