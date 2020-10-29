from abc import ABCMeta, abstractmethod

class Pertubation_class(metaclass=ABCMeta):

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