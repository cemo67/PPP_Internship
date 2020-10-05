import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column

# Read Data Heart from Sigmoid Paper
data = heart()
X_train, Y_train = data.get_train()
numerical_column = data.numerical_column
categorical_column = data.categorical_column

print(X_train)
print()

pert = Column(X_train, numerical_column, categorical_column)
X__ = pert.perturbe()

print(X__)
print()
print(pert.name())