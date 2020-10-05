import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale

# Read Data Heart from Sigmoid Paper
data = heart()
X_train, y_train = data.get_train()

print(X_train)
print()

pert = Scale(X_train, data.numerical_column, data.categorical_column)
X_perturbed = pert.perturbe()
print(X_perturbed)
print(pert.name())
