import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale
from PPP import PPP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model

# Read Heart data
data = heart()
X_train, X_test, y_train, y_test = data.get_train_test()

# Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Predictor
kernel = 1.0 * RBF(1.0)
predictor = GaussianProcessRegressor(kernel = kernel)

# PPP
ppp = PPP(classifier, predictor)

print('Classifier')
print('Score = ', classifier.score(X_train, y_train))

print()
print('No Pertubation!')
ppp.fit_ppp(X_train, y_train)
print('Score = ', ppp.score)

print()
print('Column Pertubation to X_train!')
pert = Column(X_train, data.numerical_column, data.categorical_column)
X_train_perturbed = pert.perturbe()
print(pert.name())
ppp.fit_ppp(X_train_perturbed, y_train)
print('Score = ', ppp.score)

print()
print('Scaler Pertubation to X_train!')
pert = Scale(X_train, data.numerical_column, data.categorical_column)
X_train_perturbed = pert.perturbe()
print(pert.name())
ppp.fit_ppp(X_train_perturbed, y_train)
print('Score = ', ppp.score)
