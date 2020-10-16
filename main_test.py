import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale, NoPertubation
from PPP_class import PPP
from Bayesian_Optimization.bayesian_optimization_class import BO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
import pickle
import sys
import os

SAVE = False

# Read Heart data
data = heart()
X_train, X_test, y_train, y_test = data.get_train_test()



# Predictor
kernel = 1.0 * RBF(1.0)
predictor = GaussianProcessRegressor(kernel = kernel)



# Config
MODEL_PATH = 'models/'

DEFAULT_PERTUBATION = [('No_Perturbation', NoPertubation(X_train)),
                       ('Column', Column(X_train, data.numerical_column, data.categorical_column)),
                       ('Scale', Scale(X_train, data.numerical_column, data.categorical_column))]

pertubation = DEFAULT_PERTUBATION[0]

X_train_perturbed = pertubation[1].perturbe()

# Fit ppp
# Classifier
hyperparameter = 2
classifier = KNeighborsClassifier(n_neighbors=hyperparameter)
classifier.fit(X_train, y_train)
# PPP
ppp = PPP(classifier, predictor)

ppp.fit_ppp(X_train_perturbed, y_train, hyperparameter)

# Second ppp
hyperparameter = 5
classifier = KNeighborsClassifier(n_neighbors=hyperparameter)
classifier.fit(X_train, y_train)

ppp.classifier = classifier
ppp.fit_ppp(X_train_perturbed, y_train, hyperparameter)


# Third ppp

hyperparameter = 8
classifier = KNeighborsClassifier(n_neighbors=hyperparameter)
classifier.fit(X_train, y_train)

ppp.classifier = classifier
ppp.fit_ppp(X_train_perturbed, y_train, hyperparameter)


# Fourth ppp
hyperparameter = 6
classifier = KNeighborsClassifier(n_neighbors=hyperparameter)
classifier.fit(X_train, y_train)

ppp.classifier = classifier
ppp.fit_ppp(X_train_perturbed, y_train, hyperparameter)

# BO predict next HP
bo = BO(ppp.predictor, [1, 10], ppp.meta_features)

best_hyperparameter = bo.propose_location()
print('END')
print('best_hyperparameter', best_hyperparameter)

