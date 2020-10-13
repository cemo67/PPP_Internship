import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale, NoPertubation
from PPP_class import PPP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pickle
import sys
import os

# Read Heart data
data = heart()
X_train, X_test, y_train, y_test = data.get_train_test()

# Config
num_model = 0
MODEL_PATH = 'models/' + str(num_model) + '/'

classifier = pickle.load(open(MODEL_PATH + 'classifier.pickle', "rb" ))
predictor = pickle.load(open(MODEL_PATH + 'predictor.pickle', "rb" ))

ppp = PPP(classifier, predictor)

hyperparameter = classifier.get_params()['n_neighbors']

# Returns: mu, sigma for X_test
print(ppp.predict_ppp(X_test, hyperparameter))