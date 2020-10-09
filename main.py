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

# Classifier
hyperparameter = 5
classifier = KNeighborsClassifier(n_neighbors=hyperparameter)
classifier.fit(X_train, y_train)

# Predictor
kernel = 1.0 * RBF(1.0)
predictor = GaussianProcessRegressor(kernel = kernel)

# PPP
ppp = PPP(classifier, predictor)

# Config
MODEL_PATH = 'models/'

DEFAULT_PERTUBATION = [('No_Perturbation', NoPertubation(X_train)),
                       ('Column', Column(X_train, data.numerical_column, data.categorical_column)),
                       ('Scale', Scale(X_train, data.numerical_column, data.categorical_column))]

for cnt, pertubation in enumerate(DEFAULT_PERTUBATION):

    print(cnt, pertubation[0], pertubation[1])

    X_train_perturbed = pertubation[1].perturbe()

    ppp.fit_ppp(X_train_perturbed, y_train, hyperparameter)

    # ToDo: FIT BAYESIAN OPTIMIZATION GET HP

    if SAVE:
        # Save Folder
        MODEL_PATH_TEMP = MODEL_PATH + str(cnt) + '/'
        os.makedirs(MODEL_PATH_TEMP)

        # Save Classifier, Predictor and Meta_Features
        pickle.dump(classifier, open(MODEL_PATH_TEMP + 'classifier.pickle', 'wb'))
        pickle.dump(ppp.predictor, open(MODEL_PATH_TEMP + 'predictor.pickle', 'wb'))
        pickle.dump(ppp.meta_features, open(MODEL_PATH_TEMP + 'List_predictor.pickle', 'wb'))

        ## WRITE INFORMATION ABOUT CURRENT MODEL
        file = open(MODEL_PATH_TEMP + "Information.txt", "w+")

        file.write('Model number: ' + str(cnt) + '\n\n')
        file.write('Name: ' + str(pertubation[0])+ '\n\n')
        file.write('Pertubation: ' + str(pertubation[1].name())+ '\n\n')
        file.write('List for Predictor: ' + str(ppp.meta_features) + '\n\n' )
        mu, cov = ppp.predictor.predict(ppp.meta_features, return_cov=True)
        file.write('Mean: ' + str(mu) + '\n\n' )
        file.write('Covariance: ' + str(cov) + '\n\n')

        file.close()

    print(pertubation[1].name())

    print('Score = ', ppp.meta_scores)
    print()

    ##ToDo: TRAIN CLASSIFIER with new HP
    #classifier = KNeighborsClassifier(n_neighbors=)
    #classifier
    #ppp.classifier = classifier