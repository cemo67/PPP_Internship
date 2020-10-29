from Data.toy_data_class import toy_data_class
from Pertubations.numeric_class import Column, Scale, NoPertubation
from PPP_class import PPP_class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import os
import pickle
from Data.read_heart import heart
import csv

# Toy data
data = toy_data_class()
X_train, X_test, y_train, y_test = data(data.blobs())

# Read Heart data
data = heart()
X_train, X_test, y_train, y_test = data.get_train_test()

# Config
SAVE = True
MODEL_PATH = 'models/'

DEFAULT_FRACTION = [0.1, 0.3]
DEFAULT_SCALER = [3, 10, 20, 25]

# Predictor
kernel = 1.0 * RBF(1.0)
predictor = GaussianProcessRegressor(kernel = kernel)

DEFAULT_PERTUBATION = [('No_Perturbation', NoPertubation(X_train))]

for frac in DEFAULT_FRACTION:
    DEFAULT_PERTUBATION.append(('Column_fraction_'+ str(frac), Column(X_train, fraction=frac, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))

for frac in DEFAULT_FRACTION:
    for sca in DEFAULT_SCALER:
        DEFAULT_PERTUBATION.append(('Scale_fraction_' + str(frac) + '_scaler_' + str(sca) , Scale(X_train, fraction=frac, scaler=sca, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))

# Write CSV
file_information = open('models/PPP.csv', 'w')
file_information.write('TRAIN;Classifier Predict\n')


for name, pertubation in DEFAULT_PERTUBATION:
    print(name, pertubation)

    X_train_perturbed = pertubation.perturbe()

    # Classifier
    classifier = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 2, 3]}

    # Gridsearch
    clf = GridSearchCV(classifier, parameters)
    clf.fit(X_train_perturbed, y_train)

    # Get best Hyperparameters to fit in GP
    # String HP needs to convert
    best_hyperparameter = []
    for keys in parameters.keys():
        best_hyperparameter.append(clf.best_params_[keys])

    # PPP
    ppp = PPP_class(clf, predictor)
    ppp.fit_ppp(X_train_perturbed, y_train, best_hyperparameter)

    mu, sigma = ppp.predict_ppp(X_train_perturbed, best_hyperparameter)
    print('PPP Predict', mu, sigma)

    file_information.write(str(name) + ';' + str(mu) + ';' + str(sigma) + '\n')

    if SAVE:
        # Save Folder
        MODEL_PATH_TEMP = MODEL_PATH + str(name) + '/'
        os.makedirs(MODEL_PATH_TEMP)

        # Save Classifier, Predictor and Meta_Features
        pickle.dump(clf, open(MODEL_PATH_TEMP + 'classifier.pickle', 'wb'))
        pickle.dump(ppp.predictor, open(MODEL_PATH + 'predictor.pickle', 'wb'))
        pickle.dump(ppp.meta_features, open(MODEL_PATH_TEMP + 'List_predictor.pickle', 'wb'))

        ## WRITE INFORMATION ABOUT CURRENT MODEL
        file = open(MODEL_PATH_TEMP + "Information.txt", "w+")

        file.write('Name: ' + str(name)+ '\n\n')
        file.write('Pertubation: ' + str(pertubation.name())+ '\n\n')
        file.write('Best Hyperparameter: ' + str(best_hyperparameter)+ '\n\n')
        file.write('List for Predictor: ' + str(ppp.meta_features) + '\n\n' )
        mu, cov = ppp.predictor.predict(ppp.meta_features, return_cov=True)
        file.write('Mean: ' + str(mu) + '\n\n' )
        file.write('Covariance: ' + str(cov) + '\n\n')

        file.close()


print('END')
file_information.close()