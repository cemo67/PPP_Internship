import os
import pickle
from Data.toy_data_class import toy_data_class
from PPP_class import PPP_class
import sys
from Data.read_heart import heart

DEFAULT_PATH = 'models/'
models = []
predictor = pickle.load(open(DEFAULT_PATH + 'predictor.pickle', 'rb'))

# Read Heart data
data = heart()
X_train, X_test, y_train, y_test = data.get_train_test()

for folder in os.listdir(DEFAULT_PATH):
    print(folder)
    PATH_TEMP = os.path.join(DEFAULT_PATH, folder)

    if os.path.isdir(PATH_TEMP):
        models = (pickle.load(open(PATH_TEMP + '/classifier.pickle', 'rb')))

    best_hyperparameter = []
    for keys in models.best_params_.keys():
        best_hyperparameter.append(models.best_params_[keys])

    ppp = PPP_class(models, predictor)

    mu, sigma = ppp.predict_ppp(X_train, best_hyperparameter)

    print('Model Score: ', models.score(X_train, y_train))
    print('Predictor Score: ', mu, sigma)

toy_data = toy_data_class()
X_train, X_test, y_train, y_test = toy_data(toy_data.blobs())






sys.exit()

for classifier in models:



    best_hyperparameter = []
    for keys in classifier.best_params_.keys():
        best_hyperparameter.append(classifier.best_params_[keys])

    ppp = PPP_class(classifier, predictor)

    mu, sigma = ppp.predict_ppp(X_train, best_hyperparameter)

    print('Model Score: ', classifier.score(X_train, y_train))
    print('Predictor Score: ', mu, sigma)