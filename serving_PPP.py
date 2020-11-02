import os
import pickle
from Data.toy_data_class import toy_data_class
from PPP_class import PPP_class
import sys
from Data.read_heart import heart

SAVE = True
MODEL_PATH = 'models/'
predictor = pickle.load(open(MODEL_PATH + 'predictor.pickle', 'rb'))
ppp = PPP_class(None, predictor)

# Toy data
data = toy_data_class()
X_train, X_test, y_train, y_test = data.load()

if SAVE:
    # Write CSV
    file_information = open(MODEL_PATH + 'PPP_test.csv', 'w')
    file_information.write('Pertubation;Output_Score;MU;Sigma;Meta_features\n')

for folder in os.listdir(MODEL_PATH):
    print(folder)
    PATH_TEMP = os.path.join(MODEL_PATH, folder)

    if os.path.isdir(PATH_TEMP):
        models = (pickle.load(open(PATH_TEMP + '/classifier.pickle', 'rb')))

        best_hyperparameter = []
        for keys in models.best_params_.keys():
            best_hyperparameter.append(models.best_params_[keys])

        ppp.classifier = models

        mu, sigma = ppp.predict_ppp(X_train, best_hyperparameter)

        meta_scores = models.score(X_train, y_train)

        #print('Model Score: ', models.score(X_train, y_train))
        #print('Predictor Score: ', mu, sigma)
        #print(ppp.meta_features)
        if SAVE:
            file_information.write(str(folder) + ';' + str(meta_scores) + ';' + str(mu) + ';' + str(sigma) + ';' + str(ppp.meta_features) + '\n')

if SAVE:
    file_information.close()