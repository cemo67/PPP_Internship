import os
import pickle
from Data.toy_data_class import toy_data_class
from PPP_class import PPP_class

# Toy data
data = toy_data_class()
X_train, X_test, y_train, y_test = data.load()

# Config
SAVE = True
MODEL_PATH = 'models/'

for folder in os.listdir(MODEL_PATH):
    MODEL_PATH_TEMP = os.path.join(MODEL_PATH, folder)

    if os.path.isdir(MODEL_PATH_TEMP):
        print(MODEL_PATH_TEMP)
        classifier = (pickle.load(open(MODEL_PATH_TEMP + '/classifier.pickle', 'rb')))
        predictor = pickle.load(open(MODEL_PATH_TEMP + '/predictor.pickle', 'rb'))
        ppp = PPP_class(classifier, predictor)

        mu, sigma = ppp.predict_ppp(X_train)

        real_meta_score = classifier.score(X_train, y_train)

        #print('Model Score: ', models.score(X_train, y_train))
        #print('Predictor Score: ', mu, sigma)
        #print(ppp.meta_features)
        if SAVE:
            file_information = open(MODEL_PATH_TEMP + '/PPP_test.csv', 'w')
            file_information.write('Pertubation;Output_Score;MU;Sigma;Meta_features\n')
            file_information.write(str(folder) + ';' + str(real_meta_score) + ';' + str(mu) + ';' + str(sigma) + ';' + str(ppp.meta_features) + '\n')
            file_information.close()

print('END!')