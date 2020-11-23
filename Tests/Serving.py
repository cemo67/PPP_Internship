import os
import pickle
from Data.toy_data_class import toy_data_class
from PPP_class import PPP_class
from Pertubations.Pertubation_List import get_pertubation_list, get_file_list
import sys

# Config
SAVE = True
DATA_NAME_LIST = get_file_list()
MODEL_PATH = '../models/'
models = []

for Data_name in DATA_NAME_LIST:
    print(Data_name)
    print()

    data = toy_data_class()
    X_train, X_test, y_train, y_test = data.load(Data_name)

    PERTUBATION_LIST = get_pertubation_list(X_test, data)

    MODEL_PATH = '../models/' + str(Data_name) + '/'

    for folder in os.listdir(MODEL_PATH):
        MODEL_PATH_TEMP = os.path.join(MODEL_PATH, folder)

        # If Model
        if os.path.isdir(MODEL_PATH_TEMP):
            print(MODEL_PATH_TEMP)
            classifier = (pickle.load(open(MODEL_PATH_TEMP + '/classifier.pickle', 'rb')))
            predictor = pickle.load(open(MODEL_PATH_TEMP + '/predictor.pickle', 'rb'))
            models.append([str(folder), classifier])

    if SAVE:
        file_information = open(MODEL_PATH + '/PPP_test_All.csv', 'w')
        file_information.write('Pertubation')
        last_lement = models[-1][0]

        for index, element in models:
            file_information.write(';' + str(index) + ' Mu ; Sigma')

        file_information.write(';Best model\n')

    ppp = PPP_class(classifier, predictor)

    ppp.predict_all_models(y_test, PERTUBATION_LIST, models, file_information)

    if SAVE:
        file_information.close()

print('END!')