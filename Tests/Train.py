from Data.toy_data_class import toy_data_class
from Pertubations.Pertubation_List import get_pertubation_list, get_file_list
from PPP_class import PPP_class
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
import sys

# Config
SAVE = True
DATA_NAME_LIST = get_file_list()
k_range = 15
best_sore = [1, 0] # [HP , Score]

for Data_name in DATA_NAME_LIST:
    print(Data_name)
    print()

    data = toy_data_class()
    X_train, X_test, y_train, y_test = data.load(Data_name)

    PERTUBATION_LIST = get_pertubation_list(X_train, data)

    MODEL_PATH = '../models/' + str(Data_name) + '/'

    for hp_1 in range(1, k_range + 1):
        print('K=',hp_1)
        print()

        # Classifier
        classifier = KNeighborsClassifier(n_neighbors=hp_1)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_train, y_train)
        if score > best_sore[1]:
            best_sore[0] = hp_1
            best_sore[1] = score

        # Predictor
        ppp = PPP_class(classifier=classifier)

        if SAVE:
            # Save Folder
            MODEL_PATH_TEMP = MODEL_PATH + str(hp_1) + '/'
            os.makedirs(MODEL_PATH_TEMP)

            # Write CSV
            file_information = open(MODEL_PATH_TEMP + 'PPP_train.csv', 'w')
            file_information.write('Pertubation;Output_Score\n')

            ppp.fit_ppp(y_train, PERTUBATION_LIST, file_information)
        else:
            ppp.fit_ppp(y_train, PERTUBATION_LIST)

        if SAVE:
            # Classifier
            pickle.dump(ppp.classifier, open(MODEL_PATH_TEMP + 'classifier.pickle', 'wb'))

            # Predictor
            pickle.dump(ppp.predictor, open(MODEL_PATH_TEMP + 'predictor.pickle', 'wb'))

            file_information.close()

    if SAVE:
        file_best_model = open(MODEL_PATH + 'best_model.txt', 'w')
        file_best_model.write('HP, Best Score\n')
        file_best_model.write(str(best_sore))
        file_best_model.close()

print('DONE!')