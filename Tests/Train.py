import sys
sys.path.append("../")
from Data.toy_data_class import toy_data_class
from Pertubations.Pertubation_List import get_pertubation_list, get_config
from PPP_class import PPP_class
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle

# Config
config_dict = get_config()

DATA_NAME_LIST = config_dict['file_list']
k_range = config_dict['k_range']
samples_ = config_dict['samples']

for Data_name in DATA_NAME_LIST:
    print(Data_name)
    print()

    data = toy_data_class(samples=samples_, name = Data_name)
    X_train, X_test, y_train, y_test = data.load()

    PERTUBATION_LIST = get_pertubation_list(X_train, data)

    MODEL_PATH = '../models/' + str(Data_name) + '/'

    for hp_1 in range(1, k_range + 1):
        print('K=',hp_1)
        print()

        # Classifier
        classifier = KNeighborsClassifier(n_neighbors=hp_1)
        classifier.fit(X_train, y_train)

        # Predictor
        ppp = PPP_class(classifier=classifier)

        # Save Folder
        MODEL_PATH_TEMP = MODEL_PATH + str(hp_1) + '/'
        os.makedirs(MODEL_PATH_TEMP)

        # Write CSV
        file_information = open(MODEL_PATH_TEMP + 'PPP_train.csv', 'w')
        file_information.write('Pertubation;Output_Score\n')

        ppp.fit_ppp(y_train, PERTUBATION_LIST, file_information)

        # Classifier
        pickle.dump(ppp.classifier, open(MODEL_PATH_TEMP + 'classifier.pickle', 'wb'))

        # Predictor
        pickle.dump(ppp.predictor, open(MODEL_PATH_TEMP + 'predictor.pickle', 'wb'))

        file_information.close()

print('DONE!')