import sys
sys.path.append("../")
import os
import pickle
from Data.toy_data_class import toy_data_class
from PPP_class import PPP_class
from Pertubations.Pertubation_List import get_pertubation_list, get_config
from PPP_plots import PPP_plots
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

# Config
config_dict = get_config()

DATA_NAME_LIST = config_dict['file_list']
k_range = config_dict['k_range']
samples_ = config_dict['samples']
rows_plot = config_dict['rows_plot']
cols_plot = config_dict['cols_plot']
PLOT_ALL_IN_ONE = config_dict['PLOT_ALL_IN_ONE']
PLOT = config_dict['PLOT']

for Data_name in DATA_NAME_LIST:
    print(Data_name)
    print()

    for sample in samples_:

        data = toy_data_class(samples=sample, name=Data_name)
        X_train, X_test, y_train, y_test = data.load()

        PERTUBATION_LIST = get_pertubation_list(X_test, data)

        MODEL_PATH = '../models/' + str(Data_name) + '_' + str(sample) + '/'

        file_information = open(MODEL_PATH + '/PPP_test_All.csv', 'w')
        file_information.write('Pertubation;')
        for i in range(k_range):
            file_information.write('Model;Mu;Sigma;Real Score;')
        file_information.write('BEST GP;BEST REAL\n')

        for name, pertubation in PERTUBATION_LIST:
            PLOT_PATH = MODEL_PATH + 'Plots/' + name + '/'
            os.makedirs(PLOT_PATH)

            X_test_perturbed = pertubation.perturbe()
            best_index_GP = ''
            best_mu_GP = 0
            best_index_REAL = ''
            best_mu_REAL = 0
            file_information.write(str(name) + ';')

            if PLOT:
                ppp_plot = PPP_plots(X_train, y_train, X_test_perturbed, y_test, PLOT_PATH, rows_plot, cols_plot)
                plot_cnt = 0

            for folder in os.listdir(MODEL_PATH):
                MODEL_PATH_TEMP = os.path.join(MODEL_PATH, folder)
                index = folder.split('/')[-1]

                if os.path.isdir(MODEL_PATH_TEMP) and index != 'Plots':
                    print(MODEL_PATH_TEMP)
                    classifier = (pickle.load(open(MODEL_PATH_TEMP + '/classifier.pickle', 'rb')))
                    predictor = pickle.load(open(MODEL_PATH_TEMP + '/predictor.pickle', 'rb'))
                else:
                    continue

                ppp = PPP_class(classifier=classifier, predictor=predictor)
                mu, sigma = ppp.predict_ppp(X_test_perturbed)
                real_score = classifier.score(X_test_perturbed, y_test)

                if PLOT:
                    if PLOT_ALL_IN_ONE:
                        ppp_plot.plot_all_in_one(classifier, index, plot_cnt)
                    else:
                        ppp_plot.plot_one(classifier, index)
                    plot_cnt += 1

                if mu > best_mu_GP:
                    best_mu_GP = mu
                    best_index_GP = str(index)
                elif mu == best_mu_GP:
                    best_index_GP += ' -- ' + str(index)

                if real_score > best_mu_REAL:
                    best_mu_REAL = real_score
                    best_index_REAL = str(index)
                elif real_score == best_mu_REAL:
                    best_index_REAL += ' -- ' + str(index)

                file_information.write(str(index) + ';' + str(mu) + ';' + str(sigma) + ';' + str(real_score) + ';')

            file_information.write(str(best_index_GP) + ';' + str(best_index_REAL) + '\n')
            if PLOT:
                if PLOT_ALL_IN_ONE:
                    ppp_plot.save_and_clear()

        file_information.close()
print('DONE!')