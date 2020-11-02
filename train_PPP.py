from Data.toy_data_class import toy_data_class
from Pertubations.numeric_class import Column, Scale, NoPertubation
from PPP_class import PPP_class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import os
import pickle
from Data.read_heart import heart
import csv

# Toy data
data = toy_data_class()
X_train, X_test, y_train, y_test = data.load()

# Config
SAVE = True
MODEL_PATH = 'models/'
ppp = PPP_class()

# Classifier
classifier = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 2, 3]}

# Pertubation
DEFAULT_FRACTION = [0.1]
DEFAULT_SCALER = [1]
DEFAULT_PERTUBATION = [('No_Perturbation', NoPertubation(X_train))]
for frac in DEFAULT_FRACTION:
    DEFAULT_PERTUBATION.append(('Column_fraction_'+ str(frac), Column(X_train, fraction=frac, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))
for frac in DEFAULT_FRACTION:
    for sca in DEFAULT_SCALER:
        DEFAULT_PERTUBATION.append(('Scale_fraction_' + str(frac) + '_scaler_' + str(sca) , Scale(X_train, fraction=frac, scaler=sca, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))

if SAVE:
    # Write CSV
    file_information = open(MODEL_PATH + 'PPP_train.csv', 'w')
    file_information.write('Pertubation;Output_Score;MU;Sigma;Meta_features\n')


for name, pertubation in DEFAULT_PERTUBATION:
    print(name, pertubation)

    X_train_perturbed = pertubation.perturbe()

    # Gridsearch
    clf = GridSearchCV(classifier, parameters)
    clf.fit(X_train_perturbed, y_train)

    # Get best Hyperparameters to fit in GP
    # String HP needs to convert
    best_hyperparameter = []
    for keys in parameters.keys():
        best_hyperparameter.append(clf.best_params_[keys])

    # PPP
    ppp.classifier = clf
    ppp.fit_ppp(X_train_perturbed, y_train, best_hyperparameter)

    mu, sigma = ppp.predict_ppp(X_train_perturbed, best_hyperparameter)
    #print('PPP Predict', mu, sigma)

    #print(ppp.meta_features)

    if SAVE:
        file_information.write(str(name) + ';' + str(ppp.meta_scores) + ';' + str(mu) + ';' + str(sigma) + ';' +
                               str(ppp.meta_features) + '\n')
        # Save Folder
        MODEL_PATH_TEMP = MODEL_PATH + str(name) + '/'
        os.makedirs(MODEL_PATH_TEMP)

        # Save Classifier, Predictor and Meta_Features
        pickle.dump(clf, open(MODEL_PATH_TEMP + 'classifier.pickle', 'wb'))
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


#print('END')
if SAVE:
    pickle.dump(ppp.predictor, open(MODEL_PATH + 'predictor.pickle', 'wb'))
    file_information.close()