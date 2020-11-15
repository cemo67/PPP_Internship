from Data.toy_data_class import toy_data_class
from Pertubations.numeric_class import Column, Scale, NoPertubation
from PPP_class import PPP_class
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle

# Toy data
data = toy_data_class()
X_train, X_test, y_train, y_test = data.load()

# Config
SAVE = True
MODEL_PATH = 'models/'
k_range = 3

# Pertubation
DEFAULT_FRACTION = [0.1]
DEFAULT_SCALER = [1]
PERTUBATION_LIST = [('No_Perturbation', NoPertubation(X_train))]
for frac in DEFAULT_FRACTION:
    PERTUBATION_LIST.append(('Column_fraction_' + str(frac), Column(X_train, fraction=frac, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))
for frac in DEFAULT_FRACTION:
    for sca in DEFAULT_SCALER:
        PERTUBATION_LIST.append(('Scale_fraction_' + str(frac) + '_scaler_' + str(sca) , Scale(X_train, fraction=frac, scaler=sca, numerical_column=data.numerical_column, categorical_column=data.categorical_column)))

for hp_1 in range(1, k_range + 1):
    print(hp_1)
    print()

    # Classifier
    classifier = KNeighborsClassifier(n_neighbors=hp_1)
    classifier.fit(X_train, y_train)

    # Predictor
    ppp = PPP_class(classifier=classifier)

    if SAVE:
        # Save Folder
        MODEL_PATH_TEMP = MODEL_PATH + str(hp_1) + '/'
        os.makedirs(MODEL_PATH_TEMP)

        # Write CSV
        file_information = open(MODEL_PATH_TEMP + 'PPP_train.csv', 'w')
        file_information.write('Pertubation;Output_Score;Meta_features\n')

    ppp.fit_ppp(y_train, PERTUBATION_LIST, file_information)

    if SAVE:
        # Classifier
        pickle.dump(ppp.classifier, open(MODEL_PATH_TEMP + 'classifier.pickle', 'wb'))

        # Predictor
        pickle.dump(ppp.predictor, open(MODEL_PATH_TEMP + 'predictor.pickle', 'wb'))

        file_information.close()

print('END!')