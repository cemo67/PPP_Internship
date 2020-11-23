import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import matplotlib.pyplot as plt

class PPP_class:

    def __init__(self, classifier, predictor=None):
        self.classifier = classifier
        if predictor == None:
            # Predictor
            kernel = 1.0 * RBF(1.0)
            predictor = GaussianProcessRegressor(kernel=kernel)
        self.predictor = predictor

    # Computes Quantiles for predict proba.
    # These will fit into a Resgressor later.
    # Calculates probability distribution for a class
    # Flattens the probability distribution, concatenate by row
    @staticmethod
    def compute_ppp_features(predictions, bins_per_class_output=5):
        return np.percentile(predictions,
                             np.arange(0, 101, bins_per_class_output),
                             axis=0).flatten()

    def fit_origin(self, X_train_perturbed, y_train, Pertubations_List):
        self.meta_features = []
        self.meta_scores = []
        predictions = self.classifier.predict_proba(X_train_perturbed)
        self.meta_features.append(self.compute_ppp_features(predictions))
        self.meta_scores.append(self.classifier.score(X_train_perturbed, y_train))


        print('META', self.meta_features)
        print(len(self.meta_features[0]))
        #for index, element in enumerate(self.meta_features[0]):
        #    print(index, element)


        print('Proba', predictions)
        print(len(predictions))
#        plt.plot(self.meta_features)
#        plt.show()

        print('PPP Score', self.meta_scores)
        print()
        print('####################')
        print()

        self.predictor.fit(self.meta_features, self.meta_scores)

    def fit_ppp(self, y_train, Pertubations_List, file_information=None):
        self.meta_features = []
        self.meta_scores = []

        cnt = 0
        for name, pertubation in Pertubations_List:

            X_train_perturbed = pertubation.perturbe()

            predictions = self.classifier.predict_proba(X_train_perturbed)
            self.meta_features.append(self.compute_ppp_features(predictions))
            self.meta_scores.append(self.classifier.score(X_train_perturbed, y_train))

            if file_information != None:
                file_information.write(str(name) + ';' + str(self.meta_scores[cnt]) + ';' + '\n')
            cnt+=1

        self.predictor.fit(self.meta_features, self.meta_scores)

    def predict_ppp_origin(self, X_perturbed, hyperparameter=None):
        self.meta_features = []
        predictions = self.classifier.predict_proba(X_perturbed)
        self.meta_features.append(self.compute_ppp_features(predictions))

        if hyperparameter != None:
            self.meta_features = [np.insert(self.meta_features, 0, hyperparameter)]

        return self.predictor.predict(self.meta_features, return_std=True)

    def predict_ppp(self, Pertubations_List, file_information= None):

        for name, pertubation in Pertubations_List:
            self.meta_features = []

            X_test_perturbed = pertubation.perturbe()

            predictions = self.classifier.predict_proba(X_test_perturbed)
            self.meta_features.append(self.compute_ppp_features(predictions))
            mu, sigma = self.predictor.predict(self.meta_features, return_std=True)

            if file_information != None:
                file_information.write(str(name) + ';' + str(mu) + ';' + str(sigma) + '\n')

    def predict_all_models(self, y_test, Pertubations_List, models_list, file_information):
        for name, pertubation in Pertubations_List:
            print(Pertubations_List)
            X_test_perturbed = pertubation.perturbe()
            best_index = 0
            best_mu = 0
            best_sigma = 0

            file_information.write(str(name) + ';')

            for index, model in models_list:
                print(index)
                print(model)
                self.meta_features = []
                predictions = model.predict_proba(X_test_perturbed)
                self.meta_features.append(self.compute_ppp_features(predictions))
                mu, sigma = self.predictor.predict(self.meta_features, return_std=True)

                file_information.write(str(mu) + ';' + str(sigma) + ';')

                if mu > best_mu:
                    best_index = index
                    best_mu = mu
                    best_sigma = sigma

            file_information.write( str(best_index) + '\n')
            file_information.write('Real Score;')

            best_index = 0
            best_mu = 0

            for index, model in models_list:
                real_score = model.score(X_test_perturbed, y_test)

                if real_score > best_mu:
                    best_index = index
                    best_mu = real_score

                file_information.write(str(real_score) + ';;' )

            file_information.write( str(best_index) + '\n')