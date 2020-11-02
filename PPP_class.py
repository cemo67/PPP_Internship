import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class PPP_class:

    def __init__(self, classifier=None, predictor=None):
        self.classifier = classifier
        if predictor == None:
            # Predictor
            kernel = 1.0 * RBF(1.0)
            predictor = GaussianProcessRegressor(kernel=kernel)
        self.predictor = predictor

    # Computes Quantiles for predict proba.
    # These will fit into a Resgressor later.
    @staticmethod
    def compute_ppp_features(predictions, bins_per_class_output=5):
        return np.percentile(predictions,
                             np.arange(0, 101, bins_per_class_output),
                             axis=0).flatten()

    def fit_ppp(self, X_train_perturbed, y_train, hyperparameter):
        self.meta_features = []
        self.meta_scores = []
        predictions = self.classifier.predict_proba(X_train_perturbed)
        self.meta_features.append(self.compute_ppp_features(predictions))
        self.meta_scores.append(self.classifier.score(X_train_perturbed, y_train))
        self.meta_features = [np.insert(self.meta_features[0], 0, hyperparameter)]

        print('PPP Score', self.meta_scores)

        self.predictor.fit(self.meta_features, self.meta_scores)

    def predict_ppp(self, X_perturbed, hyperparameter):
        predictions = self.classifier.predict_proba(X_perturbed)
        self.meta_features = (self.compute_ppp_features(predictions))

        self.meta_features = [np.insert(self.meta_features, 0, hyperparameter)]

        return self.predictor.predict(self.meta_features, return_std=True)