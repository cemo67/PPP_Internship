import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale
import sys

class PPP:

    def __init__(self, classifier, predictor):
        self.classifier = classifier
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

        self.predictor.fit(self.meta_features, self.meta_scores)