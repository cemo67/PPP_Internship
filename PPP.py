import numpy as np

from Data.read_heart import heart
from Pertubations.numeric import Column, Scale
import sys

class PPP:

    def __init__(self, classifier, predictor):
        self.classifier = classifier
        self.predictor = predictor
        self.plot_meta = []
        self.plot_score = []

    # Computes Quantiles for predict proba.
    # These will fit into a Resgressor later.
    @staticmethod
    def compute_ppp_features(predictions, bins_per_class_output=5):
        return np.percentile(predictions,
                             np.arange(0, 101, bins_per_class_output),
                             axis=0).flatten()

    def fit_ppp(self, X_train_perturbed, y_train):
        meta_features = []
        meta_scores = []
        predictions = self.classifier.predict_proba(X_train_perturbed)
        meta_features.append(self.compute_ppp_features(predictions))
        meta_scores.append(self.classifier.score(X_train_perturbed, y_train))

        # To plot GP
        self.plot_meta.extend(meta_features)
        self.plot_score.append(meta_scores)

        self.score = meta_scores

        self.predictor.fit(meta_features, meta_scores)