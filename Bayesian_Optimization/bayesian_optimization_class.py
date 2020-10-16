import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import sys

class BO:

    def __init__(self, predictor, bounds, meta_features):
        self.predictor = predictor
        self.bounds = bounds    # Bounds for HP
        self.meta_features_copy = np.copy(meta_features) # Copy meta_features!! Otherwise Copy by reference
        print('BO')

    def expected_improvement(self, X, X_sample, xi=8):
        # Potential Candidate X
        mu, sigma = self.predictor.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        mu_sample_opt = self.predictor.predict(X_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            #ei[sigma == 0.0] = 0.0
            if sigma == 0.0: ei = 0

        print('*************************')


        print('MU', mu)
        print('SIGMA', sigma)
        print('mu_sample_opt', mu_sample_opt)
        print('Z', Z)
        print('norm.cdf(Z)', norm.cdf(Z))
        print('norm.pdf(Z)', norm.pdf(Z))

        print('*************************')



        return ei[0][0], mu

    def propose_location(self, n_restarts=25):
        self.mu_sample = 0
        max_val = 0
        X_sample = self.meta_features_copy

        ## np.random.randint(self.bounds[0], self.bounds[1], size=(n_restarts))
        # Find the best optimum by starting from n_restart different random points.
        for x0 in range(1, 10):
            print(x0)
            self.meta_features_copy[0][0] = x0
            ei_r, mu = self.expected_improvement(self.meta_features_copy, X_sample)
            # MIXIMIZATION
            if max_val < ei_r:
                print('+++++++++++++++++++++++++++')
                print('CHANGE!!')
                print('MAX ', max_val)
                print('EI', ei_r)
                print('+++++++++++++++++++++++++++')
                print()

                max_val = ei_r
                self.mu_sample_opt = mu
                best_hyperparameter = x0
                print()
                print()
            print()

            X_sample = np.vstack((X_sample, self.meta_features_copy))

        return best_hyperparameter


