import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import pickle

class toy_data_class:

    # cluster_std = The standard deviation of the clusters.
    def __init__(self, samples=1000, name='blobs' ,classes=2, features=2, test_size=0.2, random_seed=0, cluster_std=1, center_box=[-10, 10]):
        self.samples = samples
        self.classes = classes
        self.features = features
        self.test_size = test_size
        self.numerical_column = [0, 1]
        self.categorical_column = None
        self.random=random_seed
        self.cluster_std = cluster_std
        self.center_box = center_box
        self.name = name

    def moons(self):
        self.name = 'moons'
        self.X, self.y = make_moons(n_samples=self.samples, random_state=self.random)

    def circles(self):
        self.name = 'circles'
        self.X, self.y = make_circles(n_samples=self.samples, random_state=self.random)

    def s_curve(self):
        self.name = 's_curve'
        self.X, self.y = make_s_curve(n_samples=self.samples, random_state=self.random)

    def blobs(self):
        self.name = 'blobs'
        self.X, self.y = make_blobs(n_samples=self.samples, centers=self.classes, n_features=self.features,
                          random_state=self.random, cluster_std=self.cluster_std, center_box=self.center_box)

    def cosine(self, noise=0):
        x = np.linspace(-np.pi, 2 * np.pi, self.samples // 2)
        x_1 = np.cos(x) + 1 + np.random.randn(self.samples // 2) * noise
        x_2 = np.cos(x) - 1 + np.random.randn(self.samples // 2) * noise
        self.X = np.vstack([np.hstack([x, x]), np.hstack([x_1, x_2])]).T
        self.y = np.hstack([np.ones(self.samples // 2), np.zeros(self.samples // 2)])

    def plot_data(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.y,
                    s=25, edgecolor='k')
        plt.show()

    def get_train_test(self, bool_cosine):
        if bool_cosine:
            train_test_split(self.X, self.y, test_size=self.test_size, shuffle=True)
        else:
            return train_test_split(self.X, self.y, test_size=self.test_size, shuffle=False)

    def __call__(self, name, plot = False):
        name
        if plot:
            self.plot_data()
        return self.get_train_test()

    def save(self):
        pickle.dump(self.X, open('Data/' + str(self.name) + '_samples_' + str(self.samples) + '_X.pickle', 'wb'))
        pickle.dump(self.y, open('Data/' + str(self.name) + '_samples_' + str(self.samples) + '_y.pickle', 'wb'))

    def load(self):
        self.X = pickle.load(open('Data/' + str(self.name) + '_samples_' + str(self.samples) + '_X.pickle', 'rb'))
        self.y = pickle.load(open('Data/' + str(self.name) + '_samples_' + str(self.samples) + '_y.pickle', 'rb'))

        if self.name == 'cosine':
            return self.get_train_test(True)
        else:
            return self.get_train_test(False)