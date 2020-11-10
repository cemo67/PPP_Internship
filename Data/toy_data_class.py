import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import pickle

class toy_data_class:

    # cluster_std = The standard deviation of the clusters.
    def __init__(self, samples=100000, classes=2, features=2, test_size=0.2, random_seed=0, cluster_std=1, center_box=[-10, 10]):
        self.samples = samples
        self.classes = classes
        self.features = features
        self.test_size = test_size
        self.numerical_column = [0, 1]
        self.categorical_column = None
        self.random=random_seed
        self.cluster_std = cluster_std
        self.center_box = center_box

    def moons(self):
        self.X, self.y = make_moons(n_samples=self.samples)

    def circles(self):
        self.X, self.y = make_circles(n_samples=self.samples)

    def s_curve(self):
        self.X, self.y = make_s_curve(n_samples=self.samples)

    def blobs(self):
        self.X, self.y = make_blobs(n_samples=self.samples, centers=self.classes, n_features=self.features,
                          random_state=self.random, cluster_std=self.cluster_std, center_box=self.center_box)


    def plot_data(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.y,
                    s=25, edgecolor='k')
        plt.show()

    def get_train_test(self):
        return train_test_split(self.X, self.y, test_size=self.test_size)

    def __call__(self, name, plot = False):
        name
        if plot:
            self.plot_data()
        return self.get_train_test()

    def save(self):
        pickle.dump(self.X_train, open('toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_X_train.pickle', 'wb'))
        pickle.dump(self.X_test, open('toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_X_test.pickle', 'wb'))
        pickle.dump(self.y_train, open('toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_y_train.pickle', 'wb'))
        pickle.dump(self.y_test, open('toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_y_test.pickle', 'wb'))

    def load(self):
        X_train = pickle.load(open('Data/toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_X_train.pickle', 'rb'))
        X_test = pickle.load(open('Data/toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_X_test.pickle', 'rb'))
        y_train = pickle.load(open('Data/toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_y_train.pickle', 'rb'))
        y_test = pickle.load(open('Data/toy_data_blobs_random_state_'+ str(self.random) +'_feature_2_classes_2_y_test.pickle', 'rb'))

        return X_train, X_test, y_train, y_test


t = toy_data_class()
X_train, X_test, y_train, y_test = t(t.blobs(), True)
#t.save()