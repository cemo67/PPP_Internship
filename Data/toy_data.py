import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class Toy_Data:

    def __init__(self, samples=1000, classes=2, features=2, test_size=0.8):
        self.samples = samples
        self.classes = classes
        self.features = features
        self.test_size = test_size

    def moons(self):
        self.X, self.y = make_moons(n_samples=self.samples)

    def circles(self):
        self.X, self.y = make_circles(n_samples=self.samples)

    def s_curve(self):
        self.X, self.y = make_s_curve(n_samples=self.samples)

    def blobs(self):
        self.X, self.y = make_blobs(n_samples=self.samples, centers=self.classes, n_features=self.features,
                          random_state=0)


    def plot_data(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.y,
                    s=25, edgecolor='k')
        plt.show()

    def get_train_test(self):
        return train_test_split(self.X, self.y, test_size=self.test_size)

    def __call__(self, name, plot = False):
        name
        self.plot_data()
        return self.get_train_test()



#t = Toy_Data()
#X_train, X_test, y_train, y_test = t(t.blobs())