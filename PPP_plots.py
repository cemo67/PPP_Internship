import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')


class PPP_plots:

    def __init__(self, X_train, y_train, X_test_perturbed, y_test, PATH, rows_plot, cols_plot):
        self.X_test_perturbed = X_test_perturbed
        self.X_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.x_min, self.x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        self.y_min, self.y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        self.path = PATH
        self.rows_plot = rows_plot
        self.cols_plot = cols_plot

    def plot_decision_surface(self, classifier):
        xx, yy = np.meshgrid(np.arange(self.x_min, self.x_max, 0.1),
                             np.arange(self.y_min, self.y_max, 0.1))

        # Plot decision surface
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)

        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, s=10, edgecolor='r')
        #plt.ylim([0, 8])

    def plot_all_in_one(self, classifier, index, plot_cnt):
        # !! (Rows, Columns in Plot)
        plt.axis('off')
        plt.subplot(self.rows_plot, self.cols_plot, 0 + plot_cnt + 1)

        self.plot_decision_surface(classifier)
        plt.scatter(self.X_test_perturbed[:, 0], self.X_test_perturbed[:, 1], c=self.y_test, s=20, edgecolors='b', marker='^')
        plt.title('K = ' + str(index))

    def plot_one(self, classifier, index):
        self.plot_decision_surface(classifier)
        plt.scatter(self.X_test_perturbed[:, 0], self.X_test_perturbed[:, 1], c=self.y_test, s=30, edgecolors='b', marker='^')
        plt.title('K = ' + str(index))
        plot_name = self.path + str(index) + '.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.clf()

    def save_and_clear(self):
        plot_name = self.path + '.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.clf()