import matplotlib.pyplot as plt
import numpy as np
from nmc import NMC
from utils import load_mnist_data, split_data, \
    plot_ten_digits
from data_perturb import CDataPerturbRandom, CDataPerturbGaussian


def robustness_test(clf, data_pert, param_name, param_values):
    """
    Running  a robustness test on clf using data pert model
    the test is run by setting param_name to different values(param_values)
    Parameter
    ---------
    :param clf: is an object of the class NMC that implements fit and predict functions
    :param data_pert:
    :param param_name:
    :param param_values:
    :return:
    """
    test_accuracies = np.zeros(shape=param_values.shape)
    for i, k in enumerate(param_values):
        # perturb test set
        setattr(data_pert, param_name, k)
        xp = data_pert.perturb_dataset(x_ts)
        # plot_ten_digits(xp, y)
        # compute predicted labels on the perturbed ts
        y_pred = clf.predict(xp)
        # compute classification accuracy using y_pred
        clf_accuracy = np.mean(y_ts == y_pred)
        print("test accuracy(K=", k, "): ", int(clf_accuracy * 10000) / 100, "%")
        test_accuracies[i] = clf_accuracy
    return test_accuracies


x, y = load_mnist_data()
z = x[0, :].ravel()
param_values = np.array([0, 10, 20, 50, 100, 200, 300, 400, 500])

# plot_ten_digits(xp, y)

# split MNIST data into training and test sets
n_tr = int(0.6 * x.shape[0])
x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

clf = NMC()
clf.fit(x_tr, y_tr)
ypred = clf.predict(x_ts)
clf_accuracy = np.mean(y_ts == ypred)
print("Test accuracy:", int(clf_accuracy*10000)/100, '%')
# gaussian pert
plt.figure(figsize=(10, 5))
test_accuracies = robustness_test(clf, CDataPerturbGaussian(), param_name='sigma', param_values=param_values)
plt.subplot(1, 2, 1)
plt.plot(param_values, test_accuracies)
plt.xlabel(r'$\sigma$')
plt.ylabel(r'test accuracy($\sigma$)')

# random pert
plt.figure(figsize=(10, 5))
test_accuracies = robustness_test(clf, CDataPerturbRandom(), param_name='K', param_values=param_values)
plt.subplot(1, 2, 2)
plt.plot(param_values, test_accuracies)
plt.xlabel('K')
plt.ylabel('test accuracy(K)')
plt.show()



