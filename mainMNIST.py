import numpy as np
from utils import load_mnist_data, split_data, \
    plot_ten_digits
from nmc import NMC
from sklearn.model_selection import ShuffleSplit
x, y = load_mnist_data()
img = plot_ten_digits(x, y)
print(img)

# we are using a class from sklearn named shuffle split instead of n_rep as before

splitter = ShuffleSplit(n_splits=5, train_size=.5)
test_error = np.zeros(shape=(splitter.n_splits,))
clf = NMC()
for i, (tr_idx, ts_idx) in enumerate(splitter.split(x, y)):
    x_tr, y_tr = x[tr_idx, :], y[tr_idx]
    x_ts, y_ts = x[ts_idx, :], y[ts_idx]
    clf.fit(x_tr, y_tr)
    ypred = clf.predict(x_ts)
    test_error[i] = (ypred != y_ts).mean()
    print(test_error[i])

# clf.robust_est = True

# for r in range(n_splits):
#    x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=1000, n_ts=None)
#    clf = NMC()
#    clf.fit(x_tr, y_tr)
#    y_pred = clf.predict(x_ts)
#    test_error[r] = (y_pred != y_ts).mean(), test_error.std())


