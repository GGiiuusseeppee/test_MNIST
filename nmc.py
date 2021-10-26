import numpy as np
from sklearn.metrics import pairwise_distances


class NMC(object):
    """Class implementing the Nearest mean centroid NMC classification algorithm"""
    def __init__(self, robust_est=False):
        self._centroid = None  # init centroid
        """Create the NMC object. Robust_est=true use medians to estimate error otherwise use mean"""
        self._robust_est = robust_est

    @property
    def centroid(self):
        return self._centroid

    @property
    def robust_est(self):
        return self._robust_est

    @robust_est.setter
    def robust_est(self, value):
        if not isinstance(value, bool):
            raise TypeError("value is not bool")
        self._robust_est = bool(value)

    def fit(self, x_tr, y_tr):
        """Fit the model to the data(estimating centroids)"""
        n_classes = np.unique(y_tr).size
        # count how many different elements I have in y_tr
        n_features = x_tr.shape[1]
        self._centroid = np.zeros(shape=(n_classes, n_features))
        # extract images of a given class-->
        for k in range(n_classes):
            xk = x_tr[y_tr == k, :]
            if self._robust_est is False:
                self._centroid[k, :] = np.mean(xk, axis=0)
            else:
                self._centroid[k, :] = np.median(xk, axis=0)
        return self

    def predict(self, x_ts):
        if self._centroid is None:
            raise ValueError("Centroid not set. Run fit first!!")
        dist = pairwise_distances(x_ts, self._centroid)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
