from abc import ABC, abstractmethod
import numpy as np


class CDataPerturb(ABC):
    """abstract interface to define data perturbation model"""

    @abstractmethod
    def data_perturbation(self, x):
        """

        :param x: flat vector containing n_features elements
        :return:
        """
        raise NotImplementedError("data_perturbation not implemented")

    def perturb_dataset(self, x):
        """

        :param x: a matrix fo shape= n_samples, n_features
        :return:
        """
        # implementing data_perturbdataset(x)
        # 1. initialize Xp
        # 2. loop over the rows of X, then at each iteration: point 3)
        # 3. extract the given row
        # 4. apply the data perturbation function
        # 5. copy the result in Xp
        xp = np.zeros(shape=x.shape)
        for i in range(xp.shape[0]):
            xp[i, :] = self.data_perturbation(x[i, :])
        return xp

