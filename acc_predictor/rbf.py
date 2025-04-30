from pySOT.surrogate import RBFInterpolant, CubicKernel, TPSKernel, LinearTail, ConstantTail
import numpy as np


class RBF:
    """ Radial Basis Function """

    def __init__(self, kernel='cubic', tail='linear'):
        self.kernel = kernel
        self.tail = tail
        self.name = 'rbf'
        self.model = None

    def fit(self, train_data, train_label):
        if self.kernel == 'cubic':
            kernel = CubicKernel
        elif self.kernel == 'tps':
            kernel = TPSKernel
        else:
            raise NotImplementedError("unknown RBF kernel")

        if self.tail == 'linear':
            tail = LinearTail
        elif self.tail == 'constant':
            tail = ConstantTail
        else:
            raise NotImplementedError("unknown RBF tail")

        self.model = RBFInterpolant(
            dim=train_data.shape[1], lb = np.zeros(train_data.shape[1]), ub = np.ones(train_data.shape[1]), kernel=kernel(), tail=tail(train_data.shape[1])) #raw is dim=train_data.shape[1]; no lb,ub

        for i in range(len(train_data)):
            #print('rbf testing',train_data,len(train_data))
            #print(train_data[i, :])
            #print(train_label[i])
            self.model.add_points(train_data[i, :], train_label[i]) #raw is train_data[i, :]

    def predict(self, test_data):
        assert self.model is not None, "RBF model does not exist, call fit to obtain rbf model first"
        return self.model.predict(test_data)
