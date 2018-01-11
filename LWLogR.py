import numpy as np
import math
from numpy import linalg


class LWLogR(object):

    def __init__(self, X, Y):
        if not X.shape[0] == Y.shape[0]:
            raise ValueError('Incompatible training data dimensions.')
        self.X = X
        self.Y = Y.ravel()

    def predict(self, x, tau=0.4, reg=0.0001, accuracy=0.000001):
        weights = np.exp(np.sum(np.abs((np.tile(x, (self.X.shape[0], 1)) - self.X))**2, axis=-1)/(-2*tau**2))
        theta = np.zeros((1, self.X.shape[1]))

        while True:
            h = 1./(1+np.exp(-np.sum(np.tile(theta, (self.X.shape[0], 1))*self.X, axis=-1)))

            grad = ((np.dot(self.X.T, weights*(self.Y-h))) - reg*theta).ravel()

            hess = np.dot(self.X.T, np.dot(np.diag(-weights*h*(np.ones(h.shape)-h), 0), self.X)) - reg*np.diag(np.ones(self.X.shape[1]), 0)
            delta = np.dot(linalg.inv(hess), grad)
            if np.sum(delta**2, axis=-1)**(1./2) <= accuracy:
                break
            theta -= delta.T
        return 1./(1+math.exp(-np.dot(theta, x)))



