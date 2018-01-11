import numpy as np
import math


class LogisticRegression(object):

    def __init__(self, X, Y):
        if not X.shape[0] == Y.shape[0]:
            raise ValueError('Incompatible training data dimensions.')
        self.X = np.ones((X.shape[0], X.shape[1] + 1))
        self.X[:, :-1] = X
        self.Y = Y.ravel()
        self.theta = None

    def train(self, reg=0.0001, accuracy=0.000001):
        n = self.X.shape[1]
        m = self.X.shape[0]
        theta = np.zeros(n)
        while True:
            h = 1./(1 + np.exp(-self.Y*np.sum(np.tile(theta, (m, 1))*self.X, axis=-1)))
            grad = -(1./m) * self.X.T.dot(self.Y*(1 - h)) + reg*theta
            hess = (1./m) * self.X.T.dot(np.dot(np.diag(h*(1-h)), self.X)) + reg*np.diag(np.ones(n))
            delta = -np.linalg.inv(hess).dot(grad)
            if np.sum(delta**2, axis=-1)**(1./2) <= accuracy:
                break
            theta += delta
        self.theta = theta

    def predict(self, x, probabilities=False):
        if self.theta is None:
            raise ValueError('Model have to be trained before predictions can be made.')
        res = []
        for item in x:
            a = np.array(item)
            x = np.ones(len(item)+1)
            x[:-1] = a
            p = 1./(1 + math.exp(self.theta.dot(x)))
            if p >= 0.5:
                if probabilities:
                    res.append((-1, p))
                else:
                    res.append(-1)
            else:
                if probabilities:
                    res.append((1, p))
                else:
                    res.append(1)
        return res

