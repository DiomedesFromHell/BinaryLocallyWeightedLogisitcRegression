import numpy as np
import math
from numpy import linalg
import matplotlib.pyplot as plt
import random

def read_data(filename):
    # data = np.genfromtxt(file, dtype=None, delimiter="\t", encoding=None)
    # return data
    lst = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            lst.append(tuple(float(item) for item in line.split(' ') if item != ''))
    return lst


def prepare_training_data(file_x, file_y):
    X = np.array(read_data(file_x))
    Y = np.array(read_data(file_y))
    return X, Y


def prepare_test_data():
    feature1 = []
    feature2 = []
    for i in range(500):
        sign1 = random.randint(1, 2)
        sign2 = random.randint(1, 2)
        x = random.random()
        y = random.random()
        if sign1 == 1:
            x = -x
        if sign2 == 1:
            y = -y
        feature1.append(x)
        feature2.append(y)
    return feature1, feature2


class LWLogR(object):

    def __init__(self, X, Y):
        if not X.shape[0] == Y.shape[0]:
            raise ValueError('Incompatible training data dimensions.')
        self.X = X
        self.Y = Y.ravel()

    def predict(self, x, tau=0.4, reg=0.0001):
        weights = np.exp(np.sum(np.abs((np.tile(x, (self.X.shape[0], 1)) - self.X))**2, axis=-1)/(-2*tau**2))
        theta = np.zeros((1, self.X.shape[1]))
        delta = np.zeros((1, self.X.shape[1]))
        while True:
            h = 1./(1+np.exp(-np.sum(np.tile(theta, (self.X.shape[0], 1))*self.X, axis=-1)))

            grad = ((np.dot(self.X.T, weights*(self.Y-h))) - reg*theta).ravel()
            d = np.diag(weights*(h*(np.ones(h.shape)-h)))
            hess = np.dot(self.X.T, np.dot(np.diag(-weights*h*(np.ones(h.shape)-h), 0), self.X)) - reg*np.diag(np.ones(self.X.shape[1]), 0)
            delta = np.dot(linalg.inv(hess), grad)
            if np.sum(delta**2, axis=-1)**(1./2) <= 0.000001:
                break
            theta -= delta.T
        return 1./(1+math.exp(-np.dot(theta, x)))

# array = read_data("view-source_cs229.stanford.edu_ps_ps1_logistic_x.txt")
# # shape = array.shape
# # print(shape)
# print(array)
# print(len(array))
#
# ndarr = np.array(array)

train_data = prepare_training_data("x.dat", "y.dat")
learner = LWLogR(train_data[0], train_data[1])
x = (-3.4792627e-01, 8.6257310e-01)

res = learner.predict(x)
print(res)
# print(ndarr.shape)

x1, x2 = prepare_test_data()

y = []
for i in range(len(x1)):
    prob = learner.predict((x1[i], x2[i]))
    if prob >= 0.5:
        y.append(1)
    else:
        y.append(0)

plt.scatter(x=x1, y=x2, c=y)
plt.show()
