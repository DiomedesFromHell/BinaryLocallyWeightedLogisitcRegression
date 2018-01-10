import numpy as np
import random
import matplotlib.pyplot as plt

from LWLogR import LWLogR


def read_data(filename):
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

train_data = prepare_training_data("x.dat", "y.dat")
learner = LWLogR(train_data[0], train_data[1])

x1, x2 = prepare_test_data()

y = []
for i in range(len(x1)):
    prob = learner.predict((x1[i], x2[i]))
    if prob >= 0.5:
        y.append(1)
    else:
        y.append(0)

plt.scatter(x=x1, y=x2, s=25, c=y)
plt.show()
