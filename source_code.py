
# coding: utf-8

# 기계학습개론 - 파이썬 머신러닝 발표

# 전전컴 2014440130 정현진

## 퍼셉트론 학습 알고리즘 소스코드


import numpy as np
import pandas as pd


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta: Learning rate (between 0.0 and 1.0)
    n_iter: Passes over the training dataset.

    Attributes
    -----------
    w_: Weights after fitting.
    errors_ : Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data."""
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)


# select 2 features - Sepal Length and Petal Length
X = df.iloc[0:100, [0, 2]].values

# get label (setosa is -1, another is 1)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# make perceptron object
learning_rate=0.001
num_of_iteration=10
ppn = Perceptron(learning_rate, num_of_iteration)

# train
ppn.fit(X, y)

#### Result
print("error:", ppn.errors_)

correct = 0
for i in range(len(X)):
	correct += int(y[i] == ppn.predict(X[i]))
print("correct:", correct, "/", len(X))
print("accuracy:", correct/len(X) * 100.0, "%")

