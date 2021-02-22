import numpy as np


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                delta_w = self.eta * (target - self.predict(xi))
                self.w_[1:] += delta_w * xi
                self.w_[0] += delta_w
                errors += int(delta_w != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]

        return np.where(z >= 0.0, 1, -1)
