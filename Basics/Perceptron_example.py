import numpy as np
from Perceptron import Perceptron

y = np.array([-1, 1, 1, 1])
X = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 8, 9],
    [4, 5, 6, 10, 12, 14]
])

ppn = Perceptron(eta=0.1, n_iter=10)
print(X.shape, y.shape)
ppn.fit(X, y)

print(ppn.w_)
print(ppn.errors_)

ppn = Perceptron(eta=0.1, n_iter=10, random_state=14)
ppn.fit(X, y)

print(ppn.w_)
print(ppn.errors_)
