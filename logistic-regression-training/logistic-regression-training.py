import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def cost(p, y):
    N = len(y)
    return - (1 / N) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    m, n = X.shape
    
    bias = 0.0
    weight = np.zeros(n)
    
    for _ in range(steps):
        z = np.dot(X, weight) + bias
        h = _sigmoid(z)

        w_grad = np.dot(X.T, (h - y)) / m
        b_grad = np.mean(h - y)

        weight = weight - lr * w_grad
        bias = bias - lr * b_grad

    return weight, bias
    