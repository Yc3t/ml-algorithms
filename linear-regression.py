import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # n iterations to find the minimum gradient descent
        for _ in range(self.n_iters):
            # ŷ = wx +b

            y_predicted = np.dot(X, self.weights) + self.bias

            # dw = 1/N sum(2xi(ŷ-yi))

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #update weights
            # w= w-α * dw
            self.weights -= self.lr *dw

            #update weights
            # b= b-α * db
 
            self.bias -= self.lr * db


    def predict(self, X):
        y_predicted = np.dot(X,self.weights) + self.bias
        return y_predicted
