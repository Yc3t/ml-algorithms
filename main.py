from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from linear_regression import LinearRegression

def accuracy(y_true,y_pred):
    total_predictions = len(y_true)
    accuracy = np.sum(y_true == y_pred) / total_predictions
    return accuracy


def knn():
    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

    model = KNN(k=3)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))






def linear_regression():
    pass


def main():
    knn()
        


if __name__ == "__main__":
    main()
