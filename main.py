from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

def accuracy(y_true,y_pred):
    total_predictions = len(y_true)
    accuracy = np.sum(y_true == y_pred) / total_predictions
    return accuracy


def mse(y_true,y_predicted):
    return np.mean((y_true-y_predicted)**2)
    




def knn():
    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

    model = KNN(k=3)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))



def linear_regression():
    X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=2)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=32)
    regressor = LinearRegression(lr=0.01)
    regressor.fit(X_train,y_train)
    predicted = regressor.predict(X_test)
    mse_value = mse(y_test,predicted)
    print(mse_value)
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()




def main():
    #knn()
    linear_regression()
        


if __name__ == "__main__":
    main()
