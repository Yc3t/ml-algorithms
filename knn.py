import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    result = np.sqrt(np.sum((x1-x2)**2))    
    return result


class KNN:
    def __init__(self, k=0):
        self.k = k

    
    def fit(self,X, y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        y_predictions = []
        for x in X:
            prediction = self._predict(x)
            y_predictions.append(prediction)
    
        return np.array(y_predictions)

        
    def _predict(self,x):

        #calculate distances
        
        distances = []

        for x_train in self.X_train: ## for every sample in samples
            distance = euclidean_distance(x,x_train)
            #distance between input and training set
            distances.append(distance)
        
        # sort by distance (the lower distance the first) using k-neighbors
        k_idx  = np.argsort(distances)[:self.k] # takes the first k elements

        # extract the labels of the knn that are in the training data

        k_nearest_labels = []
        for i in k_idx:
            label = self.y_train[i]
            k_nearest_labels.append(label)

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]












