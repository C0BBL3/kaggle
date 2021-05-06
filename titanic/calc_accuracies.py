import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors  import KNeighborsClassifier

def linear_calc_accuracies(dataframe, ratings, iterations = 916, split = 500):
    return regression(dataframe, ratings, LinearRegression)

def logistic_calc_accuracies(dataframe, ratings, iterations = 916, split = 500):
    return regression(dataframe, ratings, LogisticRegression)

def regression(dataframe, ratings, regression, iterations = 916, split = 500):
    training_x = np.array(dataframe[:split])
    training_y = np.array(ratings[:split])
    testing_x = np.array(dataframe[split + 1:])
    testing_y = np.array(ratings[split + 1:])
    coeffs = regression(max_iter=iterations, random_state=0).fit(training_x, training_y)
    return coeffs.score(training_x, training_y), coeffs.score(testing_x, testing_y)

def leave_one_out_classification(dataframe, k):
    np_dataframe = np.array(dataframe)
    x = [[element for element in row] for row in np_dataframe[:,1:]]
    y = [element for element in np_dataframe[:,0]]
    return knn(x, y, k, len(dataframe.index))

def leave_one_out_classification_v2(dataframe, k, prediction_column):
    prediction_column_index = list(dataframe.columns).index(prediction_column)
    x_features = [index for index, feature in enumerate(dataframe.columns) if feature != prediction_column]
    np_dataframe = np.array(dataframe)
    x = [[element for column_index, element in enumerate(row) if column_index in x_features] for row in np_dataframe]
    y = [int(element[prediction_column_index]) for element in np_dataframe]
    return knn(x, y, k, len(dataframe.index))

def knn(x, y, k, len_dataframe):
    correct = 0
    for i in range(len_dataframe):
        training_x, training_y = x[:i] + x[i + 1:], y[:i] + y[i + 1:]
        testing_x, testing_y = x[i], int(y[i])
        KNN = KNeighborsClassifier(n_neighbors = k)
        KNN.fit(training_x, training_y)
        if KNN.predict([testing_x]) == testing_y:
            correct += 1
        
    return correct / len_dataframe

