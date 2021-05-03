import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors  import KNeighborsClassifier

def linear_calc_accuracies(dataframe, ratings, iterations = 916, split = 500):
    training_x = np.array(dataframe[:split])
    training_y = np.array(ratings[:split])
    testing_x = np.array(dataframe[split + 1:])
    testing_y = np.array(ratings[split + 1:])
    coeffs = LogisticRegression(max_iter=iterations, random_state=0).fit(training_x, training_y)
    return coeffs.score(training_x, training_y), coeffs.score(testing_x, testing_y)

def logistic_calc_accuracies(dataframe, ratings, iterations = 916, split = 500):
    training_x = np.array(dataframe[:split])
    training_y = np.array(ratings[:split])
    testing_x = np.array(dataframe[split + 1:])
    testing_y = np.array(ratings[split + 1:])
    coeffs = LogisticRegression(max_iter=iterations, random_state=0).fit(training_x, training_y)
    return coeffs.score(training_x, training_y), coeffs.score(testing_x, testing_y)

def leave_one_out_classification(dataframe, k):
    correct = 0
    for i in range(len(dataframe.index)):
        removed_row = dataframe.loc[i]
        np_dataframe = np.array(dataframe.drop(index = i))
        training_x = [[element for element in row] for row in np_dataframe[:,1:]]
        training_y = [element for element in np_dataframe[:,0]]
        testing_x = removed_row[1:]
        testing_y = removed_row[0]
        KNN = KNeighborsClassifier(n_neighbors = k).fit(training_x, training_y)

        if KNN.predict([testing_x]) == testing_y:
            correct += 1
        
    return correct / len(dataframe.index)

