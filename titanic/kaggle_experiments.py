import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

dataframe = pd.read_csv(os.getcwd() + '/titanic/processed_dataset_of_knowns.csv')

ratings = dataframe['Survived']

def calc_accurcies(dataframe, ratings, iterations = 2750):
    training_x = np.array(dataframe[:500])
    training_y = np.array(ratings[:500])
    testing_x = np.array(dataframe[501:])
    testing_y = np.array(ratings[501:])
    coeffs = LogisticRegression(max_iter=iterations, random_state=0).fit(training_x, training_y)
    training_accuracy = coeffs.score(training_x, training_y)
    testing_accuracy = coeffs.score(testing_x, testing_y)
    return training_accuracy, testing_accuracy

iterations = 1000
full_dataframe_training_accuracy, full_dataframe_testing_accuracy = calc_accurcies(dataframe, ratings, iterations)
while full_dataframe_training_accuracy >= 1:
    iterations -= 1
    full_dataframe_training_accuracy, full_dataframe_testing_accuracy = calc_accurcies(dataframe, ratings, iterations)
    print(iterations, full_dataframe_training_accuracy, full_dataframe_testing_accuracy)

print(iterations+1)