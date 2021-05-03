import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from calc_accuracies import logistic_calc_accuracies

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

dataframe = pd.read_csv(os.getcwd() + '/titanic/processed_dataset_of_knowns.csv')

ratings = dataframe['Survived']

iterations = 1000
full_dataframe_training_accuracy, full_dataframe_testing_accuracy = logistic_calc_accuracies(dataframe, ratings, iterations)
while full_dataframe_training_accuracy >= 1:
    iterations -= 1
    full_dataframe_training_accuracy, full_dataframe_testing_accuracy = logistic_calc_accuracies(dataframe, ratings, iterations)
    print(iterations, full_dataframe_training_accuracy, full_dataframe_testing_accuracy)

print(iterations+1)