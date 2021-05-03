import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from calc_accurcies import logistic_calc_accuracies

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

dataframe = pd.read_csv(os.getcwd() + '/titanic/processed_dataset_of_knowns.csv')
ratings = dataframe['Survived']
features_to_use = [col for col in dataframe.columns if col != 'Survived' and col != 'id']
dataframe = dataframe[features_to_use]
removal_indices = []
full_dataframe_training_accuracy, full_dataframe_testing_accuracy = logistic_calc_accuracies(dataframe, ratings)

for index, feature in enumerate(features_to_use[:100]):
    new_features_to_use = [feature for index, feature in enumerate(features_to_use) if index not in removal_indices]
    boolean = new_features_to_use == features_to_use
    new_index = new_features_to_use.index(feature)
    training_accuracy, testing_accuracy = logistic_calc_accuracies(dataframe[new_features_to_use[new_index:] + new_features_to_use[:new_index + 1]], ratings)
    _, new_accuracy = logistic_calc_accuracies(dataframe[new_features_to_use], ratings)
    print("\ncandidate for removal:", feature, '( index', index, ')')
    print("\tTraining:", training_accuracy)
    print("\tTesting:", testing_accuracy)
    if testing_accuracy > full_dataframe_testing_accuracy and training_accuracy > full_dataframe_training_accuracy:
        removal_indices.append(index)
    else:
        print('\tKept')
    print("\tbaseline testing accuracy:", new_accuracy)
    print("\tremoved indices:", removal_indices)

new_dataframe_training_accuracy, new_dataframe_testing_accuracy = logistic_calc_accuracies(dataframe[new_features_to_use], ratings)
full_dataframe_training_accuracy, full_dataframe_testing_accuracy = logistic_calc_accuracies(dataframe, ratings)

print("\nOld Accuracies")
print("\tTraining:", full_dataframe_training_accuracy)
print("\tTesting:", full_dataframe_testing_accuracy)

print("\nNew Accuracies")
print("\tTraining:", new_dataframe_training_accuracy)
print("\tTesting:", new_dataframe_testing_accuracy)