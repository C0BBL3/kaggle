import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv(os.getcwd() + '/titanic/processed_dataset_of_knowns.csv')

ratings = dataframe['Survived']
features_to_use = [col for col in dataframe.columns if col != 'Survived']

for index in range(len(features_to_use)):
    new_dataframe = dataframe[features_to_use[:index + 1]]
    training_x = np.array(new_dataframe[:500])
    training_y = np.array(ratings[:500])
    testing_x = np.array(new_dataframe[501:])
    testing_y = np.array(ratings[501:])
    coeffs = LogisticRegression(max_iter=1000).fit(training_x, training_y)
    print('\nIndex', index)
    print("\tTraining:", coeffs.score(training_x, training_y))
    print("\tTesting:", coeffs.score(testing_x, testing_y))