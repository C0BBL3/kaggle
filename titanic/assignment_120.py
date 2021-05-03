import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from calc_accuracies import leave_one_out_classification

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


dataframe = pd.read_csv(os.getcwd() + '/titanic/processed_dataset_of_knowns.csv')
features_to_use = ['Survived', "Sex", "Pclass", "Fare", "Age", "SibSp"]

indices = [1,3,5,10,15,20,30,40,50,75]

plt.plot(indices, [leave_one_out_classification(dataframe, index) for index in indices])
plt.plot(indices, [leave_one_out_classification(dataframe[:100], index) for index in indices])
plt.plot(indices, [leave_one_out_classification(dataframe[features_to_use], index) for index in indices])
plt.plot(indices, [leave_one_out_classification(dataframe[features_to_use][:100], index) for index in indices])
plt.legend(['full', 'full, first 100', 'fixed', 'fixed, first 100 (assignment 120)'])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Leave One Out Cross Classification')
plt.savefig('120.png')
print('full', [leave_one_out_classification(dataframe, index) for index in indices])

