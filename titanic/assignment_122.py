import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from calc_accuracies import leave_one_out_classification_v2
import time
start1 = time.time()

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

dataframe = pd.read_csv(os.getcwd() + '\\titanic\\processed_dataset_of_knowns.csv')
features_to_use = ["Survived", "Sex", "Pclass", "Fare", "Age", "SibSp"]
dataframe = dataframe[features_to_use][:100]

indices = list(range(1, 100, 2))
prediction_column = 'Survived'

plt.plot(indices, [leave_one_out_classification_v2(dataframe, index, prediction_column) for index in indices])

simple_scaling_dict = {}
for column in dataframe.columns:
    simple_scaling_dict[column] = list(dataframe[column].apply(lambda element: element / max(dataframe[column])))

simple_scaling_dataframe = pd.DataFrame(simple_scaling_dict)
simple_scaling_dataframe[prediction_column] = simple_scaling_dataframe[prediction_column].apply(lambda element: int(element)) 

plt.plot(indices, [leave_one_out_classification_v2(simple_scaling_dataframe, index, prediction_column) for index in indices])

min_max_dict = {}
for column in dataframe.columns:
    min_max_dict[column] = list(dataframe[column].apply(lambda element: (element -  min(dataframe[column])) / (max(dataframe[column]) -  min(dataframe[column]))))

min_max_dataframe = pd.DataFrame(min_max_dict)
min_max_dataframe[prediction_column] = min_max_dataframe[prediction_column].apply(lambda element: int(element)) 

plt.plot(indices, [leave_one_out_classification_v2(min_max_dataframe, index, prediction_column) for index in indices])

def get_standard_deviation(row):
    avg = sum(row) / len(row)
    var = sum([((x - avg) ** 2) for x in row]) / len(row)
    return var ** 0.5

z_scoring_dict = {}
for column in dataframe.columns:
    z_scoring_dict[column] = list(dataframe[column].apply(lambda element: (element -  (sum(dataframe[column]) / len(dataframe[column]))) / get_standard_deviation(dataframe[column])))

z_scoring_dataframe = pd.DataFrame(z_scoring_dict)
z_scoring_dataframe[prediction_column] = z_scoring_dataframe[prediction_column].apply(lambda element: int(element)) 

plt.plot(indices, [leave_one_out_classification_v2(z_scoring_dataframe, index, prediction_column) for index in indices])

plt.legend(['unnormalized', 'simple scaling', 'min-max', 'z-scoring'])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Leave One Out Cross Classification')
plt.savefig('122.png')

end1 = time.time()
print('total time taken', end1 - start1)
start2 = time.time()
counter = 0
for _ in range(1000000):
    counter += 1
end2 = time.time()
print("reletive to justin's computer (it makes it worse)", 0.15 / (end2 - start2) * (end1 - start1))