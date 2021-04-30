import numpy as np
import pandas as pd
import os

dataframe = pd.read_csv(os.getcwd() + '/quiz_2-6/StudentEnrollment.csv')

print('\n2-6-2-A', sum(dataframe['training_hours']) / len(dataframe['training_hours']))

print('\n2-6-2-B', list(dataframe['target']).count(1) / len(dataframe['target']))

print('\n2-6-2-C', max(dataframe['city'].unique(), key=lambda city: len(dataframe.loc[dataframe['city'] == city])))

print('\n2-6-2-D', len(dataframe.loc[dataframe['city'] == 'city_103']))

print('\n2-6-2-E', max(dataframe['city'].unique(), key=lambda city: int(city.split('_')[1])))

print('\n2-6-2-F', len(dataframe.loc[dataframe['company_size'] == '<10']))

print('\n2-6-2-G', len(dataframe.loc[(dataframe['company_size'] == '<10') | (dataframe['company_size'] == '10/49') | (dataframe['company_size'] == '50-99')]))

print()