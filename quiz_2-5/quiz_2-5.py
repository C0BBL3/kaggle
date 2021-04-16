import numpy as np
from sklearn.linear_model import LinearRegression
from parse_line import parse_line 
import sys
sys.path.append('src/models')
from dataframe import DataFrame
from logistic_regressor import LogisticRegressor
from linear_regressor import LinearRegressor

data_types = {
    "gender": str,
    "race/ethnicity": str,
    "parental level of education": str,
    "lunch": str,
    "test preparation course": str,
    "math score": int,
    "reading score": int,
    "writing score": int
}


dataframe = DataFrame.from_csv("kaggle/quiz_2-5/StudentsPerformance.csv", data_types=data_types, parser=parse_line)

dataframe.apply('gender', lambda sex: 1.0 if sex == 'male' else 0.0)
fixed_age = [element for element in dataframe['Age'] if element is not None]
race_enthnicity = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
race_enthnicity_2 = [4.0, 2.0, 0.0, 1.0, 3.0]
dataframe.apply('race/ethnicity', lambda race: race_enthnicity_2[race_enthnicity_2.index(race)])
level_of_education = ['some high school', 'high school', "associate's degree", 'some college', "bachelor's degree", "master's degree"]
level_of_education_2 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
dataframe.apply('parental level of education', lambda edu: level_of_education_2[level_of_education.index(edu)])
dataframe.apply('lunch', lambda lunch: 1.0 if lunch == 'standard' else 0.0)
dataframe.apply('test preparation course', lambda prep: 1.0 if prep == 'completed' else 0.0)

print('2-5-1', dataframe['math score'][len(dataframe['math score']) - 3:])

print('2-5-2', sum(dataframe['math score']) / len(dataframe['math score']))

math_people_who_did_prep = [score for index, score in enumerate(dataframe['math score']) if dataframe['test preparation course'][index] == 1.0]
math_people_who_didnt_prep = [score for score in dataframe['math score'] if score not in math_people_who_did_prep]
print('2-5-3\n', 'people who did the prep', sum(math_people_who_did_prep) / len(math_people_who_did_prep), '\n', 'people who didnt the prep', sum(math_people_who_didnt_prep) / len(math_people_who_didnt_prep))

print('2-5-4 there are 6 categories of parental level of education,', 'some high school', 'high school', "associate's degree", 'some college', "bachelor's degree", "master's degree")

dataframe.append_columns({'test preparation course-parental level of education': [prep * edu for prep, edu in zip(dataframe['test preparation course'], dataframe['parental level of education'])]})

dataframes = []
training_dataframe = dataframe.select_rows_where(lambda row: row['indices'] in list(range(0, len(dataframe['math score']) - 3)))
testing_dataframe = dataframe.select_rows_where(lambda row: row['indices'] in list(range(len(dataframe['math score']) - 3, len(dataframe['math score']))))

def get_classifications(model, dataframe):
    classifications = []
    for index in range(len(dataframe)):
        observation = {column: True if array[index] >= array[len(array)//2] else False for column, array in dataframe.data_dict.items() if column != 'indices'}
        guess = model.predict(observation)
        if guess >= 0.5:
            guess = 1
        else:
            guess = 0
        classifications.append((testing_dataframe[index], guess))
    return classifications

ratings = [[survived] for survived in training_dataframe['math score']]
training_dataframe.remove_columns('math score')
training_x = np.array(training_dataframe.to_array())
training_y = np.array(ratings)
coeffs = LinearRegression().fit(training_x, training_y)
print(coeffs.score(training_x, training_y))