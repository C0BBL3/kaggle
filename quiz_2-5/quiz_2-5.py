import numpy as np
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('dataframe')
from parse_line import parse_line 
from dataframe import DataFrame
sys.path.pop(-1)

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


dataframe = DataFrame.from_csv("quiz_2-5/StudentsPerformance.csv", data_types=data_types, parser=parse_line)


dataframe.apply('gender', lambda sex: 1.0 if sex == 'male' else 0.0)
race_enthnicity = ['group A', 'group B', 'group C', 'group D', 'group E']
race_enthnicity_2 = [4.0, 2.0, 0.0, 1.0, 3.0]
dataframe.apply('race/ethnicity', lambda race: race_enthnicity_2[race_enthnicity.index(race)])
level_of_education = ['some high school', 'high school', 'associates degree', 'some college', 'bachelors degree', 'masters degree']
level_of_education_2 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
dataframe.apply('parental level of education', lambda edu: level_of_education_2[level_of_education.index(edu)])
dataframe.apply('lunch', lambda lunch: 1.0 if lunch == 'standard' else 0.0)
dataframe.apply('test preparation course', lambda prep: 1.0 if prep == 'completed' else 0.0)

print('\n2-5-1', dataframe['math score'][len(dataframe['math score']) - 3:])

print('\n2-5-2', sum(dataframe['math score']) / len(dataframe['math score']))

math_people_who_did_prep = [score for index, score in enumerate(dataframe['math score']) if dataframe['test preparation course'][index] == 1.0]
math_people_who_didnt_prep = [score for score in dataframe['math score'] if score not in math_people_who_did_prep]
print('\n2-5-3\n\n', 'people who did the prep', sum(math_people_who_did_prep) / len(math_people_who_did_prep), '\n', 'people who didnt the prep', sum(math_people_who_didnt_prep) / len(math_people_who_didnt_prep))

print('\n2-5-4 there are 6 categories of parental level of education,', 'some high school', 'high school', 'associates degree', 'some college', 'bachelors degree', 'masters degree')

dataframe.append_columns({'test preparation course-parental level of education': [prep * edu for prep, edu in zip(dataframe['test preparation course'], dataframe['parental level of education'])]})

dataframe.append_columns({'indices': [index for index in range(len(dataframe))]})

dataframes = []
training_dataframe = dataframe.select_rows_where(lambda row: row['indices'] in list(range(0, len(dataframe['indices']) - 3)))
ratings = [maths for maths in training_dataframe['math score']]
features_to_use = [col for col in training_dataframe.columns if col not in ['indices', 'math score']]
training_dataframe.filter_columns(features_to_use)

testing_dataframe = dataframe.select_rows_where(lambda row: row['indices'] in list(range(len(dataframe['indices']) - 3, len(dataframe['indices']))))
testing_math_scores = testing_dataframe['math score']
features_to_use = [col for col in testing_dataframe.columns if col not in ['indices', 'math score']]
testing_dataframe.filter_columns(features_to_use)

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


training_x = np.array(training_dataframe.to_array())
testing_x = np.array(testing_dataframe.to_array())
training_y = np.array(ratings)
coeffs = LinearRegression().fit(training_x, training_y)

predictions = coeffs.predict(testing_x)
fixed_predictions = [1 if output >= 0.5 else 0 for output in predictions]
result = [actual - 2 < prediction < actual + 2 for prediction, actual in zip(predictions, testing_math_scores)].count(True) / len(fixed_predictions)
print("\n2-5-5 accuracy", result)


