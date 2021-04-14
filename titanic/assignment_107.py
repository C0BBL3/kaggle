from parse_line import parse_line 
import sys
sys.path.append('src/models')
from dataframe import DataFrame
from logistic_regressor import LogisticRegressor
from linear_regressor import LinearRegressor

data_types = {
    "PassengerId": int,
    "Survived": int,
    "Pclass": int,
    "Name": str,
    "Sex": str,
    "Age": float,
    "SibSp": int,
    "Parch": int,
    "Ticket": str,
    "Fare": float,
    "Cabin": str,
    "Embarked": str
}


dataframe = DataFrame.from_csv("kaggle/titanic/dataset_of_knowns.csv", data_types=data_types, parser=parse_line)

dataframe.apply('Survived', lambda i: float(i))
dataframe.apply('Sex', lambda sex: 1 if sex == 'female' else 0)
fixed_age = [element for element in dataframe['Age'] if element is not None]
dataframe.apply('Age', lambda i: float(i) if i != '' and i is not None else sum(fixed_age) / len(fixed_age))
dataframe.apply('Pclass', lambda i: float(i))
dataframe.apply('SibSp', lambda i: float(i))
dataframe.apply('Parch', lambda i: float(i))
dataframe.apply('Embarked', lambda i: 0.0 if i == 'S' else 1.0 if i == 'C' else 2.0 if i == 'Q' else 0.0)
copy_df = DataFrame({key: value for key, value in dataframe.data_dict.items()}, dataframe.columns)
dataframe.append_columns({'SibSp=0': [element for element in dataframe['SibSp'] if element == 0]})
dataframe.append_columns({'SibSp>=1': [element for element in dataframe['SibSp'] if element >= 1]})
dataframe.append_columns({'Parch=0': [element for element in dataframe['Parch'] if element == 0]})
dataframe.append_columns({'Cabin=A': [element for element in dataframe['Cabin'] if element == 'A']})
dataframe.append_columns({'Cabin=B': [element for element in dataframe['Cabin'] if element == 'B']})
dataframe.append_columns({'Cabin=C': [element for element in dataframe['Cabin'] if element == 'C']})
dataframe.append_columns({'Cabin=D': [element for element in dataframe['Cabin'] if element == 'D']})
dataframe.append_columns({'Cabin=E': [element for element in dataframe['Cabin'] if element == 'E']})
dataframe.append_columns({'Cabin=F': [element for element in dataframe['Cabin'] if element == 'F']})
dataframe.append_columns({'Cabin=G': [element for element in dataframe['Cabin'] if element == 'G']})
dataframe.append_columns({'Cabin=T': [element for element in dataframe['Cabin'] if element == 'T']})
dataframe.append_columns({'Cabin=None': [element for element in dataframe['Cabin'] if element is None]})
dataframe.append_columns({'Embarked=S': [element for element in dataframe['Embarked'] if element == 'S']})
dataframe.append_columns({'Embarked=C': [element for element in dataframe['Embarked'] if element == 'C']})
dataframe.append_columns({'Embarked=Q': [element for element in dataframe['Embarked'] if element == 'Q']})
dataframe.append_columns({'Embarked=None': [element for element in dataframe['Embarked'] if element is None]})
dataframe.append_columns({'indices': [index for index in range(len(dataframe))]}) #indices for counting
dataframe.append_columns({'constant': [1 for _ in range(len(dataframe))]}) #constant
dataframe.append_columns({'SibSp': copy_df['SibSp']})
dataframe.append_columns({'Parch': copy_df['Parch']})
dataframe.append_columns({'Embarked': copy_df['Embarked']})
dataframe.remove_columns(['PassengerId', 'Ticket', 'Cabin', 'Name', 'Embarked'])

dataframes = []
training_dataframe = dataframe.select_rows_where(lambda row: row['indices'] in list(range(0, 500)))
testing_dataframe = dataframe.select_rows_where(lambda row: row['indices'] in list(range(500, len(dataframe))))

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

'''
dataframes.append(dataframe.select(['Sex', 'Survived', 'indices']).group_by('Sex').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Sex', 'ASC']]))
dataframes.append(dataframe.select(['Age', 'Survived', 'indices']).group_by('Age').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Age', 'ASC']]))
dataframes.append(dataframe.select(['SibSp', 'Survived', 'indices']).group_by('SibSp').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['SibSp', 'ASC']]))
dataframes.append(dataframe.select(['Parch', 'Survived', 'indices']).group_by('Parch').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Parch', 'ASC']]))
dataframes.append(dataframe.select(['Cabin', 'Survived', 'indices']).group_by('Cabin').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Cabin', 'ASC']])) 
dataframes.append(dataframe.select(['Embarked', 'Survived', 'indices']).group_by('Embarked').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Embarked', 'ASC']]))
dataframes.append(dataframe.select(['Pclass', 'Survived', 'indices']).group_by('Pclass').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Pclass', 'ASC']]))
dataframes.append(dataframe.select(['Fare', 'Survived', 'indices']).group_by('Fare').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Fare', 'ASC']]))
'''

sex_training_dataframe = training_dataframe.select_columns(['Sex'])
sex_pclass_training_dataframe = training_dataframe.select_columns(['Sex', 'Pclass', 'constant'])
sex_pclass_dot_dot_dot_training_dataframe = training_dataframe.select_columns(['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp=0', 'Parch=0', 'constant'])



ratings = [[survived] for survived in sex_training_dataframe['Survived']]
linear_regressor = LinearRegressor(sex_training_dataframe, ratings, prediction_column='Survived')
linear_regressor.solve_coefficients()
linear_regressor_classifications = get_classifications(linear_regressor, testing_dataframe)
for row in linear_regressor_classifications:
    print(row)

print('\n')

print('\nignore the indices thats used to calculute the number of people in each category\n')

for dataframe in dataframes:
    print('\n')
    print(dataframe.columns)
    for row in dataframe.to_array():
        print(row)
print('\n')