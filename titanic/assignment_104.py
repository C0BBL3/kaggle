from parse_line import parse_line 
import sys
sys.path.append('src/models')
from dataframe import DataFrame

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
#dataframe_indices = dataframe['indices']
#survived_people = dataframe['Survived']
#dataframe.remove_columns(['PassengerId', 'Survived', 'Ticket', 'Fare', 'Cabin', 'Name', 'indices'])
'''
dataframe.apply('Survived', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('Sex', lambda sex: 0 if sex == 'male' else 1)
dataframe.apply('Age', lambda i: i if isinstance(i, float) else float(i) if i != '' else 0)
dataframe.apply('Pclass', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('SibSp', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('Parch', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('Embarked', lambda s: 0 if s =='S' else 1 if s == 'C' else 2)'''
dataframe.append_columns({'indices': [index for index in range(len(dataframe))]})

dataframes = []

dataframes.append(dataframe.select(['Pclass', 'Survived', 'indices']).group_by('Pclass').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Pclass', True))
dataframes.append(dataframe.select(['Sex', 'Survived', 'indices']).group_by('Sex').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Sex', True))
dataframes.append(dataframe.select(['SibSp', 'Survived', 'indices']).group_by('SibSp').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('SibSp', True))
dataframes.append(dataframe.select(['Parch', 'Survived', 'indices']).group_by('Parch').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Parch', True))
#dataframes.append(dataframe.select(['Cabin', 'Survived', 'indices']).group_by('Cabin').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Cabin', True))
#dataframes.append(dataframe.select(['Embarked', 'Survived', 'indices']).group_by('Embarked').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Embarked', True))
#dataframes.append(dataframe.select(['Age', 'Survived', 'indices']).group_by('Age').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Age', True))
#dataframes.append(dataframe.select(['Fare', 'Survived', 'indices']).group_by('Fare').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Fare', True))

print('ignore the indices thats used to calculute the number of people in each category')
for dataframe in dataframes:
    print('\n')
    print(dataframe.columns)
    for row in dataframe.to_array():
        print(row)
print('\n')

