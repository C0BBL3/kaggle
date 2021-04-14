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

#dataframes.append(dataframe.select(['Pclass', 'Survived', 'indices']).group_by('Pclass').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Pclass', True))
#dataframes.append(dataframe.select(['Sex', 'Survived', 'indices']).group_by('Sex').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Sex', True))
#dataframes.append(dataframe.select(['SibSp', 'Survived', 'indices']).group_by('SibSp').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('SibSp', True))
#dataframes.append(dataframe.select(['Parch', 'Survived', 'indices']).group_by('Parch').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Parch', True))
#dataframes.append(dataframe.select(['Cabin', 'Survived', 'indices']).group_by('Cabin').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Cabin', True))
#dataframes.append(dataframe.select(['Embarked', 'Survived', 'indices']).group_by('Embarked').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by('Embarked', True))
df1 = dataframe.select(['Age', 'Survived', 'indices']).group_by('Age').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Age', 'ASC']])
df2 = dataframe.select(['Fare', 'Survived', 'indices']).group_by('Fare').aggregate('indices', 'count').aggregate('Survived', 'avg').order_by([['Fare', 'ASC']])

print('\nignore the indices thats used to calculute the number of people in each category\n')
print(df1.columns)
new_array_1 = []
for ten in range(int(max(df1.to_array(), key=lambda x: x[0])[0] / 10)):
    new_array_1.append(df1.select_rows_where(lambda row: 10 * ten < row['Age'] <= 10 * ten + 10).to_array())

final_array = []
for ten, inner_dataframe in enumerate(new_array_1):
    row = [(10 * ten, 10 * ten + 10)]
    temp = [row[1] for row in inner_dataframe]
    try:
        row.append(sum(temp) / len(temp))
    except:
        row.append(0)
    temp = [row[2] for row in inner_dataframe]
    row.append(sum(temp))
    print(row)

print('\n')

print(df2.columns)
new_array_2 = []
range_ = [0, 5, 10, 20, 50, 100, 200, int(max(df1.to_array(), key=lambda x: x[0])[0] + 1)]
for index, ran in enumerate(range_[:-1]):
    new_array_2.append(df2.select_rows_where(lambda row: ran < row['Fare'] <= range_[index + 1]).to_array())

final_array = []
for index, inner_dataframe in enumerate(new_array_2):
    if range_[index + 1] != range_[-1]:
        row = [(range_[index], range_[index + 1])]
    else:
        row = [(range_[index], 'infinity')]
    temp = [row[1] for row in inner_dataframe]
    try:
        row.append(sum(temp) / len(temp))
    except:
        row.append(0)
    temp = [row[2] for row in inner_dataframe]
    row.append(sum(temp))
    print(row)

print()

