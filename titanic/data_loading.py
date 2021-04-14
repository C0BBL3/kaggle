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

df = DataFrame.from_csv("kaggle/titanic/dataset_of_knowns.csv", data_types=data_types, parser=parse_line)

print('\nTesting...\n')

print("    Testing Dataframe Columns")
assert df.columns == ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], "Dataframe Columns was not right, it should be ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], but was {}".format(df.columns)
print("    Dataframe Columns Passed!!!\n")

print("    Testing Dataframe to_array")
assert df.to_array()[:5] == [[1, 0, 3, '"Braund, Mr. Owen Harris"', 'male', 22.0, 1, 0, 'A/5 21171', 7.25, None, 'S'],
                             [2, 1, 1, '"Cumings, Mrs. John Bradley (Florence Briggs Thayer)"', 'female', 38.0, 1, 0, 'PC 17599', 71.2833, 'C85', 'C'],
                             [3, 1, 3, '"Heikkinen, Miss. Laina"', 'female', 26.0, 0, 0, 'STON/O2. 3101282', 7.925, None, 'S'],
                             [4, 1, 1, '"Futrelle, Mrs. Jacques Heath (Lily May Peel)"', 'female', 35.0, 1, 0, '113803', 53.1, 'C123', 'S'],
                             [5, 0, 3, '"Allen, Mr. William Henry"', 'male', 35.0, 0, 0, '373450', 8.05, None, 'S']], "Dataframe to_array was not right, it should be [[1, 0, 3, 'Braund, Mr. Owen Harris', 'male', 22.0, 1, 0, 'A/5 21171', 7.25, None, 'S'], [2, 1, 1, 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'female', 38.0, 1, 0, 'PC 17599', 71.2833, 'C85', 'C'], [3, 1, 3, 'Heikkinen, Miss. Laina', 'female', 26.0, 0, 0, 'STON/O2. 3101282', 7.925, None, 'S'], [4, 1, 1, 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'female', 35.0, 1, 0, '113803', 53.1, 'C123', 'S'], [5, 0, 3, 'Allen, Mr. William Henry', 'male', 35.0, 0, 0, '373450', 8.05, None, 'S']], but was {}".format(df.to_array()[:5])
print("    Dataframe to_array Passed!!!\n")

print('ALL TESTS PASS!!!!!\n')