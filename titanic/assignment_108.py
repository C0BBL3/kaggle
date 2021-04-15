import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv(os.getcwd() + '/titanic/dataset_of_knowns.csv')
keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
dataframe = dataframe[keep_cols]
dataframe['Sex'] = dataframe['Sex'].apply(lambda sex: 1 if sex == 'female' else 0)
age_nan = dataframe['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = dataframe['Age'].apply(lambda entry: not np.isnan(entry))
dataframe.loc[age_nan, ['Age']] = dataframe['Age'][age_not_nan].mean()
dataframe['SibSp>0'] = dataframe['SibSp'].apply(lambda sibsp: 1 if sibsp > 0 else 0)
dataframe['Parch>0'] = dataframe['Parch'].apply(lambda parch: 1 if parch > 0 else 0)
dataframe['Cabin']= dataframe['Cabin'].fillna('None')
dataframe['CabinType'] = dataframe['Cabin'].apply(lambda cabin: cabin[0] if cabin != 'None' else cabin)
for cabin_type in dataframe['CabinType'].unique():
    dummy_variable_name = 'CabinType={}'.format(cabin_type)
    dummy_variable_values = dataframe['CabinType'].apply(lambda entry: int(entry == cabin_type))
    dataframe[dummy_variable_name] = dummy_variable_values
del dataframe['CabinType']
dataframe['Embarked'] = dataframe['Embarked'].fillna('None')
for embarked_type in dataframe['Embarked'].unique():
    dummy_variable_name = 'Embarked={}'.format(embarked_type)
    dummy_variable_values = dataframe['Embarked'].apply(lambda entry: int(entry == embarked_type))
    dataframe[dummy_variable_name] = dummy_variable_values
del dataframe['Embarked']

ratings = dataframe['Survived']
features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
dataframe = dataframe[features_to_use]
print("\n", dataframe)
training_ratings = ratings[:500]
testing_ratings = ratings[500:]
training_dataframe = dataframe[:500]
testing_dataframe = dataframe[500:]
training_x = np.array(training_dataframe)
training_y = np.array(training_ratings)

coeffs = LinearRegression().fit(training_x, training_y)
print("Constant", coeffs.intercept_)
final_results = {column: coefficient for column, coefficient in zip(training_dataframe.columns, coeffs.coef_)}
for col,value in final_results.items():
    print(col,value)
print("\n")

predictions = coeffs.predict(testing_dataframe)
fixed_predictions = [1 if output > 0.5 else 0 for output in predictions]
result = [prediction == actual for prediction, actual in zip(predictions, testing_dataframe['Survived'])].count(True) / len(fixed_predictions)
print("accuracy", result)

 

