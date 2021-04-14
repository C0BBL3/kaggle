import sys
sys.path.append('src/models')
from logistic_regressor import LogisticRegressor
from linear_regressor import LinearRegressor
from random_forest import RandomForest
from decision_tree import DecisionTree
from naive_bayes_classifier import NaiveBayesClassifier
from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
from dataframe import DataFrame


path_to_datasets = 'C:/Users/colbi/VSCode/Computational Math/machine-learning/kaggle/titanic/'

filename = 'dataset_of_knowns.csv'
filepath = path_to_datasets + filename
dataframe = DataFrame.from_csv(filepath, header=True)
dataframe.apply('Survived', lambda i: i if isinstance(i, float) else float(i))
dataframe.append_columns({'indices': [index for index in range(len(dataframe))]})
dataframe_indices = dataframe['indices']
survived_people = dataframe['Survived']
dataframe.remove_columns(['PassengerId', 'Survived', 'Ticket', 'Fare', 'Cabin', 'Name', 'indices'])
dataframe.apply('Sex', lambda sex: 0 if sex == 'male' else 1)
dataframe.apply('Age', lambda i: i if isinstance(i, float) else float(i) if i != '' else 0)
dataframe.apply('Pclass', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('SibSp', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('Parch', lambda i: i if isinstance(i, float) else float(i))
dataframe.apply('Embarked', lambda s: 0 if s =='S' else 1 if s == 'C' else 2)
dataframe.append_pairwise_interactions()

testing_filename = 'unknowns_to_predict.csv'
testing_filepath = path_to_datasets + testing_filename
testing_dataframe = DataFrame.from_csv(testing_filepath, header=True)
testing_dataframe.append_columns({'indices': [index for index in range(len(testing_dataframe))]})
testing_dataframe_indices = testing_dataframe['indices']
testing_passenger_ids = testing_dataframe['PassengerId']
testing_dataframe.remove_columns(['PassengerId', 'Ticket', 'Fare', 'Cabin', 'Name', 'indices'])
testing_dataframe.apply('Sex', lambda sex: 0 if sex == 'male' else 1)
testing_dataframe.apply('Age', lambda i: i if isinstance(i, float) else float(i) if i != '' else 0)
testing_dataframe.apply('Pclass', lambda i: i if isinstance(i, float) else float(i))
testing_dataframe.apply('SibSp', lambda i: i if isinstance(i, float) else float(i))
testing_dataframe.apply('Parch', lambda i: i if isinstance(i, float) else float(i))
testing_dataframe.apply('Embarked', lambda s: 0 if s =='S' else 1 if s == 'C' else 2)
testing_dataframe.append_pairwise_interactions()

#even_dataframe = dataframe.alternating_splits()
#print(len(even_dataframe))
#even_indices = even_dataframe['indices']
#even_dataframe.remove_columns(['indices'])
#odd_dataframe = dataframe.alternating_splits(False)
#odd_indices = odd_dataframe['indices']
#odd_dataframe.remove_columns(['indices'])

def classifications(model, current_dataframe):
    classifications = []
    for index in range(len(dataframe)):
        observation = {column: array[index] for column, array in dataframe.data_dict.items() if column != 'indices'}
        if isinstance(model, NaiveBayesClassifier):
            guess = model.probability(observation)
            if guess >= 0.5:
                guess = True
            else:
                guess = False
        elif isinstance(model, LinearRegressor) or isinstance(model, LogisticRegressor):
            guess = model.predict(observation)
            if guess >= 0.5:
                guess = True
            else:
                guess = False
        elif isinstance(model, RandomForest) or isinstance(model, DecisionTree):
            guess = model.classify(observation)[0]
            if guess >= 0.5:
                guess = True
            else:
                guess = False
        else:
            guess = model.classify(observation)
        #print("survived_people[index]", survived_people[2*index])
        if survived_people[index] == 1:
            answer = True
        else:
            answer = False
        #print("observation", observation)
        #print("guess", guess)
        #print("answer", answer)
        #print("guess == answer", guess == answer)
        classifications.append(guess == answer)
    return classifications

def get_classifications(model, dataframe):
    classifications = []
    for index in range(len(dataframe)):
        observation = {column: True if array[index] >= array[len(array)//2] else False for column, array in dataframe.data_dict.items() if column != 'indices'}
        if isinstance(model, NaiveBayesClassifier):
            guess = model.classify('Survived', observation)
        elif isinstance(model, LinearRegressor) or isinstance(model, LogisticRegressor):
            guess = model.predict(observation)
        elif isinstance(model, RandomForest) or isinstance(model, DecisionTree):
            guess = model.classify(observation)[0] 
        else:
            guess = model.classify(observation)
        if guess >= 0.5:
            guess = 1
        else:
            guess = 0
        classifications.append((testing_passenger_ids[index], guess))
    return classifications

print('\n')

ratings = [[survived] for index, survived in enumerate(survived_people) if index in dataframe_indices]

dataframe.append_columns({'constant': [1 for _ in dataframe_indices]})

linear_regressor = LinearRegressor(dataframe, ratings, prediction_column='Survived')
linear_regressor.solve_coefficients()
linear_regressor_classifications = get_classifications(linear_regressor, testing_dataframe)
for row in linear_regressor_classifications:
    print(row)

print('\n')
logistic_regressor = LogisticRegressor(dataframe, ratings, prediction_column='Survived')
logistic_regressor.solve_coefficients()
logistic_regressor_classifications = get_classifications(logistic_regressor, testing_dataframe)
for row in logistic_regressor_classifications:
    print(row)

dataframe.remove_columns(['constant'])

dataframe.append_columns({'Survived': [did_survive for index, did_survive in enumerate(survived_people) if index in dataframe_indices]})

print('\n')

def boolize_data(array):
    unique_data = sorted(set(array))
    mid_point = len(unique_data) // 2
    high = unique_data[mid_point:]
    low = unique_data[:mid_point]
    fixed_array = []
    for element in array:
        if element in low:
            fixed_array.append(False)
        elif element in high:
            fixed_array.append(True)
    return fixed_array
    
naive_bayes_classifier_random_50_dataframe_1 = DataFrame.from_array([boolize_data(row) for row in dataframe.to_array()], dataframe.columns)
naive_bayes_classifier = NaiveBayesClassifier(dataframe=naive_bayes_classifier_random_50_dataframe_1, dependent_variable='Survived')
naive_bayes_classifier_classifications = get_classifications(naive_bayes_classifier, testing_dataframe)
for row in naive_bayes_classifier_classifications:
    print(row)
print('\n')

max_depth_5_decision_tree = DecisionTree(dataframe=dataframe, class_name='Survived', features=[column for column in dataframe.columns if column != 'Survived'], max_depth=5)
max_depth_5_decision_tree.fit()
max_depth_5_decision_tree_classifications = get_classifications(max_depth_5_decision_tree, testing_dataframe)
for row in max_depth_5_decision_tree_classifications:
    print(row)
print('\n')
max_depth_10_decision_tree = DecisionTree(dataframe=dataframe, class_name='Survived', features=[column for column in dataframe.columns if column != 'Survived'], max_depth=10)
max_depth_10_decision_tree.fit()
max_depth_10_decision_tree_classifications = get_classifications(max_depth_10_decision_tree, testing_dataframe)
for row in max_depth_10_decision_tree_classifications:
    print(row)
'''
print('\n')
max_depth_3_random_decision_tree = RandomForest(number_of_random_trees = 100, class_name='Survived', max_depth = 5)
max_depth_3_random_decision_tree.fit(dataframe, features=[column for column in dataframe.columns if column != 'Survived'])
max_depth_3_random_decision_tree_classifications = get_classifications(max_depth_3_random_decision_tree, testing_dataframe)
for row in max_depth_3_random_decision_tree_classifications:
    print(row)

print('\n')
max_depth_5_random_decision_tree = RandomForest(number_of_random_trees = 100, class_name='Survived', max_depth = 10)
max_depth_5_random_decision_tree.fit(dataframe, features=[column for column in dataframe.columns if column != 'Survived'])
max_depth_5_random_decision_tree_classifications = get_classifications(max_depth_5_random_decision_tree, testing_dataframe)
for row in max_depth_5_random_decision_tree_classifications:
    print(row)
print('\n')

'''
k_nearest_neighbors_classifier = KNearestNeighborsClassifier()
k_nearest_neighbors_classifier.fit(dataframe, 'Survived')
k_nearest_neighbors_classifications = classifications(k_nearest_neighbors_classifier, testing_dataframe)
print('k_nearest_neighbors_classifier accuracy', k_nearest_neighbors_classifications.count(True) / len(k_nearest_neighbors_classifications))
