from matrix import Matrix
import os
import difflib
import filecmp
import math
import random
from matrix import Matrix

class DataFrame:
    def __init__(self, data_dict, column_order, indices=False):
        self.data_dict = {}
        self.columns = column_order
        if data_dict != {}:
            for key in self.columns:
                self.data_dict[key] = data_dict[key]
        else:
            self.data_dict = {}
        if 'indices' not in self.columns and indices:
             self.data_dict['indices'] = [index for index,_ in enumerate(self.data_dict[self.columns[0]])]
        self.last_base_column_index = len(column_order)
        self.sql_operations = {
            'SELECT': lambda columns: self.select(columns),
            'WHERE': lambda lambda_function: self.where(lambda_function),
            'ORDER': lambda order: self.order_by(order),
            'GROUP': lambda column: self.group_by(column),
            'AGGREGATE': lambda column, relation: self.aggregate(column, relation)
            #'JOIN': lambda df, ON_lambda_function: self.join(df, ON_lambda_function),
            #'FROM': lambda tables: self.from(tables)
        }
        self.sql_relations = ['IN', 'BY', 'ON', 'ASC', 'DESC']

    def rename_columns(self, new_column_names):
        try:
            for new_name, old_name in zip(new_column_names, self.columns):
                if new_name != old_name:
                    self.data_dict[new_name] = self.data_dict[old_name]
                    del self.data_dict[old_name]
            self.columns = new_column_names
        except:
            print('Please add more new column names or filter old columns out.')
            exit()
            
    def __getitem__(self, key):
        return self.data_dict[key]
        
    def __len__(self):
        return len(self.data_dict[self.columns[0]])

    def to_array(self):
        try:
            array = []
            rows = [self.data_dict[col] for col in self.columns]
            for row in zip(*rows):
                array.append(list(row))
            return array
        except:
            print('There were some mismatched values please re-evaluate your dataframe class')
            exit()

    @staticmethod
    def from_array(array, columns):
        dictionary = {}
        transposed_array = Matrix(array).transpose().elements
        for column, arr in zip(columns, transposed_array):
            dictionary[column] = list(arr)
        return DataFrame(dictionary, columns)

    @classmethod
    def from_csv(self, filepath, data_types={}, parser=None):
        array = []
        with open(filepath, "r") as file:
            for index, line in enumerate(file.read().split('\n')):
                if index == 0:
                    split_line = line.split(',')
                    fixed_split_line = []
                    for entry in split_line:
                        fixed_entry = ''
                        for element in list(entry):
                            if element not in ["'", '"']:
                                fixed_entry += element

                        fixed_split_line.append(fixed_entry)
                    array.append(fixed_split_line)
                else:
                    parsed_line = parser(line)
                    if len(parsed_line) == len(data_types.values()):
                        entries = []
                        for entry, type_ in zip(parsed_line, data_types.values()):
                            if entry == '':
                                entries.append(None)
                            else:
                                fixed_entry = ''
                                for element in list(entry):
                                    if element not in ["'", '"']:
                                        fixed_entry += element
                                entries.append(type_(fixed_entry))
                                    
                        array.append(entries)
                    else:
                        print('Please give a new dictionary of datatypes or a new parser, there is an difference in length between the dictionary of datatypes and the parsed line')
                        exit()
        return DataFrame.from_array(array[1:], array[0])

    def query(self, query):
        fixed_query = self.sql_parser(query) #def gonna need a update when we add more sql stuff for example assignment 103-2 wont work because of multiple table depth
        current_dataframe = self
        if 'ORDER' in fixed_query.keys():
            current_dataframe = self.sql_operations['ORDER'](fixed_query['ORDER'])
        for operation, inputs in fixed_query.items():
            if operation != 'ORDER':
                current_dataframe = current_dataframe.sql_operations[operation](*inputs)
        return current_dataframe

    def sql_parser(self, query):
        semi_fixed_query = [entry[:-1] if entry[-1] == ',' else entry for entry in query.split(' ')]
        fixed_query = {}
        index = 0
        num_of_operative_words = 0
        while index < len(semi_fixed_query) - num_of_operative_words:
            entry = semi_fixed_query[index]
            if entry in self.sql_operations.keys():
                fixed_query[entry] = [[]]
                num_of_operative_words += 1
                #fixed_query[entry].append([])
                entry_array_index = 0
                for word_index, word in enumerate(semi_fixed_query[index + 1:]):
                    if word not in self.sql_relations and word not in self.sql_operations.keys():
                        fixed_query[entry][entry_array_index].append(word)
                        #index += 1
                    elif word in self.sql_relations and word not in self.sql_operations.keys():
                        if word in ['ASC', 'DESC']:
                            fixed_query[entry][entry_array_index].append(word)
                            index += 1
                        if word != 'BY' and semi_fixed_query.index(word) < len(semi_fixed_query) - 1:
                            fixed_query[entry].append([])
                            entry_array_index += 1
                        continue
                    elif word in self.sql_operations.keys():
                        index += word_index
                        break
            else:
                index += 1
                continue

        return fixed_query

    def select(self, columns):
        return self.select_columns(columns)

    def where(self, lambda_function):
        return self.select_rows_where(lambda_function)

    def order_by(self, order):
        return self.from_array(sorted(self.to_array(), key = lambda row: self.get_key(order, row)), self.columns)
    
    def get_key(self, order, row):
        temp = []
        for col, ascending in order:
            current_column = row[self.columns.index(col)]
            converted_string = self.convert_string(current_column)
            if ascending == 'ASC':
                temp.append(converted_string)
            else:
                temp.append(converted_string[::-1])
        return temp

    def convert_string(self, string):
        if isinstance(string, str):
            string = list(string.lower())
            fixed_string = []
            for letter in string:
                if letter not in fixed_string:
                    fixed_string.append(letter)
            abcs = 'abcdefghijklmnopqrstuvwxyz'
            return [abcs.index(letter) + 1 for letter in fixed_string]
        else:
            return string

    def group_by(self, column):
        new_dict = {}
        none_indices = []
        for row_index, row_key in enumerate(self.data_dict[column]):
            if row_key is not None:
                if row_key not in new_dict.keys():
                    new_dict[row_key] = [[] for _ in range(len(self.data_dict.keys()) - 1)]
                fixed_items = {key: value for key, value in self.data_dict.items() if key != column}
                for index, value in enumerate(fixed_items.values()):
                    for element_index, element in enumerate(value):
                        if row_index == element_index:
                            new_dict[row_key][index].append(element)
        new_array = [[key] if key is not None else [sum(list(new_dict.keys())) / len(list(new_dict.keys()))] for key in new_dict.keys()] #if the key is none then give it the average age
        for key_index, value in enumerate(new_dict.values()):
            if key_index not in none_indices:
                for element in value:
                    new_array[key_index].append(element)
        return self.from_array(new_array, self.columns)

    def aggregate(self, column, relation):
        relations = {
            'count': lambda x: len(x),
            'max': lambda x: max(x),
            'min': lambda x: min(x),
            'sum': lambda x: sum(x),
            'avg': lambda x: sum(x) / len(x)
        }
        copy_dict = {key: value for key, value in self.data_dict.items()}
        new_dict = {key: value for key, value in self.data_dict.items()}
        for index, value in enumerate(copy_dict[column]):
            new_dict[column][index] = relations[relation](value)
        return DataFrame(new_dict, self.columns)
    
    def generate_new_column(self, column, new_column, function):
        dictionary = {key: value for key, value in self.data_dict.items()}
        columns = [col for col in self.columns]
        columns += [new_column]
        dictionary[new_column] = [function(x) for x in dictionary[column]]
        return DataFrame(dictionary, columns)

    def split_dataframe(self, number_of_sections, split_index_s, location_of_splits = None): #if number_of_sections is 5 then there will be 5 even sections containing about 20% of the dataset each, perfect for assignment 77. split index is for example out of 5 sections or 4 splits, index 0 is section 1, index 1 is section 2, etc. split ratios are in ratio form of the list
        if location_of_splits is None:
            splits = [(index * math.ceil(len(self)/(number_of_sections)) , ((index+1) * math.ceil(len(self)/(number_of_sections)))) for index in range(0,number_of_sections)]
        elif len(location_of_splits) == number_of_sections - 1:
            fixed_split_ratios = self.fix_split_locations(location_of_splits)
            splits = [(int(split[0] * len(self)), int(split[1] * len(self))) for split in fixed_split_ratios]
        if type(split_index_s) == tuple:
            return DataFrame({key: self.combine_lists([value[splits[split][0]:splits[split][1]] for split in split_index_s]) for key, value in self.data_dict.items()}, self.columns)
        else: 
            return DataFrame({key: value[splits[split_index_s][0]:splits[split_index_s][1]] for key, value in self.data_dict.items()}, self.columns)

    def fix_split_locations(self, location_of_splits):
        fixed_splits = []
        for index_of_split, split in enumerate(location_of_splits):
            if index_of_split == 0:
                fixed_splits.append((0, split))
            if len(location_of_splits) > 1 and index_of_split > 0:
                fixed_splits.append((location_of_splits[index_of_split - 1], split))
            if index_of_split == location_of_splits.index(location_of_splits[-1]):
                fixed_splits.append((split, 1))
        return fixed_splits

    def randomly_select_data(self, split_percentage):
        indices_of_new_data = []
        final_len_of_new_data = math.floor(split_percentage * len(self))
        while len(indices_of_new_data) < final_len_of_new_data:
            indices_of_new_data.append(random.choice(self.data_dict['indices']))
        return self.select_rows(indices_of_new_data)

    def alternating_splits(self, even = True):
        if even == True:
            return DataFrame({column: [element for index, element in enumerate(elements) if index % 2 == 0] for column, elements in self.data_dict.items()}, self.columns)
        else:
            return DataFrame({column: [element for index, element in enumerate(elements) if index % 2 != 0] for column, elements in self.data_dict.items()}, self.columns)

    def filter_columns(self, columns):
        self.data_dict = DataFrame(self.data_dict, columns, indices='indices' in columns).data_dict
        self.columns = [key for key, _ in self.data_dict.items()]

    def select_columns(self, columns):
        return DataFrame(self.data_dict, columns)

    def select_rows(self, indices):
        return self.from_array([DataFrame(self.data_dict, self.columns).to_array()[i] for i in indices], self.columns)

    def select_rows_where(self, lambda_function):
        rows = []
        for i in range(len(self.to_array())):
            x = {column: value[i] for column, value in self.data_dict.items()}
            if lambda_function(x):
                rows.append(i)
        if rows != []:
            return self.select_rows(rows)
        else:
            return DataFrame({column: [] for column in self.columns}, self.columns)

    def get_sorted_indicies(self, column, ascending, order_by = True):
        return [self.to_array().index(row) for row in sorted(self.data_dict[column], reverse = ascending)]

    def apply(self, column, function):
        self.data_dict[column] = [function(value) for value in self.data_dict[column]]

    def append_pairwise_interactions(self):
        new_cols, new_cols_in_matrix, length_of_old_cols = [], [], len(self.columns)
        fixed_columns = [column for column in self.columns if column != 'indices']
        cartesian = self.cartesian_product([fixed_columns, fixed_columns])
        for keys in cartesian:
            new_cols.append('_'.join(keys))
            new_cols_in_matrix.append([value_1 * value_2 for value_1, value_2 in zip(self.data_dict[keys[0]], self.data_dict[keys[1]])])
        self.columns += new_cols
        for i, col in enumerate(new_cols_in_matrix):
            self.data_dict[self.columns[i + length_of_old_cols]] = col
        self.array = self.to_array()

    def next_set_of_combos(self, current_arr, next_arr):
        result = []
        for col_1 in current_arr:
            for col_2 in next_arr:
                if [col_1, col_2] not in result and [col_2, col_1] not in result and col_1 != col_2: result.append([col_1, col_2])
        return result

    def cartesian_product(self, arrays):
        result, current_arr = [arrays[0]], arrays[0]
        for arr in arrays[1:]:
            current_arr = self.next_set_of_combos(current_arr, arr)
            result.append(current_arr)
        return result[-1]

    def append_columns(self, dictionary):
        for key, values in dictionary.items():
            self.columns.append(key)
            self.data_dict[key] = values

    def remove_columns(self, columns):
        self.columns = [col for col, _ in self.data_dict.items() if col not in columns]
        self.filter_columns(self.columns)

    def create_dummy_variables(self, dumb_column):
        new_cols, rows = self.get_unique_dummy_columns(dumb_column)
        self.remove_columns(dumb_column)
        self.append_columns({col: [row[i] for row in rows] for i, col in enumerate(new_cols)})

    def get_unique_dummy_columns(self, dumb_column):
        new_cols, rows = [], []
        if isinstance(self.data_dict[dumb_column][0], list):
            longest_list = max(dumb_column, key=lambda column: len(column))
            for list_1 in longest_list:
                if list_1 != [] and list_1 not in new_cols:
                    new_cols.append(list_1)
            for list_2 in self.data_dict[dumb_column]:
                rows.append([1 if value in list_2 else 0 for value in longest_list])
            return new_cols, rows
        else:
            for value_1 in self.data_dict[dumb_column]:
                if str(dumb_column) + '-' + str(value_1) not in new_cols:
                    new_cols.append(str(dumb_column) + '-' + str(value_1))
                    rows.append([1 if value_1 == value_2 else 0 for value_2 in self.data_dict[dumb_column]])
            return new_cols, rows

    @staticmethod
    def combine_lists(lists):
        final_list = []
        for arr in lists:
            final_list += arr
        return final_list
