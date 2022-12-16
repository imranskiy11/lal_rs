from functools import reduce
import numpy as np
import random
import pandas as pd

def row_to_list(row_string: str, drop_chars= ["]", "[", "'"]) -> list:
    """
    transform string to list
    example input: "[1,2,4,4]" -> return ['1', '2', '4', '4'] 
    """
    for ch in drop_chars:
        row_string = row_string.replace(ch, '')
    return [x.strip() for x in row_string.split(',')]

# family status 
def is_married(row: str) -> int:
    """"  return 1 if married and 0 if not marrind """
    if row == '[]':
        return None
    elif '6' in row:
        return 1
    return 0

# eployment type
def employ_replace_by_maping(string_row: str, ) -> str:
    """ multiple replacing by mapping dict """
    map_dict = {
        '1': 'employed',
        '2': 'pensioner',
        '3': 'student', 
        '4': 'pupil',
        '[]': '[unknown_emp]'
        }
    return reduce(lambda a, kv: a.replace(*kv), map_dict.items(), string_row)

def remove_space_cls(np_arr):
    """ strip from mlb classes """
    return np.array([cls.strip() for cls in np_arr])


#####################################################################
def get_columns_names_with_NA_values(dataframe):
    """ get columns list with cols with NA values """
    return list(dataframe.columns[dataframe.isnull().any()])

def get_distrib(data, norm=False):
    """ build destribution value from dataframe column """
    distrib = data.value_counts(normalize=norm)
    return list(distrib.index), list(distrib) # values and counts


def get_fill_list(dataframe, col_name):
    """ get list with values from distribution """
    # get distribution
    distr_values, distr_counts = get_distrib(dataframe[col_name])
    # get column dim with na values
    column_nan_dim = dataframe[dataframe[col_name].isnull()].shape[0]
    return random.choices(distr_values, weights=distr_counts, k=column_nan_dim)


def list_to_string(l: list) -> str:
    return f"[{', '.join(str(e) for e in l)}]"

