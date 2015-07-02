from pandas import read_csv
from .file_identification import *

def data_for_value(data, value, column_name, key_name='Timestamp'):
    """
    Given a dataframe with a key column that uniquely identifies additional data files in
    the current directory, load and concatenate those files that share the same value.
    :param data: The dataframe that maps values to keys (keys are usually timestamps)
    :param value: the values to lookup in data (for example a wavelength)
    :param column_name: the column to search for values (for example the wavelength column)
    :param key_name: the name of the column that uniquely maps value to an associated measurement data file
    :return: returns a dataframe that concatenates all of the files in the directory that map to `value` via key
    """
    rs = data[data[column_name] == value]
    result = read_csv(filename_containing_string(str(int(rs.iloc[0, key_name]))))
    if rs.count().Timestamp > 1:
        for i, r in rs[1:].iterrows():
            d = read_csv(filename_containing_string(str(int(r[key_name]))))
            result = result.append(d)
    return result
