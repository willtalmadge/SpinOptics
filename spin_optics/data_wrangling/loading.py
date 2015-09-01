from pandas import read_csv

from spin_optics.data_wrangling.file_identification import *


def data_for_value(data, value, column_name, loader, key_name='Timestamp'):
    """
    Given a dataframe with a key column that uniquely identifies additional data files in
    the current directory, load and concatenate those files that share the same value.
    :param data: The dataframe that maps values to keys (keys are usually timestamps)
    :param value: the values to lookup in data (for example a wavelength)
    :param column_name: the column to search for values (for example the wavelength column)
    :param loader: a function that produces a data frame for the given key_name
    :param key_name: the name of the column that uniquely maps value to an associated measurement data file
    :return: returns a dataframe that concatenates all of the files in the directory that map to `value` via key
    """
    matching_rows = data[data[column_name] == value]
    result = loader(matching_rows.iloc[0, key_name])
    if len(matching_rows.index) > 1:
        for i, r in matching_rows[1:].iterrows():
            d = loader(r[key_name])
            result = result.append(d)
    return result
