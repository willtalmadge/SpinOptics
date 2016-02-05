from spin_optics import ureg
import pint
from inspect import signature

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

def parse_units(data: dict) -> dict:
    """
    Take a dictionary as input and try to parse each string field into a pint value with units.

    :param data: A dictionary that may have strings with dimensional values ie '1.2 volts'
    :return: A dictionary with all parsable fields converted to pint units
    """

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            try:
                result[key] = ureg(value)
            except pint.UndefinedUnitError:
                result[key] = value
        else:
            result[key] = value

    return result


def expand_kwargs(func, values):
    """
    Given a function and dictionary, select a subset from the dictionary of values that will be accepted
    as kwargs by func.

    For this to have any effect, func must have arg names that identically match keys in values.
    If values does not supply all the args to func, the rest must be provided at the call site.

    :param func:
    :param values:
    :return:
    """
    args = set(signature(func).parameters.keys())
    keys = set(values.keys())
    passable_keys = args & keys

    return {key:value for key,value in values.items() if key in passable_keys}