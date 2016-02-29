import numpy as np

def drop_rows_with_outliers(df, columns, sigma=3):
    """
    For removing entire rows from a dataframe that may contain anomalous values. Rejects, values more than
    "sigma" from the mean of the specified columns.
    :param df: the dataframe to filter
    :param columns: the columns to check for outliers
    :param sigma: the half width of the acceptance band in units of standard deviations
    :return:
    """
    selection = np.full(len(df.index), True, dtype=np.dtype('bool'))
    if not isinstance(columns, list):
        columns = [columns]
    for var in columns:
        std_var = np.std(df[var])
        mean_var = np.mean(df[var])
        in_range = np.logical_and(df[var] > mean_var - sigma*std_var,
                                  df[var] < mean_var + sigma*std_var)
        selection = np.logical_and(selection, in_range)
    return df[selection]