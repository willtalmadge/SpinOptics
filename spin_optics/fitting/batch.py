import scipy.optimize as opt

def progressive_fit(data, loader, init_p, p_names, model,
                    keys, key_name='Timestamp', x_name='Field', y_name='FR', 
                    post_process=lambda: None):
    """
    Fits a `model` to data files returned by loader, mapped by keys in data. This batch
    procedure fits in the order the keys appear in `keys` using the fitting solution
    from the previous fit as the input to the next fit. The parameters are placed into `data`.

    :param data: A dataframe with keys that uniquely identify datasources produced by `loader`
    :param loader: a function that takes a key from `data` and returns a dataframe that will be fit to `model`
    :param init_p: the initial parameter list for the fitting progression
    :param p_names: the names of the parameters as they should appear in `data` after the fit progression
    :param model: the model to fit to the dataframes produced by loader
    :param keys: the keys that will be fed to `loader` and fit in the order they are provided
    :param key_name: the column name in `data` of the key that maps to the files via loader
    :param x_name: the column name of the x values in the dataframe produced by `loader`
    :param y_name: the column name of the y values in the dataframe produced by `loader`
    :param post_process: a function that takes the fit parameters, x values, y values and
    the key and performs some post processing step, like storing plots
    :return: None
    """
    last_p = init_p
    for k in keys:
        print(k)
        d = loader(k)
        if len(d.index) == 0:
            return
        print(last_p)
        p, cov = opt.curve_fit(model, d[x_name], d[y_name], last_p, maxfev=10000)
        print(p)
        for i in range(0, len(p)):
            data.loc[data[key_name] == k, p_names[i]] = p[i]
        last_p = p
        post_process(p, d[x_name], d[y_name], k)