import scipy.optimize as opt
import numpy as np

def rms_error_cost(p, x, y, model, regularization=None, measured_offset=None):
    # Compute the RMS error between the proposed model and the data

    #The measured offset induces a penalty on the offset parameter for a 2 peak
    #lorentzian model as a difference from a measured offset value
    #TODO: This is specific to a 2 parameter lorentzian, should find a general way to solve this problem or factor it into the hanle specific code
    err = y - model(x, *p)
    if measured_offset is not None:
        return np.dot(err, err) + y.size*(p[4]-measured_offset)**2
    elif regularization is not None:
        return np.dot(err, err) + regularization*np.dot(p, p)
    else:
        return np.dot(err, err)

def global_curve_fit(model, x, y, init_p, cost_func=rms_error_cost,
                     basinhopping_kwargs=None,
                     cost_func_kwargs=None):
    """
    Global fit the parameters of a curve with a basin hopping algorithm
    :param model:
    :param x:
    :param y:
    :param init_p:
    :return:
    """

    #TODO: this is a hack specific to fitting hanle data, it makes this fitter not generalize to other problems
    cost_args = [x, y, model, None, None]

    if cost_func_kwargs is not None:
        if 'regularization' in cost_func_kwargs.keys():
            cost_args[3] = cost_func_kwargs['regularization']
        if 'measure_offset' in cost_func_kwargs.keys():
            cost_args[4] = cost_func_kwargs['measured_offset']

    if basinhopping_kwargs is None:
        basinhopping_kwargs = {}

    p = opt.basinhopping(cost_func, np.array(init_p),
                         minimizer_kwargs={'args': tuple(cost_args)},
                         **basinhopping_kwargs)
    return p

def regularized_curve_fit(model, x, y, init_p, regularization=0):
    def cost(p):
        err = y - model(x, *p)
        return np.dot(err, err) + regularization * np.dot(p, p)
    p = opt.minimize(cost, init_p)
    return (p.x, None)

def progressive_fit(data, loader, init_p, p_names, model,
                    keys, fitter=opt.curve_fit, key_name='Timestamp', x_name='Field', y_name='FR',
                    post_process=lambda a, b, c, d: None):
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
        d = loader(k)
        if len(d.index) == 0:
            return
        p, cov = fitter(model, d[x_name], d[y_name], last_p)
        for i in range(0, len(p)):
            data.loc[data[key_name] == k, p_names[i]] = p[i]
        last_p = p
        post_process(p, d[x_name], d[y_name], k)