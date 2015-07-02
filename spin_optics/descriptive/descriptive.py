from numpy import polyfit, poly1d

def interp_peak(xs, ys):
    """
    Interpolates the given xs and ys with a parabola and calculates its extrema.
    Useful for identifying a peak feature in a dataset.
    :param xs: x values of the data points
    :param ys: y values of the data points
    :return: a tuple of the location of the parabola extrema and the parabola fit at that location
    """
    fit = polyfit(xs, ys, 2)
    p = poly1d(fit)
    a = fit[0]
    b = fit[1]
    c = fit[2]
    return (-b/(2*a), p)