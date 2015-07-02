import matplotlib.pyplot as plt
import scipy.optimize as opt
from .models import double_lorentzian_cenered, lorentzian

def double_lorentzian_fig(p, xs, ys, key, title):
    ps, conv = opt.curve_fit(lorentzian, xs, ys, [p[0], p[1], 0, p[4]], maxfev=4000)
    plt.ioff()
    f2, ax2 = plt.subplots()
    ax2.plot(xs, ys, '.k', rasterized=True)
    ax2.plot(xs, [double_lorentzian_centered(x, *p) for x in xs], linewidth=3)
    ax2.plot(xs, [lorentzian(x, *[p[0], p[1], 0, p[4]]) for x in xs], linewidth=3, label='peak 1')
    ax2.plot(xs, [lorentzian(x, *[p[2], p[3], 0, p[4]]) for x in xs], linewidth=3, label='peak 2')
    ax2.plot(xs, [lorentzian(x, *ps) for x in xs], linewidth=3, label='single')
    plt.legend()
    ax2.set_yticklabels(ax2.get_yticks()/1e-6)
    ax2.set_ylabel('Faraday Rotation ($\mu rad$)')
    ax2.set_xlabel('Field (Gauss)')
    ax2.set_title(title)
    plt.ion()
    return f2