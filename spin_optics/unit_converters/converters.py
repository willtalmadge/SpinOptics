def hanle_lifetime_gauss_in_sec(k, g=1.0):
    """
    Calculates the lifetime in seconds from the coefficient in front of the field
    :param k: the lifetime coefficient extracted from a Lorentzian fit in units of 1/Gauss
    :param g: the g factor of the system under study
    :return: returns the lifetime of the system in seconds
    """
    return 10000*1.05457173e-34 * k /(g* 9.27400968e-24)

def energy_ev(wavelength_nm):
    return 6.62606957e-34*3e8/(1e-9*wavelength_nm*1.602176565e-19)
