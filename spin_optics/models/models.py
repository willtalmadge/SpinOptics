import numpy as np

def lorentzian(x, *p):
    """

    :param x: value to evaluate the Lorentzian at
    :param p: parameter ordering -- 0: A, 1: k, 2: xc, 3: yc
    :return: the value of the Lorentzian at x
    """
    return p[0]/(1+(p[1]*(x-p[2]))**2)+p[3]
def double_lorentzian_centered(x, *p):
    """
    A double Lorentzian model centered at x=0
    :param x:
    :param p: parameter ordering -- 0: A1, 1: k1, 2:A2, 3:k2, 4: yc
    :return: the value of the centered double Lorentzian
    """
    return p[0]/(1+(p[1]*x)**2) + p[2]/(1+(p[3]*(x))**2) + p[4]
    
def double_lorentzian_centered_no_off(x, *p):
    """
    A double Lorentzian model centered at x=0 with no offset, but it still takes
    the offset parameter so it can be used interchangeably with the offset version.
    The offset parameter will not affect the optimizer output.
    :param x:
    :param p: parameter ordering -- 0: A1, 1: k1, 2:A2, 3:k2, 4: yc
    :return: the value of the centered double Lorentzian
    """
    return p[0]/(1+(p[1]*x)**2) + p[2]/(1+(p[3]*x)**2)

def double_lorentzian_centered_fixed_off(offset):
    """
    A double Lorentzian model centered at x=0 with fixed offset, but it still takes
    the offset parameter so it can be used interchangeably with the offset version.
    The offset parameter will not affect the optimizer output.
    :param x:
    :param p: parameter ordering -- 0: A1, 1: k1, 2:A2, 3:k2, 4: yc
    :return: the value of the centered double Lorentzian
    """
    def func(x, *p):
        return p[0]/(1+(p[1]*x)**2) + p[2]/(1+(p[3]*x)**2) + offset
    return func

def centered_lorentzian_mixture(lorentzian_count, constant_offset=None):
    """
    Generates a Lorentzian mixture model with an optional fixed offset. It is up to the user to pass
    the correct length initial parameter list into an optimizer that uses this model. A runtime check
    is done for correct parameter list length to catch errors. A ValueError is thrown if the wrong length
    parameter list is passed.
    :param lorentzian_count:
    :param constant_offset:
    :return:
    """
    if constant_offset is None:
        def func(x, *p):
            if not (len(p) == 2*lorentzian_count + 1):
                raise ValueError("Passed parameter list is of length %d, expected %d" % (len(p), 2*lorentzian_count + 1))
            result = np.zeros(x.size)
            for i in range(0, lorentzian_count):
                result += p[0 + i*2]/(1+(p[1 + i*2]*x)**2)
            result += p[-1]
            return result
    else:
        def func(x, *p):
            if not (len(p) == 2*lorentzian_count):
                raise ValueError("Passed parameter list is of length %d, expected %d" % (len(p), 2*lorentzian_count))
            result = np.full(x.size, constant_offset)
            for i in range(0, lorentzian_count):
                result += p[0 + i*2]/(1+(p[1 + i*2]*x)**2)
            return result
    return func