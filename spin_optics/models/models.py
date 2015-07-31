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