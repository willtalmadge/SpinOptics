import base64
import os
import math

def b32uuid(n=10):
    return base64.b32encode(bytearray(os.urandom(n))).decode('utf-8').upper()

def trunc(f, decimal_precision=8):
    """
    Truncate experimental values to finite decimal precision. This allows floating point
    numbers to be used as database keys without epsilon bracketing, even if a user decides t
    o cut and paste the numbers into a query.
    :param f: the floating point number to be truncated
    :param decimal_precision: the number of digits after the decimal to retain
    :return:
    """
    if math.isnan(f):
        return float('nan')
    else:
        return int(10**decimal_precision*f)/10**decimal_precision