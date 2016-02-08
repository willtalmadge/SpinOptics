import base64
import os
import math
import numpy as np
import os.path
import json

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

def rephase(x, y, theta_offset):
    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x) + np.pi + theta_offset) % np.pi
    return(r * np.cos(theta), r*np.sin(theta))

def spinoptics_settings():
    try:
        path = os.path.expanduser('~/.spinoptics.json')
        with open(path) as file:
            settings = json.load(file)
        return settings
    except FileNotFoundError:
        raise FileNotFoundError("Call create_spinoptics_settings_template() to add .spinoptics.json to your home folder")

def create_spinoptics_settings_template():
    path = os.path.expanduser('~/.spinoptics.json')
    with open(path, mode='w') as file:
        file.write('{\n'
                   '\t"mongodb_connection": "mongodb://user:pass@ds031587.mongolab.com:31587/spin_optics"\n'
                   '}\n')
    print('~/.spinoptics.json created, open it in an editor and file in log in details')