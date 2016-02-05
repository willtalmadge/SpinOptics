from pymongo import MongoClient
import numpy as np
import pandas as pd

def collection_fields(cursor):
    """
    Return a set of all the fields in the given cursor
    :param cursor:
    :return:
    """
    keys = set()
    for doc in cursor:
        keys = keys | set(doc.keys())
    return keys

def field_dataframe(cursor, fields):
    """
    Create a data frame from a list of fields
    :param cursor: the mongodb collection
    :param fields:
    :return:
    """
    data_dict = {}
    for field in fields:
        data_dict[field] = []
    for doc in list(cursor):
        for field in fields:
            try:
                value = doc[field]
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if (field + '_' + str(i)) not in data_dict.keys():
                            if field in data_dict.keys():
                                del data_dict[field]
                            data_dict[field + '_' + str(i)] = []
                        data_dict[field + '_' + str(i)].append(v)
                else:
                    data_dict[field].append(doc[field])
            except Exception:
                data_dict[field].append(np.nan)

    return pd.DataFrame(data_dict)

def get_hanle_params(query, con):
    samples = con.spin_optics.samples
    hanle_curve_fits = con.spin_optics.hanle_curve_fits
    fits = hanle_curve_fits.find(query)
    es = np.zeros(fits.count())
    amps1 = np.zeros(fits.count())
    amps2 = np.zeros(fits.count())
    l1 = np.zeros(fits.count())
    l2 = np.zeros(fits.count())
    pi = np.zeros(fits.count())
    offset = np.zeros(fits.count())
    fixed = np.zeros(fits.count())
    background = np.zeros(fits.count())

    constrained = False
    for fit, i in zip(fits, range(0, fits.count())):
        es[i] = fit['probe_energy']
        amps1[i] = fit['amplitude'][0]
        l1[i] = fit['inv_hwhm'][0]
        if len(fit['amplitude']) < 2:
            amps2[i] = 0
            l1[i] = 0
        else:
            amps2[i] = fit['amplitude'][1]
            l2[i] = fit['inv_hwhm'][1]
        pi[i] = fit['probe_intensity']
        offset[i] = fit['offset']
        background[i] = fit['probe_background']
        if 'constrained_offset' in fit.keys():
            constrained = True
            fixed[i] = fit['constrained_offset']
    pmt = np.argsort(es)
    es = es[pmt]
    amps1 = amps1[pmt]
    amps2 = amps2[pmt]
    l1 = l1[pmt]
    l2 = l2[pmt]
    pi = pi[pmt]
    offset = offset[pmt]


    result = {
        'es': es,
        'amps1': amps1,
        'amps2': amps2,
        'l1': l1,
        'l2': l2,
        'pi': pi,
        'offset': offset,
        'background': background
    }
    if constrained:
        fixed = fixed[pmt]
        result.update({'fixed': fixed})
    return result