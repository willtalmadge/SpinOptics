from spin_optics.fitting import progressive_fit, global_curve_fit
from spin_optics.unit_converters import hanle_lifetime_gauss_in_sec
from pandas import *
from spin_optics.models import double_lorentzian_centered_no_off, centered_lorentzian_mixture, lorentzian
from spin_optics.data_wrangling import filename_containing_string_in_dirs, filename_containing_string
from spin_optics.plotting import double_lorentzian_fig
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import scipy.optimize as opt

from pymongo import MongoClient

from spin_optics import ureg, Q_
from functools import partial

from spin_optics.misc import trunc

import pytz
import time
"""
Goal is to split this into two phases, generate the hanle parameters and potentially cache or aggregate
into a central database.

Second phase is to generate plots for all of the hanle parameter rows.

To deal with duplicate independent variables, simply enter the hanle parameters into all fields

The fit paths should run along the independent variable, use a helper function to convert
timestamp ranges to independent variables. A path implies a sequence of independent variables.
"""

def generate_hanle_params(data,
                          file_key_column,
                          independent_variable_name,
                          search_dirs,
                          fit_paths,
                          file_key_format='%04d',
                          file_column_names=['Field', 'X', 'Y', 'PDA', 'FR', 'Timestamp'],
                          mag_field_col_name='Field',
                          faraday_rot_col_name='FR',
                          X_col_name='X',
                          Y_col_name='Y',
                          model=double_lorentzian_centered_no_off,
                          fitter=opt.curve_fit):
    hanle_model_cols = ['A1', 'k1', 'A2', 'k2', 'y0']
    for c in hanle_model_cols:
        if not c in data:
            data[c] = np.nan

    def curve_loader(value):
        rs = data[data[independent_variable_name] == value]
        result = read_csv(filename_containing_string_in_dirs(file_key_format % rs[file_key_column].iloc[0], search_dirs))
        result.columns = file_column_names
        if len(rs.index) > 1:
            for i, r in rs[1:].iterrows():
                d = read_csv(filename_containing_string_in_dirs(file_key_format % rs[file_key_column].iloc[0], search_dirs))
                d.columns = file_column_names
                result = result.append(d)
        return result
    for init_p, path in fit_paths:
        progressive_fit(data,
                        curve_loader,
                        init_p,
                        hanle_model_cols,
                        model,
                        path,
                        key_name=independent_variable_name,
                        x_name=mag_field_col_name,
                        y_name=faraday_rot_col_name,
                        fitter=fitter)

    data['L1'] = data[hanle_model_cols[1]].apply(lambda k: hanle_lifetime_gauss_in_sec(abs(k)))
    data['L2'] = data[hanle_model_cols[3]].apply(lambda k: hanle_lifetime_gauss_in_sec(abs(k)))

    #Calculate median phase of each Hanle curve and store it in the results
    if not 'MP' in data:
        data['MP'] = np.nan
    for i, r in data.iterrows():
        d = read_csv(filename_containing_string_in_dirs(file_key_format % r[file_key_column], search_dirs))
        d.columns = file_column_names
        data.loc[i, 'MP'] = np.median(np.arctan2(d[Y_col_name], d[X_col_name]))

def plot_and_save_hanle_params(data,
                               key_col='Timestamp',
                               var_col='W',
                               A1_col='A1',
                               A2_col='A2',
                               L1_col='L1',
                               L2_col='L2',
                               TP_col='',
                               PP_col='',
                               independent_var_label='Wavelength (nm)',
                               amplitude_title='',
                               transmission_title='',
                               absorbance_title='',
                               lifetime_title='',
                               phases_title='',
                               file_key_format='%04d'
                               ):
    #TODO: add the individual plot fits
    #Amplitude
    fig, ax = plt.subplots()
    plt.plot(data[var_col], data[A1_col], 'or')
    plt.plot(data[var_col], data[A2_col], 'ob', alpha=0.5)

    for x in range(0,data.W.count()):
        plt.text(data[var_col].iloc[x], max(data[A2_col].iloc[x], data[A1_col].iloc[x]),
                 str(data[key_col].iloc[x]), fontsize=6)

    ax.set_yticklabels(ax.get_yticks()/1e-6)
    plt.ylabel('Faraday Rotation ($\mu rad$)')
    plt.xlabel(independent_var_label)
    plt.title(amplitude_title)
    plt.axhline()
    plt.savefig('amplitude.pdf')
#    DataFrame({key_col: data[key_col],
#               'Wavelength (nm)' : data.W,
#               'A1 (rad)': data.A1,
#               'A2 (rad)': data.A2}).to_csv('amplitude.dat')

    #Transmission
    if TP_col != '' and PP_col != '':
        fig, ax = plt.subplots()
        plt.plot(data[var_col], data[TP_col]/data[PP_col], 'og')
        plt.xlabel(independent_var_label)
        plt.ylabel('Transmittance')
        plt.title(transmission_title)
        for x in range(0,data[var_col].count()):
            plt.text(data.W.iloc[x], data[TP_col].iloc[x]/data[PP_col].iloc[x], str(data[key_col].iloc[x]), fontsize=6)
        plt.savefig('transmission.pdf')
#    DataFrame({'Timestamp':data.Timestamp,
#               'Wavelength (nm)':data.W,
#               'transmission': data.TP/data.PP}).to_csv('transmission.dat')

    #Absorption
    if TP_col != '' and PP_col != '':
        fig, ax = plt.subplots()
        plt.plot(data[var_col], -np.log(data[TP_col]/data[PP_col]), 'og')
        plt.xlabel(independent_var_label)
        plt.ylabel('Absorbance')
        plt.title(absorbance_title)
        plt.savefig('absorption.pdf')

    #Lifetime
    fig, ax = plt.subplots()
    plt.semilogy(data[var_col], abs(data[L1_col]), 'or')
    plt.semilogy(data[var_col], abs(data[L2_col]), 'ob', alpha=0.5)
    ax.set_yticklabels(ax.get_yticks()/1e-9)
    #for x in range(0,data_u.W.count()):
    #    text(data_u.W.iloc[x], t1, str(data_u.Timestamp.iloc[x]), fontsize=6)

    plt.ylabel('Lifetime (ns)')
    plt.xlabel(independent_var_label)
    plt.title(lifetime_title)
    plt.savefig('lifetime.pdf')
#    DataFrame({'Timestamp': data.Timestamp,
#               'Wavelength (nm)': data.W,
#               'lifetime 1 (sec)': data.L1,
#               'lifetime 2 (sec)': data.L2}).to_csv('lifetime.dat')

def plot_hanle_curve_fits(data,
                          key_col='Timestamp',
                          var_col='W',
                          A1_col='A1',
                          A2_col='A2',
                          k1_col='k1',
                          k2_col='k2',
                          L1_col='L1',
                          L2_col='L2',
                          y0_col='y0',
                          Field_col='Field',
                          FR_col='FR',
                          file_key_format='%04d',
                          search_dirs='.',
                          file_column_names=['Field', 'X', 'Y', 'PDA', 'FR', 'Timestamp']
                          ):
    for i in range(0, len(data.index)):
        d = read_csv(filename_containing_string_in_dirs(file_key_format % data.loc[i, key_col], search_dirs))
        d.columns = file_column_names
        p = [data.loc[i, A1_col],
             data.loc[i, k1_col],
             data.loc[i, A2_col],
             data.loc[i, k2_col],
             data.loc[i, y0_col]]
        dm = d.groupby('Field')
        fig = double_lorentzian_fig(p, d[Field_col], d[FR_col],
                                    xm=dm.Field.mean(), ym=dm.FR.mean(),
                                    title=(file_key_format + '\n %fnm, %fns, %fns') % (data.loc[i, key_col],
                                                                                       data.loc[i, var_col],
                                                                                       data.loc[i, L1_col]/1e-9,
                                                                                       data.loc[i, L2_col]/1e-9))
        fig.savefig('./fits/%04d_%04d.pdf' % (data.loc[i, var_col]*10, data.loc[i, key_col]))
        plt.close(fig)

def global_hanle_curve_fit(field_data, faraday_rotation_data, lorentzian_count,
                           niter=100,
                           T=100,
                           stepsize=500,
                           threads_for_repeats=8,
                           constant_offset=None,
                           penalize_offset=False,  #This precludes regularization
                           regularization=None):

    # Construct initial conditions that spread out the widths. We expect some distribution typically
    # so an evenly spaced distribution of widths allows the fitter to search for a range of widths
    # in the source data.
    inv_hwhm_init = 1/np.arange(0.1, 1, 1/lorentzian_count)
    amplitudes_init = np.random.rand(lorentzian_count)

    if constant_offset is None:
        # Construct a model with the background constant as a free parameter
        model = centered_lorentzian_mixture(lorentzian_count=lorentzian_count)
        init_p = np.zeros(2*lorentzian_count + 1)
        init_p[:-1:2] = amplitudes_init
    else:
        # Construct a model with the background as a fixed constant (not available to be optimized)
        model = centered_lorentzian_mixture(lorentzian_count=lorentzian_count,
                                            constant_offset=constant_offset) #TODO: this needs to be properly scaled or its essentially zero
        init_p = np.zeros(2*lorentzian_count)
        init_p[::2] = amplitudes_init

    init_p[1::2] = inv_hwhm_init

    # Create scalers to give data zero mean and unit standard deviation
    scaler_Field = StandardScaler().fit(field_data)
    scaler_FR = StandardScaler().fit(faraday_rotation_data)

    field = scaler_Field.transform(field_data)
    fr = scaler_FR.transform(faraday_rotation_data)

    p = global_curve_fit(model, field, fr, init_p, basinhopping_kwargs={
        'niter': niter,
        'stepsize': stepsize,
        'T': T
    })

    # Extract the parameters from the solution, and rescale the background
    if constant_offset is None:
        amplitudes_opt = p.x[:-1:2]
        offset_opt = p.x[-1]*scaler_FR.std_ + scaler_FR.mean_
    else:
        amplitudes_opt = p.x[::2]
        offset_opt = constant_offset
    inv_hwhm_opt = np.abs(p.x[1::2])

    # Sort the results with the narrow Lorentzians appearing at the lowest indices
    permute = inv_hwhm_opt.argsort()
    inv_hwhm_opt = inv_hwhm_opt[permute]
    amplitudes_opt = amplitudes_opt[permute]

    # Remove the scaling from the parameters so they are in the units of the source data
    amplitudes_opt = amplitudes_opt*scaler_FR.std_
    inv_hwhm_opt = inv_hwhm_opt/scaler_Field.std_

    return {"amplitude": amplitudes_opt,
            "inv_hwhm" : inv_hwhm_opt,
            "offset" : offset_opt,
            "rms_error": p.fun}

def store_hanle_curve_fit(sample_id,
                          sample_temperature,
                          probe_energy,
                          probe_intensity,
                          pump_energy,
                          pump_intensity,
                          hanle_model_params,
                          when,
                          when_end,
                          parameter_dict={},
                          db_conn=None,
                          rms_filter=True):

    if (when.tzinfo is not pytz.UTC) or (when_end.tzinfo is not pytz.UTC):
        raise ValueError("When should be a datetime with a tzinfo of pytz.UTC")

    if db_conn is None:
        db_conn = MongoClient()

    db = db_conn.spin_optics

    samples = db.samples
    if not (samples.find({'_id' : sample_id}).count() == 1):
        raise ValueError('The provided sample id does not exist in the spin optics samples collection')

    hanle_curve_fits = db.hanle_curve_fits

    doc = {
        'sample_id': sample_id,
        'sample_temperature': trunc(sample_temperature.to(ureg.kelvin).magnitude),
        'probe_energy': trunc(probe_energy.to(ureg.eV).magnitude),
        'probe_intensity': trunc(probe_intensity.to(ureg.watts).magnitude),
        'pump_energy': trunc(pump_energy.to(ureg.eV).magnitude),
        'pump_intensity': trunc(pump_intensity.to(ureg.watts).magnitude),
        'when':when,
        'when_end': when_end
    }
    new_doc = doc.copy()
    new_doc.update({
        'amplitude': list(hanle_model_params['amplitude']),
        'inv_hwhm': list(hanle_model_params['inv_hwhm']),
        'offset': hanle_model_params['offset'],
        'rms_error' : hanle_model_params['rms_error']
    }) 
    new_doc.update(parameter_dict)
    old = hanle_curve_fits.find_one(doc)

    if rms_filter and (old is not None):
        # The rms filter prohibits updating documents if the new fit is worse than what is in the db
        if new_doc['rms_error'] >= old['rms_error']:
            return {'id': old['_id'], 'did_update': False}

    if old is not None:
        hanle_curve_fits.remove({'_id': old['_id']})
        new_doc.update({'_id': old['_id']})

    return {'id': hanle_curve_fits.insert(new_doc), 'did_update': True}

def plot_hanle_curve(hanle_curve_data, hanle_curve_fit_params):
    f2, ax2 = plt.subplots()

    # Plot the data
    ax2.plot(hanle_curve_data.Field, hanle_curve_data.FR, 'o', color=sb.xkcd_rgb['black'], rasterized=True, alpha=0.3)

    # Plot the curve mean
    dm = hanle_curve_data.groupby('Field')
    ax2.plot(dm.Field.mean(), dm.FR.mean(), '-o', color=sb.xkcd_rgb['mango'], alpha=0.5, markersize=4)


    # Plot the multiple lorentzian
    count = len(list(hanle_curve_fit_params['amplitude']))
    model = centered_lorentzian_mixture(count)
    params = np.zeros(2*count + 1)
    params[:-1:2] = hanle_curve_fit_params['amplitude']
    params[1:-1:2] = hanle_curve_fit_params['inv_hwhm']
    params[-1] = hanle_curve_fit_params['offset']

    ax2.plot(hanle_curve_data.Field, [model(x, *params) for x in hanle_curve_data.Field],
             color=sb.xkcd_rgb['tomato red'], linewidth=3)

    colors = [sb.xkcd_rgb['cobalt'], sb.xkcd_rgb['azure']] + list(sb.xkcd_rgb.values())
    for i in range(0, count):
        ax2.plot(hanle_curve_data.Field, [lorentzian(x, *[params[0+2*i], params[1+2*i], 0, 0]) for x in hanle_curve_data.Field],
                 color=colors[i], linewidth=2, label=('peak %d' % (i+1)))

    ax2.set_yticklabels(ax2.get_yticks()/1e-6)

    plt.legend()

    ax2.set_ylabel('Faraday Rotation ($\mu rad$)')
    ax2.set_xlabel('Field (Gauss)')


def hanle_curve_title_from_stored(curve_id, db_conn=None):
    if db_conn is None:
        db_conn = MongoClient()
    db = db_conn.spin_optics
    samples = db.samples
    hanle_curve_fits = db.hanle_curve_fits

    curve = hanle_curve_fits.find_one({
        '_id': curve_id
    })
    sample = samples.find_one({
        '_id': curve['sample_id']
    })

    if curve is not None:
        result = ( \
                     'Hanle Effect on $%s$ \n'
                     'Pump %.2f eV @ %d $\mu W$ Probe %.2f eV @ %d $\mu W$\n'
                     'Temperature %.2f K'
                     )% (sample['system'],
                         curve['when'],
                         curve['pump_energy'],
                         curve['pump_intensity'],
                         curve['probe_energy'],
                         curve['probe_intensity'],
                         curve['sample_temperature'])
        return result
    else:
        raise ValueError('Bad curve id, no curve found')