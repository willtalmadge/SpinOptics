from spin_optics.fitting import progressive_fit, global_curve_fit
from spin_optics.unit_converters import hanle_lifetime_gauss_in_sec
from pandas import *
from spin_optics.models import double_lorentzian_centered_no_off, centered_lorentzian_mixture, lorentzian
from spin_optics.data_wrangling import filename_containing_string_in_dirs, filename_containing_string
from spin_optics.plotting import double_lorentzian_fig
from spin_optics.unit_converters import energy_ev

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
import dateutil

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
                           regularization=None,
                           measured_offset=None):

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
                                            constant_offset=constant_offset) #TODO: this needs to be properly scaled or its essentially zero, this feature is useless right now
        init_p = np.zeros(2*lorentzian_count)
        init_p[::2] = amplitudes_init

    init_p[1::2] = inv_hwhm_init

    # Create scalers to give data zero mean and unit standard deviation
    scaler_Field = StandardScaler().fit(field_data)
    scaler_FR = StandardScaler().fit(faraday_rotation_data)

    field = scaler_Field.transform(field_data)
    fr = scaler_FR.transform(faraday_rotation_data)


    cost_func_kwargs = {}
    if measured_offset is not None:
        measured_offset = scaler_FR.transform(np.array([measured_offset]))[0]
        print("using a measured offset penalty for scaled offset %f" % measured_offset)
        cost_func_kwargs['measured_offset'] = measured_offset
    if regularization is not None:
        cost_func_kwargs['regularization'] = regularization

    p = global_curve_fit(model, field, fr, init_p, basinhopping_kwargs={
        'niter': niter,
        'stepsize': stepsize,
        'T': T
    }, cost_func_kwargs=cost_func_kwargs)

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
                          probe_background=None,
                          additional_query_params=None,
                          additional_params=None,
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

    if additional_params is None:
        additional_params = {}
    if probe_background is not None:
        additional_params.update({'probe_background': trunc(probe_background.to(ureg.radian).magnitude)})

    doc = {
        'sample_id': sample_id,
        'sample_temperature': trunc(sample_temperature.to(ureg.kelvin).magnitude),
        'probe_energy': trunc(probe_energy.to(ureg.eV).magnitude),
        'probe_intensity': trunc(probe_intensity.to(ureg.watts).magnitude),
        'pump_energy': trunc(pump_energy.to(ureg.eV).magnitude),
        'pump_intensity': trunc(pump_intensity.to(ureg.watts).magnitude),
        'when': when,
        'when_end': when_end
    }

    # Generally a fit should depend just on when it was taken and the experimental conditions
    # but if we play around with fitter parameters, we might want to store several versions
    # of a fit with the same experimental conditions
    if additional_query_params is not None:
        doc.update(additional_query_params)

    new_doc = doc.copy()
    new_doc.update({
        'amplitude': list(hanle_model_params['amplitude']),
        'inv_hwhm': list(hanle_model_params['inv_hwhm']),
        'offset': hanle_model_params['offset'],
        'rms_error' : hanle_model_params['rms_error']
    }) 
    new_doc.update(additional_params)
    old = hanle_curve_fits.find_one(doc)

    if rms_filter and (old is not None):
        # The rms filter prohibits updating documents if the new fit is worse than what is in the db
        if new_doc['rms_error'] >= old['rms_error']:
            return {'id': old['_id'], 'did_update': False}

    if old is not None:
        hanle_curve_fits.remove({'_id': old['_id']})
        new_doc.update({'_id': old['_id']})

    return {'id': hanle_curve_fits.insert(new_doc), 'did_update': True}

def update_hanle(sample_id,
                 data,
                 temperature,
                 wavelength,
                 probe_intensity,
                 pump_wavelength,
                 pump_intensity,
                 mbr_brf_displacement,
                 probe_background,
                 constrain_background=False,
                 regularization=None, peaks=2, rms_filter=True,
                 additional_params=None,
                 db_conn=None):

    if additional_params is None:
        additional_params = {}
    additional_params.update({
            'mbr_brf_displacement': mbr_brf_displacement.to(ureg.millimeters).magnitude
        })
    additional_params.update(additional_params)

    additional_query_params = {}
    if constrain_background:
        additional_query_params.update({"constrained_offset": 1})
    else:
        additional_query_params.update({"constrained_offset": 0})

    if regularization is not None:
        additional_params.update({'fitter_regularization_used': regularization})

    if constrain_background:
        measured_offset = probe_background
    else:
        measured_offset = None

    p = global_hanle_curve_fit(data.Field.values, data.FR.values, peaks, stepsize=500,
                               T=200, niter=100, regularization=regularization,
                              measured_offset=measured_offset)

    total_amp = sum(p['amplitude'])
    for a in p['amplitude']:
        if abs(total_amp/a) < 0.1:
            print('Nearly equal peak cancellation, rejecting solution')
            return None

    when = pytz.timezone("MST").localize(
        dateutil.parser.parse(data.iloc[0]['Timestamp'])
    ).astimezone(pytz.utc)
    when_end = pytz.timezone("MST").localize(
        dateutil.parser.parse(data.iloc[-1]['Timestamp'])
    ).astimezone(pytz.utc)

    result = store_hanle_curve_fit(
        sample_id=sample_id,
        sample_temperature=temperature,
        probe_energy=energy_ev(wavelength.to(ureg.nanometers).magnitude) * ureg.eV,
        probe_intensity=probe_intensity,
        pump_energy=energy_ev(pump_wavelength.to(ureg.nanometers).magnitude) * ureg.eV,
        pump_intensity=pump_intensity,
        when=when,
        when_end=when_end,
        hanle_model_params=p,
        probe_background=probe_background,
        additional_params=additional_params,
        additional_query_params=additional_query_params,
        db_conn=db_conn,
        rms_filter=rms_filter
    )

    if result['did_update']:
        return p

def plot_hanle_curve(hanle_curve_data, hanle_curve_fit_params):
    f2, ax2 = plt.subplots()

    # Plot the data
    ax2.plot(hanle_curve_data.Field, hanle_curve_data.FR, 'o', color=sb.xkcd_rgb['black'], rasterized=True, alpha=0.3)

    # Plot the curve mean
    dm = hanle_curve_data.groupby('Field')
    ax2.plot(dm.Field.mean(), dm.FR.mean(), '-o', color=sb.xkcd_rgb['mango'], alpha=0.5, markersize=4)

    field_grid = np.linspace(np.min(hanle_curve_data.Field), np.max(hanle_curve_data.Field), 1000)
    # Plot the multiple lorentzian
    count = len(list(hanle_curve_fit_params['amplitude']))
    model = centered_lorentzian_mixture(count)
    params = np.zeros(2*count + 1)
    params[:-1:2] = hanle_curve_fit_params['amplitude']
    params[1:-1:2] = hanle_curve_fit_params['inv_hwhm']
    params[-1] = hanle_curve_fit_params['offset']

    ax2.plot(field_grid, [model(x, *params) for x in field_grid],
             color=sb.xkcd_rgb['tomato red'], linewidth=3)
    print(np.array([model(x, *params) for x in dm.Field.mean()]).flatten().shape)
    print(np.array(dm.FR.mean()).shape)
    ax2.plot(dm.Field.mean(), 10*(np.array([model(x, *params) for x in dm.Field.mean()]).flatten() - dm.FR.mean()),
             color=sb.xkcd_rgb['dark yellow'], linewidth=3)

    colors = [sb.xkcd_rgb['cobalt'], sb.xkcd_rgb['azure']] + list(sb.xkcd_rgb.values())
    for i in range(0, count):
        ax2.plot(field_grid, [lorentzian(x, *[params[0+2*i], params[1+2*i], 0, 0]) for x in field_grid],
                 color=colors[i], linewidth=2, label=('peak %d' % (i+1)))

    ax2.set_yticklabels(ax2.get_yticks()/1e-6)

    plt.legend()

    ax2.set_ylabel('Faraday Rotation ($\mu rad$)')
    ax2.set_xlabel('Field (Gauss)')

def hanle_curve_title_from_params(fit_params, db_conn):

    sample = db_conn.spin_optics.samples.find_one({
        '_id': fit_params['sample_id']
    })

    result = ('Hanle Effect on $%s$\n' % sample['system'] )
    if 'capping' in sample.keys():
        result += ('Capping layer "%s"' % sample['capping'])
    if 'substrate' in sample.keys():
        result += (', Substrate "%s"\n' % sample['substrate'])
    else:
        result += '\n'
    result += ('Pump %.2f eV @ %d $\mu W$ Probe %.2f eV @ %d $\mu W$\n' % (
        fit_params['pump_energy'],
        fit_params['pump_intensity']*1e6,
        fit_params['probe_energy'],
        fit_params['probe_intensity']*1e6
    ))
    result += ('Temperature %.1f K\n' % fit_params['sample_temperature'])

    for a, l, i in zip(fit_params['amplitude'],
                       fit_params['inv_hwhm'],
                       range(0, len(list(fit_params['amplitude'])))):
        result += ("Peak %d HWHM %d Gauss (g=1 lifetime is %.3fns\n" % (i+1, 1/l, 1e9*hanle_lifetime_gauss_in_sec(l)))

    result += ("(SID %s, Measured %s)" % (sample['_id'], str(fit_params['when'])))
    return result


def hanle_fit_for_data(hanle_curve_data, db_conn):
    when = pytz.timezone("MST").localize(
        dateutil.parser.parse(hanle_curve_data.iloc[0]['Timestamp'])
    ).astimezone(pytz.utc)
    when_end = pytz.timezone("MST").localize(
        dateutil.parser.parse(hanle_curve_data.iloc[-1]['Timestamp'])
    ).astimezone(pytz.utc)

    return db_conn.spin_optics.hanle_curve_fits.find_one({
        'when': when,
        'when_end': when_end
    })

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