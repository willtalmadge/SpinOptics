from spin_optics.fitting import progressive_fit
from spin_optics.unit_converters import hanle_lifetime_gauss_in_sec
from pandas import *
from spin_optics.models import double_lorentzian_centered_no_off
from spin_optics.data_wrangling import filename_containing_string_in_dirs, filename_containing_string
from spin_optics.plotting import double_lorentzian_fig
import matplotlib.pyplot as plt
import numpy as np

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
                          model=double_lorentzian_centered_no_off):
    hanle_model_cols = ['A1', 'k1', 'A2', 'k2']
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
                        double_lorentzian_centered_no_off,
                        path,
                        key_name=independent_variable_name,
                        x_name=mag_field_col_name,
                        y_name=faraday_rot_col_name)

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
        plt.text(data[var_col].iloc[x], data[A2_col].iloc[x], str(data[key_col].iloc[x]), fontsize=6)

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
             data.loc[i, k2_col], 0]
        fig = double_lorentzian_fig(p, d[Field_col], d[FR_col],
                                    (file_key_format + '\n %fnm, %fns, %fns') % (data.loc[i, key_col],
                                                                                 data.loc[i, var_col],
                                                                         data.loc[i, L1_col]/1e-9,
                                                                         data.loc[i, L2_col]/1e-9))
        fig.savefig('./fits/%04d_%04d.pdf' % (data.loc[i, var_col]*10, data.loc[i, key_col]))
        plt.close(fig)