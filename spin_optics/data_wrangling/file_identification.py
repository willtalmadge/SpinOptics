import os, os.path
import re
from enum import Enum

def filename_contains_string(string, filename):
    """
    Determine if a filename contains a string
    :param string: string to search for
    :param filename: the filename
    :return: True or False
    """
    return re.search('.*' + string + '.*', filename) is not None


def filename_containing_string(string, directory='.'):
    """
    Find a list of file paths containing the specified string in the specified directory
    :param string: string to search for
    :param directory: directory to search
    :return:
    """
    result = [x for x in os.listdir(directory) if filename_contains_string(string, x) == True]
    if len(result) > 0:
        if directory != '.':
            result = os.path.join(directory, result[0])
            return result
        else:
            return result[0]
    else:
        return ''


def filename_containing_string_in_dirs(sub_string, search_dirs=['.']):
    """
    Find a list of file paths containing the specified string, searching in the specified subdirectories
    :param sub_string:
    :param search_dirs:
    :return:
    """
    filename = filename_containing_string(sub_string)
    if filename == '':
        for d in search_dirs:
            filename = filename_containing_string(sub_string, directory=d)
            if filename != '':
                return filename
    return filename


def data_path(year, month, day):
    """
    Find the google drive path for a data folder based on year/month/data handles the case where
    the spin optics data folder is shared or on the main dat acquisition system
    :param year:
    :param month:
    :param day:
    :return: The full path to the data folder
    """
    if os.path.exists(os.path.expanduser('~/Google Drive/Spin Optics/Data')):
        dir = os.path.expanduser('~/Google Drive/Spin Optics/Data/%04d/%02d/%02d' % (year, month, day))
    else:
        raise FileNotFoundError('Could not locate the spin optics data path')

    if os.path.exists(dir):
        return dir
    else:
        return None


def count_files_in_dir(path):
    """
    Count how many files are in a directory, not including the folders
    :param path:
    :return: an integer that counts the files in the directory
    """

    if path is None:
        return None
    if os.path.exists(path):
        return len([name for name in os.listdir(path) if (
            os.path.isfile(os.path.join(path, name)) and name != 'Icon\r')])
    else:
        return None


def date_id(year, month, day, hour=None, minute=None, second=None):
    """
    Generates an identifier based on the date and optionally the time, constructed such that sorting
    this string by character will sort chronologically

    :param year:
    :param month:
    :param day:
    :param hour:
    :param minute:
    :param second:
    :return:
    """
    did = '%04d%02d%02d' % (year, month, day)
    if hour is not None:
        did += '%02d' % hour
    if minute is not None:
        did += '%02d' % minute
    if second is not None:
        did += '%02d' % second
    return did


class measurement_types(Enum):
    hanle_effect='Hanle Effect'
    photoluminescence_spectroscopy='Photoluminescence Spectroscopy'
    transmission='Transmission'


def experiments_base_path(*path_append):
    base = os.path.expanduser('~/Google Drive/Spin Optics/Experiments')
    if len(path_append) > 0:
        return os.path.join(base, *path_append)
    return base


def experiment_path(sample_id, measurement, name, eid, sample_name=None):
    """
    Returns a path to a folder where experiment related data can be stored. The experiment
    folder is
    if eid is generated by date_id then the folders created by this
    :param sample_id: an identifier that uniquely identifies the sample
    :param measurement: the type of measurement
    :param name: name of the experiment
    :param eid: an identifier that uniquely identifies the experiment (it is suggested that date_id is used for this)
    :return:
    """
    if sample_name is None:
        sample_name = ''
    else:
        sample_name = sample_name + ' '
    path = experiments_base_path()
    if os.path.exists(path):
        path = os.path.join(path, sample_name + sample_id, measurement.value, eid + ' ' + name)
    else:
        raise FileNotFoundError('Could not locate the spin optics data path')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def exp_dir_from_env(envir):
    path = experiment_path(envir['sample_id'], measurement_types.hanle_effect,
                          envir['additional_params']['experiment'], envir['eid'],
                         sample_name=envir['sample_description'])
    return path