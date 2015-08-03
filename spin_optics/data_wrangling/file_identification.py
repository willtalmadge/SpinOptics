import os
import re

def filename_contains_string(string, filename):
    return re.search('.*' + string + '.*', filename) is not None
def filename_containing_string(string, directory='.'):
    result = [x for x in os.listdir(directory) if filename_contains_string(string, x) == True]
    if len(result) > 0:
        if directory != '.':
            result = directory + result[0]
            return result
        else:
            return result[0]
    else:
        return ""