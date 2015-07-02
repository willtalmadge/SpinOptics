import os
import re

def filename_contains_string(string, filename):
    return re.search('.*' + string + '.*', filename) is not None
def filename_containing_string(string, directory='.'):
    return [x for x in os.listdir(directory) if filename_contains_string(string, x) == True][0]