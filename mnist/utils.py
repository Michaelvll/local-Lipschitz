import os


def valid_dir(dir_name):
    dir_name = os.path.dirname(dir_name)
    if not os.path.exists(dir_name) and (dir_name is not ''):
        os.makedirs(dir_name)
