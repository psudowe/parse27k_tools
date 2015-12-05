"""
Module to hold filesystem related utility functions.
"""

import os, shutil

from pUtils.user_query import query_yes_no

def recreate_dir(path, prompt_user=True):
    """If the directory 'path' is a directory, delete it and recreate it.
    If it is not present, make a new directory.
    By default the user will be prompted before the directory is removed
    """
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError("path is present, but not a directory")

    if os.path.isdir(path):
        query = "-- {0}\n-- the directory exists. Delete it?".format(path)
        if prompt_user and not query_yes_no(query):
            raise Exception("directory exists.")
        else:
            shutil.rmtree(path)
    # finally, create the directory
    os.makedirs(path)
