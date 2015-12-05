"""
Helper routine that dumps a argparse.Namespace object to a file
"""

import os
import json


def dump_args(args, directory='./', filename='parameters'):
    """dumps variables in args to a file 'parameters' in directory
    This can be used to make a record of the exact way a program/script
    was called by passing the ArgumentParser object to this.
    """
    if not os.path.isdir(directory):
        raise IOError('directory does not exist')
    out = os.path.join(directory, filename)
    if os.path.isfile(out):
        raise IOError('the parameters record file already exists!')

    with open(out, 'w') as outfile:
        outfile.write(json.dumps(vars(args), sort_keys=True, indent=4))
