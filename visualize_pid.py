#!/usr/bin/env python
'''
Shows the full image in which the pedestrian with pedestrianID
is contained.
This is mainly just to get a quick feel for how to use the annotations.
'''

from __future__ import print_function

import os
import sqlite3
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

def main(args):
    try:
        dbFile = os.path.join(args.parse_path, 'annotations.sqlite3')
        db = sqlite3.connect(dbFile)
        dbc = db.cursor()
    except sqlite3.Error as e:
        raise Exception(e)

    query = '''SELECT directory, filename
                FROM Pedestrian p
                      JOIN Image i ON i.imageID = p.imageID
                      JOIN Sequence s ON s.sequenceID = i.sequenceID
                WHERE p.pedestrianID = {}
            '''
    result = dbc.execute(query.format(args.pid)).fetchone()

    imgfn = os.path.join(args.parse_path, 'sequences', result[0], result[1])
    print('image file: ', imgfn)
    try:
        img = np.array(Image.open(imgfn).convert('RGB'))
    except IOError as e:
        print('could not load image file')
        sys.exit(1)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--parse_path',
                        default=os.getenv('PARSE_PATH',
                                          '/work/sudowe/datasets/parse/'))
    parser.add_argument('pid', type=int,
                        help='show image containing the pedestrian with PID')
    args = parser.parse_args()
    main(args)
