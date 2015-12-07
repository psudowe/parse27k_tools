#!/usr/bin/env python
'''
Generates random predictions.
This is only to give an example of the dataformat that is expected by
the script: parse_evaluation.py
'''

from __future__ import print_function
import h5py
from argparse import ArgumentParser 
import numpy as np

def main(args):
    gt = h5py.File(args.groundtruth, 'r')
    attribute_names = gt['labels'].attrs['names']
    valid_values = gt['labels'].attrs['valid_values']

    N = gt['labels'].shape[0]

    out = h5py.File(args.output, 'w')
    for idx, attr in enumerate(attribute_names):
        K = valid_values[idx]
        predictions = np.random.random((N,K))
        if args.mode == 'retrieval': 
           # make the first column 0
           predictions[:,0] = 0
        # normalize rows -- keepdims to allow broadcast
        sums = np.sum(predictions, axis=1, keepdims=1)
        predictions = predictions / sums
        out.create_dataset(attr, data=predictions)
    out.close()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('groundtruth', help='HDF5 file with groundtruth labels')
    parser.add_argument('output', help='filename output')
    parser.add_argument('--mode', choices=('retrieval', 'classification'),
                        default='classification')
    args = parser.parse_args()
    main(args)
