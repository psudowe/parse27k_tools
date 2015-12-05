#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import h5py
from matplotlib import pyplot as plt
from argparse import ArgumentParser

def read_crops(args):
    if args.verbose:
        print('reading from file: ', args.crop_file)
    if args.mode == 'npy':
        if args.verbose:
            print('mode: npy')
        data = np.load(args.crop_file)
        crops = data['crops']
        return crops, None
    elif args.mode == 'hdf5':
        if args.verbose:
            print('mode: hdf5')
        h = h5py.File(args.crop_file, 'r')
        # if present this will return the pids - otherwise None
        if 'pids' in h.keys():
            pids = h['pids'][:]
        else:
            pids = None
        return h['crops'], pids
    else:
        raise NotImplemented('mode unknown' + args.output_mode)

def main(args):
    print('loading crops from: ', args.crop_file)
    crops, pids = read_crops(args)
    print('crops: ', crops.shape)
    if pids is not None:
        print('read pids...')

    indices = range(len(crops))
    if args.index:
        if args.index < 0 or args.index >= len(crops):
            print('invalid index - valid are 0..', len(crops))
            sys.exit(1)
        indices = [args.index]
    if args.random:
        np.random.shuffle(indices)
    for iii in indices:
        print('example idx: ', iii)
        c = crops[iii,:]
        t = c.transpose((2,1,0))
        o = args.sub
        if o > 0:
            plt.imshow(t[o:-o,o:-o,:])
        else:
            plt.imshow(t)

        title = 'example idx: ' + str(iii)
        if pids is not None:
            title += 'pid: ' + str(pids[iii])
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--random', '-r', action='store_true', help='random order')
    parser.add_argument('--index', '-i', type=int, help='only visualize this example')
    parser.add_argument('--sub', type=int, default=0,
                        help='only show a centered subcrop (e.g. 8,16,32)')
    parser.add_argument('--mode', '-m',
                        choices=('npy', 'hdf5'),
                        default='hdf5',
                        help="data format of the prepared datasets")
    parser.add_argument('crop_file')
    args = parser.parse_args()

    main(args)
