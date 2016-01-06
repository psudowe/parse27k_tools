#!/usr/bin/env python

"""
Reads the images, crops pedestrian examples with sufficient additional padding (-p),
and saves the result in a single file for fast loading.

For convenience, we save two copies of each crop - original and mirrored.
For the mirrored example, the labels are adjusted accordingly. Note, that this is non-trivial
because the bag attributes depend on the orientation.
"""

from __future__ import print_function
import os, sys, time, os.path as path
from argparse import ArgumentParser
from progressbar import ProgressBar
import concurrent.futures
import h5py

import numpy as np
from skimage.util import pad
from skimage import transform
from PIL import Image

from attribute_datasets import DataPreprocessorPARSE as PARSE
from pUtils import git, dump_args, recreate_dir

def crop(imgfn, box, target_size, padding, padding_mode='zero', mirror=False):
    '''
    Load an image and crop bounding box from it.
    Handles additional padding - if the box is too close to the image boundary,
    the image is padded according to args.padding_mode (i.e. edge or zero)

    Parameters
    ----------
      box: the bounding box to be cropped - tuple (min_x, min_y, max_x, max_y)

      target_size: (2-tuple), scale each image s.t. the bounding box
         is of height target_size[1]
         ( then adapt in x-direction s.t. it matches target_size[0])

      padding: number of pixels of additional padding around the bounding box

      padding_mode: 'zero' or 'edge' - controls how the padded pixels are filled

      mirror: if true - the resulting crop is mirrored (reversed x-axis)

    Return
    ------
      a crop of shape: target_size + padding
      type: np.array in ordering [c,w,h] - of type uint8 for compact memory footprint
    '''
    img = np.array(Image.open(imgfn).convert('RGB'))
    cur_h, cur_w, cur_c = img.shape

    # 1. rescale the whole image s.t. bounding box has target height
    # 2. adapt box accordingly (scale to the new image dim, then adapt width)
    # 3a. add additional 'padding' to bounding box
    # 3b. add padding around the image - in case it is required
    # 4. take the crop
    # 5. transpose the dimensions
    sf = float(target_size[1]) / (box[3]-box[1])

    # 1.
    img_l = transform.resize(img, (int(cur_h * sf), int(cur_w * sf), cur_c))
    pb = [np.floor(sf*x + .5) for x in box]

    # 2.
    delta = (target_size[0]-(pb[2]-pb[0])) / 2.0
    pb[0] -= np.floor(delta + 0.5)
    pb[2] += np.floor(delta)
    if pb[2]-pb[0] <> target_size[0]:
        raise Exception('new box width does not match target')
    if pb[3]-pb[1] <> target_size[1]:
        raise Exception('new box height does not match target')

    # 3a
    pb[0] -= padding # the padding around bounding box (not the whole image)
    pb[1] -= padding
    pb[2] += padding
    pb[3] += padding
    pb = [int(x) for x in pb]

    # 3b
    if pb[0] < 0 or pb[1] < 0 or pb[2] >= img_l.shape[1] or pb[3] >= img_l.shape[0]:
        pad_offset = np.max(target_size) + padding
        pb = [int(x+pad_offset) for x in pb]

        if padding_mode == 'edge':
            img = pad(img_l, [(pad_offset, pad_offset),
                              (pad_offset, pad_offset),
                              (0,0)], mode='edge')
        elif padding_mode == 'zero':
            img = pad(img_l, [(pad_offset, pad_offset),
                              (pad_offset, pad_offset),
                              (0,0)], mode='constant', constant_values=0)
        else:
            raise NotImplemented('padding mode not implemented: ', padding_mode)
    else:
        img = img_l # no extra padding around the image required

    # 4.
    if mirror:
        acrop = img[pb[1]:pb[3], pb[2]:pb[0]:-1, :] # reversed x-dimension
    else:
        acrop = img[pb[1]:pb[3], pb[0]:pb[2], :]
    # transform and pad implicitly convert to float32 (range [0,1])
    out = (255. * acrop).astype(np.uint8)
    return out.transpose((2,1,0)) # transpose to (c,w,h)

def do_one_crop(e):
    c = crop(e.image_filename, e.box,
             (args.width, args.height),
             padding=args.padding, padding_mode=args.padding_mode, mirror=False)
    return c, e.labels(), e.labels(mirrored=True), e.pid

def preprocess_examples(args, examples, include_mirrors=True):
    '''
    '''
    final_width = args.width + 2 * args.padding
    final_height = args.height + 2 * args.padding

    N = len(examples)
    NN = 2 * N if include_mirrors else N
    crops = np.zeros((NN, 3, final_width, final_height), dtype=np.uint8)
    labels = []
    labels_mirror = [] # note that the labels do differ for some attributes!
    pids = []

    if not args.single_threaded:
        with ProgressBar(max_value=len(examples)) as progress:
            pex = concurrent.futures.ProcessPoolExecutor(max_workers=None)
            for idx, results in enumerate(pex.map(do_one_crop, examples)):
                c, l, lm, p = results # crop, label, labels mirrored
                crops[idx, :] = c
                labels.append(l)
                pids.append(p)
                if include_mirrors:
                    crops[idx+N, :] = c[:, ::-1, :]
                    labels_mirror.append(lm)
                if idx % 50 == 0:
                    progress.update(idx)
    else:
        with ProgressBar(max_value=len(examples)) as progress:
            for idx, e in enumerate(examples):
                if idx % 50 == 0:
                    progress.update(idx)
                c, l, lm, p = do_one_crop(e)
                crops[idx, :] = c
                labels.append(l)
                pids.append(p)
                if include_mirrors:
                    crops[idx+N, :] = c[:, ::-1, :]
                    labels_mirror.append(lm)
    print('')
    if include_mirrors:
        labels = labels + labels_mirror
        pids   = pids + pids
    return crops, labels, pids

def write_results(args, split, crops, labels, valid_labels, attribute_names, mean, pids):
    if args.output_mode == 'npy':
        out = path.join(args.output_dir, split)
        os.makedirs(out)
        np.save(path.join(out, 'crops'), crops)
        np.save(path.join(out, 'labels'), labels)
        np.save(path.join(out, 'valid_labels'), valid_labels)
        if mean is not None:
            np.save(path.join(out, 'mean'), mean)
        print('results written as NPY files to directory: ', out)
    elif args.output_mode == 'hdf5':
        if not path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        out = path.join(args.output_dir, split + '.hdf5')
        h = h5py.File(out, 'w')
        h.create_dataset('crops', data=crops)
        # also save the pedestrianID from the database
        # this is helpful for debugging and cross-checking with other tools
        h.create_dataset('pids', data=pids)

        h.create_dataset('labels', data=labels)
        h['labels'].attrs.create('valid_values', valid_labels)
        h['labels'].attrs.create('names', attribute_names)
        if mean is not None:
            h.create_dataset('mean', data=mean)
        h.close() # write to file
        print('results written as HDF5 to: ', out)
    else:
        raise NotImplemented('output mode unknown' + args.output_mode)

def main(args):
    for split in ('train', 'val', 'test'):
        print('-'*50, '\n', 'processing split: ', split)

        annotations = PARSE(args.parse_path, attributes='all', split=split)
        if args.debug:
            examples = annotations.all_examples[:500]
        else:
            examples = annotations.all_examples

        include_mirrors = True if split =='train' and not args.no_train_mirrors else False
        crops, labels, pids = preprocess_examples(args, examples, include_mirrors=include_mirrors)
        print('extracted %d crops'%len(crops))
        mean = np.mean(crops[:len(examples), :, :, :], axis=0) if split == 'train' else None
        valid_labels = examples[0].valid_values

        write_results(args, split, crops, labels, valid_labels,
                      annotations.attributes, mean, pids)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--single_threaded', '-s', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--output_dir', '-o',
                        default=os.environ.get('OUTPUT_DIR', '/tmp/parse_crops'))
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--padding', '-p', type=int, default=32,
                        help='# additional padding after scaling to w x h')
    parser.add_argument('--output_mode',
                        choices=('npy', 'hdf5'),
                        default='hdf5',
                        help="data format of the resulting files")
    parser.add_argument('--padding_mode', '-m',
                        choices=('zero', 'edge'),
                        default='edge',
                        help='''how to pad the crops, in case the
                        bounding box is too close to the image boundaries.''')
    parser.add_argument('--no-train-mirrors', action='store_true',
                        help="do not generate mirrored examples from"
                             "train split (by default we do)")

    parser.add_argument('--parse_path',
                        default=os.getenv('PARSE_PATH',
                                          '/work/sudowe/datasets/parse/'))
    args = parser.parse_args()
    args.git_revision = git.current_revision(path.dirname(__file__))

    recreate_dir(args.output_dir)
    # save parameters to file (for reproducibility of results)
    dump_args(args, args.output_dir)
    main(args)
