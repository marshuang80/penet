"""Get the mean voxel from the training set."""

import argparse
import h5py
import numpy as np
import os
import pickle
import util

from ct.ct_pe_constants import CONTRAST_HU_MIN, CONTRAST_HU_MAX
from tqdm import tqdm


def main(args):
    with open(os.path.join(args.data_dir, 'series_list.pkl'), 'rb') as pkl_fh:
        calcium_scores = pickle.load(pkl_fh)
    hdf5_fh = h5py.File(os.path.join(args.data_dir, 'data.hdf5'), 'r+')

    means = []
    stds = []
    for calcium_score in tqdm(calcium_scores):
        scan = hdf5_fh[str(calcium_score.study_num)][...]
        scan = scan.astype(np.float32) + args.hu_intercept
        hdf5_fh[str(calcium_score.study_num)][...] = scan

        scan = (scan - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
        scan = np.clip(scan, 0., 1.)
        means.append(np.mean(scan))
        stds.append(np.std(scan))

    print('Pixel mean: {}'.format(np.mean(means)))
    print('Pixel std: {}'.format(np.mean(stds)))

    hdf5_fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/cc_data',
                        help='Base data dir with data.hdf5 and data.pkl files.')

    parser.add_argument('--do_shift', type=util.str_to_bool, default=False,
                        help='If True, shift the HU values by hu_intercept.')
    parser.add_argument('--hu_intercept', type=float, default=-1024.,
                        help='Intercept to add to all pixel values. Will modify the HDF5 file.')

    main(parser.parse_args())
