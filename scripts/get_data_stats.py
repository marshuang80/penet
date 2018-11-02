import argparse
import h5py
import numpy as np
import os
import pickle
import util

from collections import defaultdict, OrderedDict
from ct.ct_head_constants import *
from tqdm import tqdm


def get_data_stats(pkl_path, use_contrast, use_aneurysm_masks, use_brain_masks, use_split, get_mean):
    """Get statistics for data with info in the given pickle file."""
    stats_dict = OrderedDict()
    with open(pkl_path, 'rb') as pkl_file:
        all_modes_series = pickle.load(pkl_file)
    all_series = [s for s in all_modes_series if (s.mode == 'contrast' if use_contrast else 'non_contrast')]
    if use_aneurysm_masks:
        all_series = [s for s in all_series if s.aneurysm_mask_path is not None]
    if use_brain_masks:
        all_series = [s for s in all_series if s.brain_mask_path is not None]
    if use_split:
        all_series = [s for s in all_series if s.phase is not None]
    stats_dict['num_series'] = len(all_series)
    stats_dict['num_aneurysm'] = sum(int(s.is_aneurysm) for s in all_series)
    stats_dict['num_normal'] = sum(int(not s.is_aneurysm) for s in all_series)

    for phase in ('train', 'val', 'test'):
        stats_dict[phase + '_set_size'] = sum(int(s.phase == phase) for s in all_series)
        for label, aneurysm in zip(('aneurysms','normals'), (True, False)):
            stats_dict[phase + '_set_' + label] = sum(int(s.phase == phase and s.is_aneurysm == aneurysm)\
                                                     for s in all_series)

    aneurysm_sizes = []
    stats_dict['year, scanner'] = defaultdict(lambda:defaultdict(int))
    stats_dict['slice_thickness'] = defaultdict(lambda: defaultdict(int))

    for label, aneurysm in zip(('aneurysms', 'normals'), (True, False)):
        for series in [s for s in all_series if s.is_aneurysm == aneurysm]:
            stats_dict['slice_thickness'][label][series.slice_thickness] += 1
            stats_dict['year, scanner'][label][(series.date.year, series.scanner_make)] += 1

            if series.is_aneurysm and series.aneurysm_size is not None:
                aneurysm_sizes.append(series.aneurysm_size)

    if get_mean:
        pixel_dict = {
            'min_val': CONTRAST_HU_MIN if use_contrast else NON_CON_HU_MIN,
            'max_val': CONTRAST_HU_MAX if use_contrast else NON_CON_HU_MAX,
        }
        means = []
        with h5py.File(os.path.join(os.path.dirname(pkl_path), 'data.hdf5'), 'r') as hdf5_fh:
            print('Getting mean HU value...')
            for s in tqdm(all_series):
                if s.phase == 'train':
                    volume = hdf5_fh[s.dset_path][...]

                    # Apply same normalization as we would during training, then compute the mean
                    volume = volume.astype(np.float32)
                    volume = (volume - pixel_dict['min_val']) / (pixel_dict['max_val'] - pixel_dict['min_val'])
                    volume = np.clip(volume, 0., 1.)

                    means.append(np.mean(volume))
        mean_hu = np.mean(means)
        stats_dict['mean_train_HU_value'] = mean_hu

    stats_dict['aneurysm_size_avg'] = np.mean(aneurysm_sizes)
    stats_dict['aneurysm_size_stdev'] = np.std(aneurysm_sizes)

    # Print out statistics
    # Print out statistics
    print('Stats for {} data at {}:'\
          .format('contrast' if use_contrast else 'non_contrast', os.path.dirname(args.pkl_path)))
    for k, v in stats_dict.items():
        if type(v) is not defaultdict:
            print('    {} -> {}'.format(k, v))
        else:
            for label, g in v.items():
                print('    {} -> {}'.format(k, label))
                for group, count in g.items():
                    print('          {}: {}'.format(group, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_mean', action='store_true',
                        help='Get the mean HU value of the training set.')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to pickle file for a study.')
    parser.add_argument('--use_contrast', type=util.str_to_bool, required=True,
                        help='If true, get stats for contrast data. If false, get stats for non_contrast data.')
    parser.add_argument('--use_aneurysm_masks', type=util.str_to_bool,
                        help='If true, get stats only for data with aneurysm mask files.')
    parser.add_argument('--use_brain_masks', type=util.str_to_bool,
                        help='If true, get stats only for data with brain mask files.')
    parser.add_argument('--use_split', type=util.str_to_bool, required=True,
                        help='If true, get stats only for data with phase assigned.')

    args = parser.parse_args()
    get_data_stats(args.pkl_path, args.use_contrast, args.use_aneurysm_masks, args.use_brain_masks, args.use_split,
                   args.get_mean)
