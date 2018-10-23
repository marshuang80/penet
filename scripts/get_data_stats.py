import argparse
import h5py
import numpy as np
import os
import pickle
import util

from collections import defaultdict, OrderedDict, Iterable
from ct.ct_head_constants import *
from tqdm import tqdm


def get_data_stats(args):
    """Get statistics for data with info in the given pickle file."""
    stats_dict = OrderedDict()
    with open(args.pkl_path, 'rb') as pkl_file:
        all_modes_series = pickle.load(pkl_file)
    series_list = [s for s in all_modes_series if include_series(s)]

    # Get stats for dataset specified in args
    if args.use_aneurysm_masks:
        series_list = [s for s in series_list if s.aneurysm_mask_path is not None]
    if args.use_brain_masks:
        series_list = [s for s in series_list if s.brain_mask_path is not None]
    if args.use_split:
        series_list = [s for s in series_list if s.phase is not None]
    if len(args.phase) > 0:
        series_list = [s for s in series_list if s.phase == args.phase]

    # Get label distribution for total and each phase
    stats_dict['num_series'] = len(series_list)
    stats_dict['num_aneurysm'] = sum(int(s.is_aneurysm) for s in series_list)
    stats_dict['num_normal'] = sum(int(not s.is_aneurysm) for s in series_list)

    for phase in ('train', 'val_{}'.format(args.val_split), 'test'):
        stats_dict[phase + '_set_size'] = sum(int(is_training_phase(s.phase, args.val_split) if phase == 'train'
                                                  else s.phase == phase) for s in series_list)
        stats_dict[phase + '_num_patients'] = len(set([s.medical_record_number for s in series_list
                                                       if (is_training_phase(s.phase, args.val_split) if phase == 'train'
                                                           else s.phase == phase)]))
        stats_dict[phase + '_num_females'] = 0
        ages = []
        age_unidentified = 0
        for s in series_list:
            if is_training_phase(s.phase, args.val_split) if phase == 'train' else s.phase == phase:
                dcm = util.read_dicom(os.path.join(s.dcm_dir, os.listdir(s.dcm_dir)[0]))
                if str(dcm.PatientSex) == 'F':
                    stats_dict[phase + '_num_females'] += 1
                age = int(str(dcm.PatientAge).strip('Y')) if str(dcm.PatientAge).endswith('Y') else int(dcm.PatientAge)
                if age < 150:
                    ages.append(age)
                else:
                    age_unidentified += 1
        stats_dict[phase + '_min_age'] = min(ages)
        stats_dict[phase + '_max_age'] = max(ages)
        stats_dict[phase + '_mean_age'] = np.mean(ages)
        stats_dict[phase + '_age_stdev'] = np.std(ages)
        stats_dict[phase + '_age_unidentified'] = age_unidentified

        for label, aneurysm in zip(('aneurysms', 'normals'), (True, False)):
            stats_dict[phase + '_set_' + label] = sum(int((is_training_phase(s.phase, args.val_split) if phase == 'train'
                                                  else s.phase == phase) and s.is_aneurysm == aneurysm) for s in series_list)

    # Get year, scanner manufacturer, slice thickness, and aneurysm size distribution
    aneurysm_sizes = []
    stats_dict['year, scanner'] = defaultdict(lambda:defaultdict(int))
    stats_dict['slice_thickness'] = defaultdict(lambda: defaultdict(int))

    for label, is_aneurysm in zip(('aneurysms', 'normals'), (True, False)):
        for series in [s for s in series_list if s.is_aneurysm == is_aneurysm]:
            stats_dict['slice_thickness'][label][series.slice_thickness] += 1
            stats_dict['year, scanner'][label][(series.date.year, series.scanner_make)] += 1

            if series.is_aneurysm and series.aneurysm_size is not None:
                aneurysm_sizes.append(series.aneurysm_size)

    stats_dict['aneurysm_size_avg'] = np.mean(aneurysm_sizes)
    stats_dict['aneurysm_size_stdev'] = np.std(aneurysm_sizes)

    # Get mean Hounsfield Unit value
    if args.get_mean:
        pixel_dict = {
            'min_val': CONTRAST_HU_MIN if args.use_contrast else NON_CON_HU_MIN,
            'max_val': CONTRAST_HU_MAX if args.use_contrast else NON_CON_HU_MAX,
        }
        means = []
        with h5py.File(os.path.join(os.path.dirname(args.pkl_path), 'data.hdf5'), 'r') as hdf5_fh:
            print('Getting mean HU value...')
            for s in tqdm(series_list):
                if s.phase == 'train':
                    volume = hdf5_fh[s.dset_path][...]

                    # Apply same normalization as we would during training, then compute the mean
                    volume = volume.astype(np.float32)
                    volume = (volume - pixel_dict['min_val']) / (pixel_dict['max_val'] - pixel_dict['min_val'])
                    volume = np.clip(volume, 0., 1.)

                    means.append(np.mean(volume))
        mean_hu = np.mean(means)
        stats_dict['mean_train_HU_value'] = mean_hu

    # Print out statistics
    print('Stats for {} data at {}:'\
          .format('contrast' if args.use_contrast else 'non_contrast', os.path.dirname(args.pkl_path)))
    for k, v in stats_dict.items():
        if type(v) is not defaultdict:
            print('    {} -> {}'.format(k, v))
        else:
            for label, g in v.items():
                print('    {} -> {}'.format(k, label))
                for group in sorted(g.keys(), key=lambda x: (x[0], x[1]) if isinstance(x, Iterable) else x):
                    count = g[group]
                    print('          {}: {}'.format(group, count))


def include_series(s):
    """Apply same rules that determine loading a series at train time."""
    if s.phase not in ['train', 'val_1', 'val_2', 'val_3', 'test']:
        return False

    if s.phase == 'train':
        # In training mode, need all aneurysm studies to be well formed
        if s.is_aneurysm and s.aneurysm_mask_path is None:
            return False

    if s.slice_thickness not in [1.0, 1.25]:
        return False

    if s.mode != 'contrast':
        return False

    return True


def is_training_phase(phase, val_split):
    possible_splits = [1, 2, 3]
    if phase == 'train':
        return True
    for s in [s for s in possible_splits if s != val_split]:
        if phase == 'val_{}'.format(s):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_mean', action='store_true',
                        help='Get the mean HU value of the training set.')
    parser.add_argument('--pkl_path', type=str, default='/data3/HEAD-CT-0830/all/series_list.pkl',
                        help='Path to pickle file for a study.')
    parser.add_argument('--use_contrast', type=util.str_to_bool, default=True,
                        help='If true, get stats for contrast data. If false, get stats for non_contrast data.')
    parser.add_argument('--use_aneurysm_masks', type=util.str_to_bool,
                        help='If true, get stats only for data with aneurysm mask files.')
    parser.add_argument('--use_brain_masks', type=util.str_to_bool,
                        help='If true, get stats only for data with brain mask files.')
    parser.add_argument('--use_split', type=util.str_to_bool, default=True,
                        help='If true, get stats only for data with phase assigned.')
    parser.add_argument('--phase', type=str, default='',
                        help='If specified, get stats only for data for that phase.')
    parser.add_argument('--val_split', type=int, choices=(0, 1, 2, 3), required=True,
                        help='Split number for validation set. Zero means None, non-zero specifies a split.')

    args = parser.parse_args()
    get_data_stats(args)
