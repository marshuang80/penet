"""Print data statistics needed for the paper."""
import argparse
import numpy as np
import os
import pickle
import util

from collections import defaultdict, Iterable


def get_data_stats(args):
    """Get statistics for data with info in the given pickle file."""
    stats_dict = {}
    with open(args.pkl_path, 'rb') as pkl_file:
        all_modes_series = pickle.load(pkl_file)
    series_list = [s for s in all_modes_series if include_series(s)]

    # Get stats for dataset specified in args
    if len(args.phase) > 0:
        series_list = [s for s in series_list if s.phase == args.phase]

    # Get label distribution for total and each phase
    stats_dict['num_studies'] = len(series_list)
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

    # Get year, scanner manufacturer
    stats_dict['year, scanner'] = defaultdict(lambda:defaultdict(int))
    stats_dict['slice_thickness'] = defaultdict(lambda: defaultdict(int))

    for label, is_aneurysm in zip(('aneurysms', 'normals'), (True, False)):
        for series in [s for s in series_list if s.is_aneurysm == is_aneurysm]:
            stats_dict['slice_thickness'][label][series.slice_thickness] += 1
            stats_dict['year, scanner'][label][(series.date.year, series.scanner_make)] += 1

    # Print out statistics
    print('Stats for data at {}:'\
          .format(os.path.dirname(args.pkl_path)))
    for phase in ('train', 'val_{}'.format(args.val_split), 'test'):
        for key in ['_set_size', '_num_patients']:
            print('{}: {}'.format(phase + key, stats_dict[phase + key]))
        for key in ['_num_females', '_set_aneurysms']:
            print('{}: {} ({:.2f}%)'.format(phase + key, stats_dict[phase + key],
                                          stats_dict[phase + key] / stats_dict[phase + '_set_size'] * 100))
        for key in ['_mean_age', '_age_stdev']:
            print('{}: {:.2f}'.format(phase + key, stats_dict[phase + key]))


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
    parser.add_argument('--pkl_path', type=str, default='/data3/HEAD-CT-0830/all/series_list.pkl',
                        help='Path to pickle file for the studies.')
    parser.add_argument('--phase', type=str, default='',
                        help='If specified, get stats only for data for that phase.')
    parser.add_argument('--val_split', type=int, choices=(0, 1, 2, 3), required=True,
                        help='Split number for validation set. Zero means None, non-zero specifies a split.')

    args = parser.parse_args()
    get_data_stats(args)
