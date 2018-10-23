"""Split converted dataset into train/val/test"""
import argparse
import json
import os
import pandas as pd
import pickle
import random
import util

from collections import defaultdict


def assign_to_train(args):
    """Assign phase 'train' to all series in pkl file."""
    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)

    for s in series_list:
        s.phase = 'train'

    # Write summary file for all series
    util.print_err('Dumping pickle file...')
    with open(os.path.join(args.output_dir, 'series_list.pkl'), 'wb') as pkl_file:
        pickle.dump(series_list, pkl_file)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(args.output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in series_list], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)


def split(args):
    """Split data into train, val, test."""
    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)

    mode_list = [s for s in series_list if s.mode == ('contrast' if args.use_contrast else 'non_contrast')]
    ann_df = pd.read_csv(args.csv_path)
    duplicate_patients = set(ann_df.duplicated(subset='MRN', keep=False).tolist())
    random.seed('head-ct')

    if args.initialize_phase:
        for s in mode_list:
            s.phase = None

    if args.split_method == 'random':
        random_split(mode_list, args, duplicate_patients)

    elif args.split_method == 'stratify':
        stratify(mode_list, args, duplicate_patients)

    elif args.split_method == 'fix_thickness':
        fixed_list = [s for s in mode_list if s.slice_thickness in args.fix_thickness]
        random_split(fixed_list, args, duplicate_patients)

    # Write summary file for all series
    util.print_err('Dumping pickle file...')
    with open(os.path.join(args.output_dir, 'series_list.pkl'), 'wb') as pkl_file:
        pickle.dump(series_list, pkl_file)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(args.output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in series_list], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)


def cannot_stratify(label_series_dict):
    """Check if stratification category contains no series of a particular label.

    Args:
        label_series_dict: Dictionary containing labels as keys and corresponding series list

    Returns:
        True if category is missing series in one or more labels.
    """
    for label, series in label_series_dict.items():
        if len(series) == 0:
            return True
    return False


def assign_by_stratification(stratified_series, phase, num_phase):
    """Assign to test or val set evenly among stratification groups and labels.

    Args:
        stratified_series: Dictionary containing {label: corresponding series list}
        phase: Phase to assign to, must be 'val' or 'test'
        num_phase: Number of examples in the phase set.
    """
    while num_phase > 0:
        for group, label_series_dict in stratified_series.items():
            for label, series in label_series_dict.items():
                s = series.pop(random.randrange(0, len(series)))
                s.phase = phase
                num_phase -= 1


# TODO: Finish implementing stratification split.
def stratify(mode_list, args, duplicate_patients):
    """Record counts for each stratification category, for each label (aneurysm/normal).

    Args:
        mode_list: List containing all contrast/non_contrast series in the pkl file.
        args: Command line arguments.
        duplicate_patients: List containing medical record numbers of patients that have multiple scans in the dataset.
    """
    # Maps (tuple of attribute values) -> {(series label): (list of series with those attribute values)}
    # E.g. ('siemens', 2017) -> {[0]: [series_00001, series_00024, ..., series_00049],
    #                            [1]: [series_00003, series_00017, ..., series_00121]}
    stratified_series = defaultdict(lambda: defaultdict(list))
    normals_list = [s for s in mode_list if not s.is_aneurysm]
    aneurysms_list = [s for s in mode_list if s.is_aneurysm]

    for s in mode_list:
        # Stratify on scanner make and year, also keep labels to control distribution in val / test set
        group = (s.scanner_make, s.date.year)
        label = int(s.is_aneurysm)
        stratified_series[group][label].append(s)

    # Remove categories that cannot be stratified
    for group, label_series_dict in stratified_series.items():
        if cannot_stratify(label_series_dict):
            stratified_series.pop(group)

    # Make the train, val, test split
    num_val_label = int(args.split_proportion['val'] * len(mode_list)) // 2
    num_val = num_val_label * 2
    num_test_label = int(args.split_proportion['test'] * len(mode_list)) // 2
    num_test = num_test_label * 2
    num_train_normals = len(normals_list) - (num_val_label + num_test_label)
    num_train_aneurysms = len(aneurysms_list) - (num_val_label + num_test_label)

    # Assign to test first to ensure stratification, then val to get similar distribution
    # If number of examples for each category is not enough for perfect stratification, randomly add from train
    raise NotImplementedError('Dataset cannot be stratified. To be removed when we convert more normals.')


def is_unusual(series, args, duplicate_patients):
    """Check if series should be assigned to the training set.

    Args:
        series: CTSeries object to check.
        args: Command line arguments.
        duplicate_patients: List containing medical record numbers of patients that have multiple scans in the dataset.

    Returns:
         True if series doesn't have ideal slice thickness,
         or if the patient corresponding to the series has multiple studies in the dataset.
    """
    if args.use_contrast and series.slice_thickness not in args.contrast_thicknesses:
        return True
    if not args.use_contrast and series.slice_thickness not in args.non_contrast_thicknesses:
        return True
    if str(series.date.year) in args.non_ideal_years:
        return True
    if series.medical_record_number in duplicate_patients:
        return True
    return False


def random_split(mode_list, args, duplicate_patients):
    """Splits set of series randomly 80/10/10 into train, val, and test sets.
    Does not stratify by year or scanner manufacturer.

    Args:
        mode_list: List containing all contrast/non_contrast series in the pkl file.
        args: Command line arguments.
        duplicate_patients: List containing medical record numbers of patients that have multiple scans in the dataset.
    """
    for series in mode_list:
        if is_unusual(series, args, duplicate_patients):
            series.phase = 'train'

    aneurysms_list = [s for s in mode_list if s.is_aneurysm]
    normals_list = [s for s in mode_list if not s.is_aneurysm]

    for phase in ('test', 'val'):
        for label_list in [normals_list, aneurysms_list]:
            remaining = [s for s in label_list if s.phase == None]
            if len(remaining) < int(args.split_proportion[phase]/2 * len(mode_list)):
                util.print_err('Not enough valid examples to go in {}...'.format(phase))
                util.print_err('Filling it up with random examples from train.')
                train_list = [s for s in label_list if s.phase == 'train']
                remaining.extend(random.sample(train_list,
                                               int(args.split_proportion[phase]/2 * len(mode_list) - len(remaining))))
            for series in random.sample(remaining, int(args.split_proportion[phase]/2 * len(mode_list))):
                series.phase = phase

    remaining = [s for s in mode_list if s.phase == None]
    for series in remaining:
        series.phase = 'train'


def hold_out(args):
    """Holds out aneurysm studies from specified years for the test set."""
    annotations = pd.read_csv(args.csv_path, dtype={'Acc': object, 'MRN': object})
    patients = set(annotations['MRN'].tolist())

    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)

    for s in series_list:
        s.phase = None

    aneurysm_list = [s for s in series_list if s.is_aneurysm]
    normal_list = [s for s in series_list if not s.is_aneurysm]

    # Put aneurysms from hold out years in test set
    for series in aneurysm_list:
        if series.date.year in args.hold_out_years:
            series.phase = 'test'

    test_mrns = set([a.medical_record_number for a in aneurysm_list if a.phase == 'test'])
    duplicate_mrns = []
    for patient in patients:
        for series in aneurysm_list:
            if series.medical_record_number == patient:
                if patient in test_mrns and series.date.year not in args.hold_out_years:
                    duplicate_mrns.append(patient)

    test_aneurysms = [a for a in aneurysm_list if a.phase == 'test']

    normal_hold_out = [n for n in normal_list if n.date.year in args.hold_out_years]
    normal_test = random.sample(normal_hold_out, len(test_aneurysms))
    for series in normal_test:
        series.phase = 'test'

    # Split rest into train and val
    aneurysm_list = [a for a in aneurysm_list if a.date.year not in args.hold_out_years]
    normal_list = [n for n in normal_list if n.date.year not in args.hold_out_years]

    for label_list in [normal_list, aneurysm_list]:
        remaining_mrns = set([l.medical_record_number if l.medical_record_number != 'nan' else l.accession_number
                              for l in label_list if l.phase is None])
        val_set = random.sample(remaining_mrns, len(test_aneurysms))
        for series in label_list:
            if series.medical_record_number in val_set or series.accession_number in val_set:
                series.phase = 'val'
            else:
                series.phase = 'train'

    # Exclude test set examples with duplicate patients
    for series in [a for a in aneurysm_list if a.phase == 'test']:
        if series.medical_record_number in duplicate_mrns:
            series.phase = None

    # Write summary file for all series
    util.print_err('Dumping pickle file...')
    with open(os.path.join(args.output_dir, 'series_list.pkl'), 'wb') as pkl_file:
        pickle.dump(series_list, pkl_file)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(args.output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in series_list], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to pickle file for a study.')
    parser.add_argument('--csv_path', type=str, default='/data3/CTA/annotations/annotation.csv',
                        help='Path to CSV file (for finding duplicate patients only).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for JSON and Pickle files.')
    parser.add_argument('--all_to_train', type=util.str_to_bool, default=False,
                        help='If true, all the series in the pkl file go into the training set. \
                              To be used only for adding new sets of data to the training set.')
    parser.add_argument('--split_method', type=str, choices=('random', 'stratify', 'fix_thickness', 'hold_out'),
                        required=True,
                        help='Method to use for splitting series. If "random", does not check for study year or scanner.\
                              If "stratify", checks for study year and scanner.\
                              If "fix_thickness", fixes slice thicknesses for train and val set. \
                              Assigns phase = None to different slice thicknesses. \
                              If "hold_out", holds out specific years to test set \
                              (for checking distribution differences only.)')
    parser.add_argument('--contrast_thicknesses', type=str, default='1.0,1.25',
                        help='Ideal contrast thicknesses. If stratify is true, put all contrast studies of other \
                              thicknesses into the training set.')
    parser.add_argument('--non_contrast_thicknesses', type=str, default='2.5,3.0',
                        help='Ideal non-contrast thicknesses. If stratify is true, put all contrast studies of other \
                              thicknesses into the training set.')
    parser.add_argument('--non_ideal_years', type=str, default='1',
                        help='All series corresponding to these years go into the training set.')
    parser.add_argument('--initialize_phase', type=util.str_to_bool, required=True,
                        help='All series get phases reassigned. Existing phases will be overwritten.')
    parser.add_argument('--use_contrast', type=util.str_to_bool, default=True,
                        help='Flag to choose contrast/non_contrast series.')
    parser.add_argument('--split_proportion', type=str, default='0.8,0.1,0.1',
                        help='Proportion of train/val/test set size. Must add up to 1. \
                              (Only for split methods "random" and "stratify".)')
    parser.add_argument('--fix_thickness', type=str, default='1.25',
                        help='Slice thicknesses to fix.')
    parser.add_argument('--hold_out_years', type=str, default='2016',
                        help='Years to put in the test set.')

    args_ = parser.parse_args()

    # Get ideal thicknesses and non-ideal years
    args_.contrast_thicknesses = util.args_to_list(args_.contrast_thicknesses, allow_empty=False, arg_type=float)
    args_.non_contrast_thicknesses = util.args_to_list(args_.non_contrast_thicknesses, allow_empty=False,
                                                       arg_type=float)
    args_.non_ideal_years = util.args_to_list(args_.non_ideal_years, allow_empty=True, arg_type=int)
    args_.fix_thickness = util.args_to_list(args_.fix_thickness, allow_empty=False, arg_type=float)
    args_.hold_out_years = util.args_to_list(args_.hold_out_years, allow_empty=True, arg_type=int)

    proportion = util.args_to_list(args_.split_proportion, allow_empty=False, arg_type=float)
    assert sum(proportion) == 1.0, 'Split proportion does not add up to 1.'
    args_.split_proportion = {'train': proportion[0], 'val': proportion[1], 'test': proportion[2]}

    print(json.dumps(vars(args_), indent=4, sort_keys=True))

    if args_.all_to_train:
        assign_to_train(args_)
    elif args_.split_method == 'hold_out':
        hold_out(args_)
    else:
        split(args_)
