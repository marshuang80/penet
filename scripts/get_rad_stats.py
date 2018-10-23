import argparse
import datetime
import dateutil.parser as date_parser
import json
import numpy as np
import os
import pandas as pd
import pickle
import util

from collections import OrderedDict
from math import sqrt
from sklearn.metrics import *

Z_VALUE = 1.96 # 95% Confidence Interval


def get_rad_stats(args):
    """Get statistics for radiologist performance."""
    stats_dict = OrderedDict()
    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)
    series_list = [s for s in series_list if s.phase == 'test']
    rad_df = pd.read_csv(args.csv_path, dtype={'Accession Number': object,
                                               'Label': int,
                                               'Timestamp': object})
    accs = rad_df['Accession Number'].tolist()
    rad_labels = rad_df['Label'].tolist()
    timestamps = [date_parser.parse(t) for t in rad_df['Timestamp'].tolist()]

    true_labels = []
    aneurysm_sizes = []
    for acc in accs:
        for series in series_list:
            if str(series.accession_number) == acc:
                true_labels.append(int(series.is_aneurysm))
                if series.is_aneurysm:
                    aneurysm_sizes.append(series.aneurysm_size)
                else:
                    aneurysm_sizes.append(0)

    stats_dict['Accuracy'] = accuracy_score(true_labels, rad_labels)
    stats_dict['Precision'] = precision_score(true_labels, rad_labels)
    stats_dict['Recall'] = recall_score(true_labels, rad_labels)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(true_labels, rad_labels).ravel()
    stats_dict['Specificity'] = true_negative / (true_negative + false_positive)

    if args.get_confidence_interval:
        specificity = true_negative / (false_positive + true_negative)
        stats_dict['Specificity_CI'] = \
            wilson_confidence_interval(specificity, Z_VALUE, false_positive + true_negative)
        sensitivity = true_positive / (true_positive + false_negative)
        stats_dict['Sensitivity_CI'] = \
            wilson_confidence_interval(sensitivity, Z_VALUE, true_positive + false_negative)
        accuracy = (true_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)
        stats_dict['Accuracy_CI'] = \
            wilson_confidence_interval(accuracy, Z_VALUE, (true_positive + false_negative + false_positive + true_negative))

    # Print out statistics
    print('Stats for radiologist labels at {}:'.format(args.csv_path))
    print('Incorrect: ', false_negative + false_positive)
    print('False positive: ', false_positive)
    print('False negative: ', false_negative)

    if args.get_confidence_interval:
        print('Precision: {:.4f}'.format(stats_dict['Precision']))
        print('Accuracy: {:.4f} ({:.4f}, {:.4f})'.format(stats_dict['Accuracy'], stats_dict['Accuracy_CI'][0],
                                             stats_dict['Accuracy_CI'][1]))
        print('Specificity: {:.4f} ({:.4f}, {:.4f})'.format(stats_dict['Specificity'], stats_dict['Specificity_CI'][0],
                                             stats_dict['Specificity_CI'][1]))
        print('Sensitivity: {:.4f} ({:.4f}, {:.4f})'.format(stats_dict['Recall'], stats_dict['Sensitivity_CI'][0],
                                             stats_dict['Sensitivity_CI'][1]))
    else:
        for k, v in stats_dict.items():
            print('{}: {:.4f}'.format(k, v))

    if args.print_missed_examples:
        false_positive_examples = [accs[i] for i, (t, r) in enumerate(zip(true_labels, rad_labels)) if t == 0 and r == 1]
        false_negative_examples = [accs[i] for i, (t, r) in enumerate(zip(true_labels, rad_labels)) if t == 1 and r == 0]
        print('Accession number for {} false positives: {}'.format(len(false_positive_examples), false_positive_examples))
        print('Accession number for {} false negatives: {}'.format(len(false_negative_examples), false_negative_examples))

    if args.print_time_analysis:
        time_diffs = []
        for i in range(len(timestamps) - 1):
            time_diffs.append((timestamps[i + 1] - timestamps[i]).total_seconds())

        normal_label_times = [time for time, label in zip(time_diffs, rad_labels) if label == 0]
        aneurysm_label_times = [time for time, label in zip(time_diffs, rad_labels) if label == 1]
        true_normal_times = [time for time, truth in zip(time_diffs, true_labels) if truth == 0]
        true_aneurysm_times = [time for time, truth in zip(time_diffs, true_labels) if truth == 1]

        print('Time spent: Reader\'s Label')
        print('    -Total time spent: aneurysm: {} | normal: {} | total: {}'.format(
            str(datetime.timedelta(seconds=sum(aneurysm_label_times))), str(datetime.timedelta(seconds=sum(normal_label_times))),
            str(datetime.timedelta(seconds=sum(aneurysm_label_times + normal_label_times)))))
        print('    -Average time spent: aneurysm: {} | normal: {} | total: {}'.format(
            str(datetime.timedelta(seconds=int(np.mean(aneurysm_label_times)))), str(datetime.timedelta(seconds=int(np.mean(normal_label_times)))),
            str(datetime.timedelta(seconds=int(np.mean(aneurysm_label_times + normal_label_times))))))
        print('Time spent: True Label')
        print('    -Total time spent: aneurysm: {} | normal: {} | total: {}'.format(
            str(datetime.timedelta(seconds=sum(true_aneurysm_times))), str(datetime.timedelta(seconds=sum(true_normal_times))),
            str(datetime.timedelta(seconds=sum(true_aneurysm_times + true_normal_times)))))
        print('    -Average time spent: aneurysm: {} | normal: {} | total: {}'.format(
            str(datetime.timedelta(seconds=int(np.mean(true_aneurysm_times)))), str(datetime.timedelta(seconds=int(np.mean(true_normal_times)))),
            str(datetime.timedelta(seconds=int(np.mean(true_aneurysm_times + true_normal_times))))))

    filename = os.path.basename(args.csv_path).split('_')
    folder = 'without_model' if filename[1] == 'wo' else 'with_model'
    with open(os.path.join(args.output_dir, folder, str(filename[0] + '.json')), 'w') as outfile:
        json.dump(stats_dict, outfile)


def wilson_confidence_interval(p_hat, z, n):
    mid_point = 2 * n * p_hat + z ** 2
    interval = z * sqrt(z ** 2 + 4 * n * p_hat * (1 - p_hat))
    lower_bound = (mid_point - interval) / (2 * (n + z ** 2))
    upper_bound = (mid_point + interval) / (2 * (n + z ** 2))
    return lower_bound, upper_bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='/data3/HEAD-CT-0830/all/series_list.pkl',
                        help='Path to pickle file for a study.')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to csv with radiologist labels.')
    parser.add_argument('--print_missed_examples', type=util.str_to_bool,
                        help='Print accession numbers of missed examples.')
    parser.add_argument('--print_time_analysis', type=util.str_to_bool,
                        help='Print analysis of timestamps. Assumes labeling was done in one sitting.')
    parser.add_argument('--output_dir', type=str, default='/home/ajypark/hct_rad_performance',
                        help='Directory to save radiologist performance.')
    parser.add_argument('--get_confidence_interval', type=util.str_to_bool,
                        help='Print confidence intervals for metrics. Uses Wilson\'s score.')

    args = parser.parse_args()
    get_rad_stats(args)
