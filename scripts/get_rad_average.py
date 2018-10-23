import argparse
import json
import os
import pandas as pd
import pickle
import util

from collections import OrderedDict
from sklearn.metrics import *


def get_rad_average(args):
    """Get micro-average statistics for radiologist performance."""
    stats_dict = OrderedDict()
    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)
    series_list = [s for s in series_list if s.phase == 'test']
    csv_list = [os.path.join(args.csv_dir, c) for c in os.listdir(args.csv_dir) if c.endswith('.csv')]
    accs = []
    rad_labels = []
    for csv in csv_list:
        if args.only_gen_rads:
            if 'sur' in os.path.basename(csv) or 'res' in os.path.basename(csv):
                continue
        rad_df = pd.read_csv(csv, dtype={'Accession Number': object,
                                                   'Label': int})
        accs += rad_df['Accession Number'].tolist()
        rad_labels += rad_df['Label'].tolist()

    true_labels = []
    for acc in accs:
        for series in series_list:
            if str(series.accession_number) == acc:
                true_labels.append(int(series.is_aneurysm))

    stats_dict['Accuracy'] = accuracy_score(true_labels, rad_labels)
    stats_dict['Precision'] = precision_score(true_labels, rad_labels)
    stats_dict['Recall'] = recall_score(true_labels, rad_labels)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(true_labels, rad_labels).ravel()
    stats_dict['Specificity'] = true_negative / (true_negative + false_positive)

    model_augment = True if os.path.basename(args.csv_dir) == 'with_model' else False

    # Print out statistics
    print('Stats for micro-average radiologist labels {} model augmentation:'.format('with' if model_augment else 'without'))
    print('Incorrect: ', false_negative + false_positive)
    print('False positive: ', false_positive)
    print('False negative: ', false_negative)
    for k, v in stats_dict.items():
        print('{}: {:.4f}'.format(k, v))

    filename = 'avg_gen_rads.json' if args.only_gen_rads else 'avg_experts.json'
    with open(os.path.join(args.csv_dir, filename), 'w') as outfile:
        json.dump(stats_dict, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='/data3/HEAD-CT-0830/all/series_list.pkl',
                        help='Path to pickle file for a study.')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Path to csvs with radiologist labels.')
    parser.add_argument('--only_gen_rads', type=util.str_to_bool, default=True,
                        help='If true, include only general rads. If not, include residents and neurosurgeons.')

    args = parser.parse_args()
    get_rad_average(args)
