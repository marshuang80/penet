"""Merge dataset directories that have already been converted using convert.py."""

import argparse
import json
import os
import pickle
import shutil
import util


def merge(pkl_paths, output_dir):
    if os.path.exists(output_dir):
        raise ValueError('Output directory {} already exists. Converted dirs should be combined manually.'
                         .format(output_dir))
    os.makedirs(output_dir)

    # Collect studies from annotation file
    all_series = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as pkl_file:
            add_series = pickle.load(pkl_file)
            all_series += add_series

    # Check for duplicate series
    unique_series = {(s.study_name, s.series_number) for s in all_series}
    if len(unique_series) < len(all_series):
        raise RuntimeError('Found duplicate series in directories {}.'.format(', '.join(pkl_paths)))

    # Write summary file for all series
    util.print_err('Dumping pickle file...')
    with open(os.path.join(output_dir, 'series_list.pkl'), 'wb') as pkl_file:
        pickle.dump(all_series, pkl_file)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in all_series], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('pkl_paths', nargs='*', type=str,
                        help='Paths for pickle files to merge.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for merged pickle and json file.')

    args = parser.parse_args()

    merge(args.pkl_paths, args.output_dir)
