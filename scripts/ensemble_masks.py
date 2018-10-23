"""Ensemble masks by averaging them."""

import argparse
import numpy as np
import os
import re

from collections import defaultdict
from tqdm import tqdm


def merge(input_dirs, output_dir):
    if os.path.exists(output_dir):
        raise ValueError('Output directory {} already exists. Converted dirs should be combined manually.'
                         .format(output_dir))
    os.makedirs(output_dir)

    # Collect segmentation outputs from each directory
    series_id_re = re.compile(r'(\d{5})_prob_(0\.\d{4})')
    series2paths = defaultdict(list)
    series2probs = defaultdict(list)
    for input_dir in input_dirs:
        file_names = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        for file_name in file_names:
            name_match = series_id_re.match(file_name)
            series_id = name_match.group(1)
            series2paths[series_id].append(os.path.join(input_dir, file_name))
            series2probs[series_id].append(float(name_match.group(2)))

    # Average each output and write the results to disk
    for series_id in tqdm(series2paths):
        # Average all masks
        paths = series2paths[series_id]
        ensemble_mask = np.load(paths[0])
        for path in paths[1:]:
            ensemble_mask += np.load(path)
        ensemble_mask /= len(paths)

        # Write to disk
        avg_prob = np.mean(series2probs[series_id]).item()
        np.save(os.path.join(output_dir, '{}_ensemble_{}.npy'.format(series_id, avg_prob)), ensemble_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dirs', nargs='*', type=str,
                        help='Paths for pickle files to merge.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for merged pickle and json file.')

    args = parser.parse_args()

    merge(args.input_dirs, args.output_dir)
