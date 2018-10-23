"""Ensemble classification predictions by concatenating the max from each."""

import argparse
import numpy as np
import os
import util


def merge(input_dirs, output_dir):
    if os.path.exists(output_dir):
        raise ValueError('Output directory {} already exists. Converted dirs should be combined manually.'
                         .format(output_dir))
    os.makedirs(output_dir)

    # Load input features and labels
    features, labels = [], []
    for input_dir in input_dirs:
        features_path = os.path.join(input_dir, 'inputs.npy')
        features.append(np.load(features_path))

        labels_path = os.path.join(input_dir, 'labels.npy')
        labels.append(np.load(labels_path))

    # Validate
    num_examples = features[0].shape[0]
    for i in range(1, len(features)):
        if features[i].shape[0] != num_examples or labels[i].shape[0] != num_examples:
            raise RuntimeError('Unexpected number of examples at index {} (expected to have {}).'
                               .format(i, features[0].shape[0]))
    for i in range(1, len(labels)):
        if np.any(labels[0] != labels[i]):
            raise RuntimeError('Unexpected labels at index {} (expected to match labels at index 0).'.format(i))

    # Combine features by taking max and concatenating
    merged_features = []
    for i in range(num_examples):
        # Take mean of features
        example_features = features[0][i]
        for j in range(1, len(features)):
            example_features += features[j][i]
        example_features /= len(features)
        merged_features.append(example_features)
    merged_features = np.array(merged_features)
    merged_labels = labels[0]  # All label arrays were verified to be the same

    # Save merged features and labels
    features_path = os.path.join(output_dir, 'inputs.npy')
    np.save(features_path, merged_features)

    labels_path = os.path.join(output_dir, 'labels.npy')
    np.save(labels_path, merged_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dirs', nargs='*', type=str,
                        help='Paths for dirs containing XGBoost input features to merge.')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory for merged XGBoost features and labels files.')
    parser.add_argument('--add_features', type=util.str_to_bool, default=False,
                        help='Add min, max, std of all predictions.')

    args = parser.parse_args()

    merge(args.input_dirs, args.output_dir)
