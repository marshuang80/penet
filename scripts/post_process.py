import argparse
import h5py
import moviepy.editor as mpy
import numpy as np
import os
import pickle
import re
import skimage.morphology as morph
import time
import util

from sklearn.datasets import dump_svmlight_file
from tqdm import tqdm


def post_process(args, series_list):
    """Post-process a directory of masks written by test.py.
    Args:
        args: Command-line arguments.
        series_list: List of `CTSeries` objects for getting metadata about masks.
    """
    # Create feature vectors for training a classifier from segmentation outputs
    feature_vectors = []
    labels = []

    mask_paths = [os.path.join(args.masks_dir, f) for f in os.listdir(args.masks_dir) if f.endswith('.npy')]
    for mask_path in tqdm(mask_paths):
        # Load mask and original series
        mask = np.load(mask_path)
        base_name = os.path.basename(mask_path)
        series_num = int(re.match(r'(\d+)_.*', base_name).group(1))
        dset_path = '/series/{:05d}'.format(series_num)

        # Get series from series list
        series = None
        for s in series_list:
            if s.dset_path is not None and s.dset_path == dset_path:
                series = s
        assert series is not None, 'Could not find series with path {}'.format(dset_path)

        if not series.is_bottom_up:
            mask = np.flipud(mask)

        if args.get_gif:
            # Create a GIF heat-map overlay with the segmentation mask
            with h5py.File(os.path.join(args.data_dir, 'data.hdf5'), 'r') as hdf5_fh:
                volume = hdf5_fh[dset_path][...]
            if not series.is_bottom_up:
                volume = np.flipud(volume)

            print('Generating heat map...')
            volume = np.expand_dims(volume, -1)
            volume = util.apply_window(volume, 40., 400.)
            volume = np.float32(volume) / 255.

            # Overlay mask and write to GIF
            volume = util.add_heat_map(volume, mask, alpha_img=0.33, color_map='binary', normalize=False)
            video_clip = mpy.ImageSequenceClip(list(volume), fps=args.gif_fps)

            output_path = os.path.join(os.path.dirname(mask_path), os.path.basename(mask_path)[:-4] + '.gif')
            print('Writing GIF...')
            video_clip.write_gif(output_path)

        if args.get_features:
            # Create features from each segmentation mask
            tick = time.time()
            feature_vector, label = get_mask_features(series, mask, max_num_features=args.max_num_features)
            feature_vectors.append(feature_vector)
            labels.append(label)

    if args.get_features:
        # Pad feature vectors to the same length
        feature_vectors = [v + [0] * (args.max_num_features - len(v)) for v in feature_vectors]

        # Make classifier inputs and labels
        inputs = np.array(feature_vectors, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # Save output
        features_path = os.path.join(args.features_dir, args.phase + '_feat_vecs.' + args.features_type)
        labels_path = os.path.join(args.features_dir, args.phase + 'labels.' + args.features_type)

        if args.features_type == 'svmlight':
            dump_svmlight_file(inputs, labels, features_path)
        elif args.features_type == 'npy':
            np.save(features_path, inputs)
            np.save(labels_path, labels)
        else:
            np.savetxt(features_path, inputs, delimiter=',')
            np.savetxt(labels_path, inputs, delimiter=',')


def get_mask_features(series, mask, threshold=0.5, max_num_features=25):
    """Get features from a series and its predicted mask.
    These features will be used to train a classifier mapping segmentation
    outputs to classification outputs.
    Args:
        series: `CTSeries` that was used as input to generate the mask.
        mask: Segmentation output mask.
        threshold: Threshold for features computed on a binary mask
        max_num_features: Maximum number of features in a feature vector.
    Returns:
        NumPy array of features, should be useful for classification.
    """

    # Threshold the mask
    preds = (mask > threshold)

    # Get the connected components after thresholding
    mask_labels = morph.label(preds)
    unique_labels, counts = np.unique(mask_labels, return_counts=True)

    # Forget the background label
    unique_labels = unique_labels[1:]
    counts = counts[1:]

    # Create list of features
    feature_vector = []
    for label, count in zip(unique_labels, counts):
        if count < 25:
            # Throw out tiny specks
            continue

        # Features for one connected component
        cc_features = CCFeatures()
        # Feature 1: Min/max position of the connected component along each axis
        for axis in (0,):
            axis_min, axis_max = util.get_range((mask_labels == label), axis=axis)

            # Scale min/max to be in [0, 1].
            if axis == 0:
                total_range = [x - 1 for x in series.brain_range]
                if axis_max < total_range[0] or axis_min > total_range[1]:
                    continue
            else:
                total_range = [0, 511]

            axis_min = (axis_min - total_range[0]) / (total_range[1] - total_range[0] + 1)
            axis_max = (axis_max - total_range[0]) / (total_range[1] - total_range[0] + 1)

            cc_features.set_axis(axis_min, axis_max, axis)

        # Feature 2: Max probability within the connected component
        cc_probs = mask[mask_labels == label]
        cc_features.max_prob = np.max(cc_probs)

        # Feature 3: Scaled probability of the connected component
        cc_features.scaled_volume = np.sum(cc_probs)

        # Feature 4: Number of pixels above the threshold in the connected component
        cc_features.volume = count

        # Add connected component to total feature vector
        feature_vector.append(cc_features)

    label = int(series.is_aneurysm)

    # Sort in descending order of max prob, then truncate
    feature_vector.sort(key=lambda cc: cc.max_prob, reverse=True)
    fv_flat = []
    for cc in feature_vector:
        fv_flat += cc.to_list()
    feature_vector = fv_flat[:max_num_features]
    print('[{}] -> {}'.format(label, feature_vector))

    return feature_vector, label


class CCFeatures(object):
    """Features for a single connected component."""
    def __init__(self):
        self.x_min = None
        self.x_max = None

        self.y_min = None
        self.y_max = None

        self.z_min = None
        self.z_max = None

        self.volume = 0.
        self.scaled_volume = 0.
        self.max_prob = 0.

    def set_axis(self, axis_min, axis_max, axis):
        if axis == 0:
            self.z_min = axis_min
            self.z_max = axis_max
        elif axis == 1:
            self.y_min = axis_min
            self.y_max = axis_max
        elif axis == 2:
            self.x_min = axis_min
            self.x_max = axis_max

    def _axes_list(self):
        """List the min/max values along each axis."""
        axes = [self.z_min, self.z_max, self.y_min, self.y_max, self.x_min, self.x_max]
        return axes

    def to_list(self):
        """List all features."""
        if self.z_min is None or self.z_max is None:
            self.z_min = 0.
            self.z_max = 0.
        lst = [self.z_min, self.z_max, self.max_prob, self.scaled_volume, self.volume]
        return lst

    def __len__(self):
        return len(self.to_list())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/pidg/cta/all',
                        help='Root directory holding HDF5 data file.')
    parser.add_argument('--masks_dir', type=str, required=True,
                        help='Directory holding NumPy mask files.')
    parser.add_argument('--gif_fps', type=int, default=8,
                        help='Frames per second in an output GIF.')
    parser.add_argument('--get_gif', type=util.str_to_bool, default=False,
                        help='If set, create a GIF overlay of the mask.')
    parser.add_argument('--get_features', type=util.str_to_bool, default=True,
                        help='If set, generate features from the segmentation mask.')
    parser.add_argument('--features_dir', type=str, default='features/',
                        help='Output directory for features.')
    parser.add_argument('--phase', type=str, default='',
                        help='Phase of outputs in masks_dir. To be used for naming feature filename.')
    parser.add_argument('--features_type', type=str, choices=('svmlight', 'npy', 'csv'), default='svmlight',
                        help='Filetype to save features in. Can be svmlight, numpy array, or csv.')
    parser.add_argument('--max_num_features', type=int, default=25,
                        help='Maximum number of features in a feature vector.')

    args_ = parser.parse_args()

    os.makedirs(args_.features_dir, exist_ok=True)

    if args_.get_features and args_.phase == '':
        raise ValueError('Must specify phase of data in mask_dir if get_features is true.')

    with open(os.path.join(args_.data_dir, 'series_list.pkl'), 'rb') as pkl_fh:
        series_list_ = pickle.load(pkl_fh)
    post_process(args_, series_list_)
