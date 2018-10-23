"""Update brain bouding box attribute."""
import argparse
import h5py
import json
import os
import pickle
import util

from ct.ct_head_constants import *


def update_pkl_bbox(args):
    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)

    counter = 1
    for series in series_list:
        if series.brain_bbox is None:
            print('{}: Generating bounding box for study {}...'.format(counter, series.accession_number))
            try:
                update_series_bbox(args, series)
            except:
                series.brain_bbox = None
                print('Could not generate bbox for study {}.'.format(series.accession_number))
            counter += 1

    # Write summary file for all series
    util.print_err('Dumping pickle file...')
    with open(os.path.join(args.output_dir, 'series_list.pkl'), 'wb') as pkl_file:
        pickle.dump(series_list, pkl_file)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(args.output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in series_list], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)


def update_series_bbox(args, series):
        with h5py.File(os.path.join(os.path.dirname(args.pkl_path), 'data.hdf5'), 'r') as hdf5_fh:
            if not series.is_bottom_up:
                volume = hdf5_fh[series.dset_path][:args.num_slices]
            else:
                volume = hdf5_fh[series.dset_path][-args.num_slices:]

        skull_bbox = None
        for i in range(volume.shape[0]):
            windowed_slice = util.apply_window(volume[i, :, :], W_CENTER_DEFAULT, W_WIDTH_DEFAULT)
            skull_bbox = util.get_min_bbox(skull_bbox, util.get_skull_bbox(windowed_slice))
        series.brain_bbox = list(skull_bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to pickle file for a study.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for JSON and pickle files.')
    parser.add_argument('--num_slices', type=int, default=100,
                        help='Number of slices from the top to get bounding boxes.')

    args = parser.parse_args()
    update_pkl_bbox(args)
