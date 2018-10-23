import argparse
import h5py
import json
import math
import numpy as np
import os
import pickle
import util

from tqdm import tqdm


def add_spacing_info(series_list, output_dir):
    hdf5_fh = h5py.File(os.path.join(output_dir, 'data.hdf5'), 'a')

    # Go through all DICOMs and get spacing info
    for s in tqdm(series_list):
        try:
            dcm_names = [slice_name + '.dcm' for slice_name in s.slice_names]
            dcm_path_0, dcm_path_1 = os.path.join(s.dcm_dir, dcm_names[0]), os.path.join(s.dcm_dir, dcm_names[1])
            dcm_first, dcm_second = util.read_dicom(dcm_path_0), util.read_dicom(dcm_path_1)
            s.spacing_between_slices = math.fabs(dcm_first.ImagePositionPatient[2] - dcm_second.ImagePositionPatient[2])
        except Exception as e:
            print('Error for {}'.format(s.dcm_dir))
            print(e)

    # Dump pickle and JSON (updated dset_path and mask_path attributes)
    util.print_err('Dumping pickle file...')
    with open(os.path.join(output_dir, 'series_list.pkl'), 'wb') as pkl_fh:
        pickle.dump(series_list, pkl_fh)
    util.print_err('Dumping JSON file...')
    with open(os.path.join(output_dir, 'series_list.json'), 'w') as json_file:
        json.dump([dict(series) for series in series_list], json_file,
                  indent=4, sort_keys=True, default=util.json_encoder)

    # Clean up
    hdf5_fh.close()


def get_aneurysm_range(mask):
    """Get range of slice indices where an aneurysm lives."""
    is_aneurysm = np.any(mask, axis=(1, 2))
    slice_min, slice_max = np.where(is_aneurysm)[0][[0, -1]]

    return [int(slice_min), int(slice_max)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to pickle file for a study.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for JSON and Pickle files.')
    args_ = parser.parse_args()

    with open(args_.pkl_path, 'rb') as pkl_file:
        all_series = pickle.load(pkl_file)

    add_spacing_info(all_series, args_.output_dir)
