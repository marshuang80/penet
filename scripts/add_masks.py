import argparse
import h5py
import json
import numpy as np
import os
import pickle
import util


# List of accession numbers that were missing masks
MASK_ACCS = [5918511,
             5610111,
             5648322,
             5659130,
             5786350,
             5786819,
             5965425,
             5966352,
             5974373,
             5977103,
             6005462,
             6088559,
             6132582,
             6255856]


def add_to_hdf5(series_list, output_dir):
    hdf5_fh = h5py.File(os.path.join(output_dir, 'data.hdf5'), 'a')

    # Print summary
    util.print_err('BEFORE:')
    util.print_err('Series: {}'.format(len(hdf5_fh['/series'])))
    util.print_err('Aneurysm Masks: {}'.format(len(hdf5_fh['/aneurysm_masks'])))

    assert len(series_list) < 1e5, 'Too many series for 5-digit IDs.'
    for i, s in enumerate(series_list):
        if s.accession_number not in MASK_ACCS or s.dset_path not in hdf5_fh:
            continue

        aneurysm_mask_path = os.path.join(s.dcm_dir, 'aneurysm_mask.npy')
        if os.path.exists(aneurysm_mask_path):
            s.aneurysm_mask_path = aneurysm_mask_path
            aneurysm_mask = np.transpose(np.load(s.aneurysm_mask_path), [2, 0, 1])
        else:
            s.aneurysm_mask_path = None
            aneurysm_mask = None
            util.print_err('No aneurysm mask for {}'.format(s.accession_number))

        if aneurysm_mask is not None:
            s.aneurysm_mask_path = '/aneurysm_masks/{:05d}'.format(i+1)
            util.print_err('Added aneurysm mask for {} to HDF5'.format(s.accession_number))
            hdf5_fh.create_dataset(s.aneurysm_mask_path, data=aneurysm_mask, dtype='?', chunks=True)

    # Print summary
    util.print_err('AFTER:')
    util.print_err('Series: {}'.format(len(hdf5_fh['/series'])))
    util.print_err('Aneurysm Masks: {}'.format(len(hdf5_fh['/aneurysm_masks'])))

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

    add_to_hdf5(all_series, args_.output_dir)


