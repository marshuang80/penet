import argparse
import h5py
import json
import numpy as np
import os
import pickle
import time
import util


from tqdm import tqdm


def create_hdf5(series_list, output_dir, resample=False, max_series=1e5):
    hdf5_fh = h5py.File(os.path.join(output_dir, 'data.hdf5'), 'a')
    for group_name in ('series', 'aneurysm_masks'):
        if group_name not in hdf5_fh:
            hdf5_fh.create_group('/{}'.format(group_name))

    assert len(series_list) < 1e5, 'Too many series for 5-digit IDs.'
    for i, s in enumerate(series_list):
        if i >= max_series:
            break
        dset_path = '/series/{:05d}'.format(i+1)
        if dset_path in hdf5_fh:
            continue
        print('Processing series {} from study {}...'.format(s.series_number, s.study_name))
        pixel_arrays = []
        is_valid_series = True
        for slice_name in tqdm(s.slice_names, total=len(s), unit=' slices'):
            # Process and write slices
            dcm_path = os.path.join(s.dcm_dir, slice_name + '.dcm')
            dcm = util.read_dicom(dcm_path)
            try:
                pixel_arrays.append(util.dcm_to_raw(dcm))
            except NotImplementedError:
                print('Unsupported image format, not converting study: {}'.format(s.study_name))
                is_valid_series = False
                break
        if not is_valid_series:
            continue

        volume = np.stack(pixel_arrays)

        aneurysm_mask_path = os.path.join(s.dcm_dir, 'aneurysm_mask.npy')
        if os.path.exists(aneurysm_mask_path):
            s.aneurysm_mask_path = aneurysm_mask_path
            aneurysm_mask = np.transpose(np.load(s.aneurysm_mask_path), [2, 0, 1])
        else:
            s.aneurysm_mask_path = None
            aneurysm_mask = None

        assert aneurysm_mask is None or aneurysm_mask.shape == volume.shape, \
            'Mismatched aneurysm mask and volume shapes: {} and {}'.format(aneurysm_mask.shape, volume.shape)
        if len(s) > 0 and resample:
            util.print_err('Resampling volume... Shape before: {}'.format(volume.shape))
            tick = time.time()
            dcm = util.read_dicom(os.path.join(s.dcm_dir, s.slice_names[0] + '.dcm'))
            volume, real_scale = util.resample(volume, dcm.SliceThickness, dcm.PixelSpacing, (1.5, 1., 1.))
            util.print_err('Shape after: {}. Resample took {} s.'.format(volume.shape, time.time() - tick))
            if aneurysm_mask is not None:
                util.print_err('Resampling mask... Shape before: {}, count before: {}.'.format(aneurysm_mask.shape, np.sum(aneurysm_mask > 0)))
                tick = time.time()
                aneurysm_mask, mask_scale = util.resample(aneurysm_mask, dcm.SliceThickness, dcm.PixelSpacing, (1.5, 1., 1.))
                util.print_err('Mask shape after: {}, count after: {}. Resample took {} s.'.format(aneurysm_mask.shape, np.sum(aneurysm_mask > 0), time.time() - tick))
                if not aneurysm_mask.any():
                    raise RuntimeError('Mask has zero volume after resampling.')

                if s.is_aneurysm:
                    # Recompute slice numbers where the aneurysm lives
                    s.aneurysm_bounds = get_aneurysm_range(aneurysm_mask)
                    s.aneurysm_ranges = [s.aneurysm_bounds]
                    s.absolute_range = [0, aneurysm_mask.shape[0]]

        # Create one dataset for the volume (int16), one for the mask (bool)
        s.dset_path = dset_path
        hdf5_fh.create_dataset(s.dset_path, data=volume, dtype='i2', chunks=True)

        if aneurysm_mask is not None:
            s.aneurysm_mask_path = '/aneurysm_masks/{:05d}'.format(i+1)
            hdf5_fh.create_dataset(s.aneurysm_mask_path, data=aneurysm_mask, dtype='?', chunks=True)

    # Print summary
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
    parser.add_argument('--resample', type=util.str_to_bool, required=True,
                        help='If true, resample the volumes to a scale of 1x1x1 mm.')
    parser.add_argument('--max_series', type=int, default=1e5,
                        help='Maximum number of series to convert. Mostly used for debugging.')
    args_ = parser.parse_args()

    with open(args_.pkl_path, 'rb') as pkl_file:
        all_series = pickle.load(pkl_file)

    create_hdf5(all_series, args_.output_dir, args_.resample, args_.max_series)
