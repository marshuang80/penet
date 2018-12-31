import argparse
import nibabel as nib
import numpy as np
import os
import util

from tqdm import tqdm

def save_mask_npy(input_dir, reconvert, flip):
    """Convert mask NIfTI files into Numpy arrays, and save in .npy format.

    Args:
        input_dir: Directory containing mask.nii.gz files.
    """
    for dir_path, _, filenames in tqdm(list(os.walk(input_dir))):
        mask_list = [f for f in filenames if f.endswith('.nii.gz') and not f.startswith('.')]
        if len(mask_list) == 0:
            continue
        if not reconvert and len([f for f in filenames if f.endswith('mask.npy')]) > 0:
                continue
        if len(mask_list) > 1:
            raise RuntimeError('{} mask files in {}.'.format(len(mask_list), dir_path))
        mask_filename = mask_list[0]
        mask_file = nib.load(os.path.join(dir_path, mask_filename))
        mask_header = mask_file.header
        mask_array = mask_file.get_data()
        if mask_header['sform_code'] <= 0 or mask_header['srow_x'][0] > 0:
            raise RuntimeError('Check header and x-y axis for mask file in {}.'.format(dir_path))

        dcm_files = sorted([d for d in filenames if d.endswith('.dcm')])
        num_dcms = len(dcm_files)
        if mask_array.shape[2] != num_dcms:
            raise RuntimeError('Mask file has {} slices, but {} dicoms present in folder {}.'
                               .format(mask_array.shape[2], num_dcms, dir_path))
        mask_array = mask_array.transpose(1, 0, 2)

        if flip:
            dcm_first = util.read_dicom(os.path.join(dir_path, dcm_files[0]))
            dcm_second = util.read_dicom(os.path.join(dir_path, dcm_files[1]))
            if dcm_second.ImagePositionPatient[2] - dcm_first.ImagePositionPatient[2] < 0:
                mask_array = np.flip(mask_array, axis=2)
                print('Flipped mask for series {}.'.format(dir_path))

        mask_path = os.path.join(dir_path, 'mask.npy')
        np.save(mask_path, mask_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/data3/CTA/aneurysm_0306',
                        help='Root directory holding CTA studies (e.g. "/data3/CTA/aneurysm_0306".')
    parser.add_argument('--reconvert', type=util.str_to_bool, default='true',
                        help='If true, overwrite existing "mask.npy" files.')
    parser.add_argument('--flip', type=util.str_to_bool, default='true',
                        help='If true, flips masks for scans with top-down scan directions.')

    args_ = parser.parse_args()

    save_mask_npy(args_.input_dir, args_.reconvert, args_.flip)

