import argparse
import nibabel as nib
import numpy as np
import os
import pickle
import util

from tqdm import tqdm


def generate_nifti(args):
    """Generate binary masks from model output and convert them to NIfTI format."""
    mask_list = [m for m in os.listdir(args.mask_dir) if m.endswith('.npy')]
    dset_nums, file_suffix = zip(*[m.split('_') for m in mask_list])
    save_dir = os.path.join(args.save_dir, 'niftis_threshold_{}'.format(args.threshold))
    os.makedirs(save_dir, exist_ok=True)

    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)

    for num, suffix in tqdm(zip(dset_nums, file_suffix)):
        output = np.load(os.path.join(args.mask_dir, num + '_' + suffix))
        series = None
        for s in series_list:
            if s.dset_path == '/series/' + num:
                series = s
        assert series is not None, 'Could not find series {} in series list pickle file.'.format(num)
        dcm_list = sorted([d for d in os.listdir(series.dcm_dir) if d.endswith('.dcm')])
        assert len(dcm_list) == output.shape[0], \
            '{} dcm files in directory, but mask dimension is {}.'.format(len(dcm_list), output.shape)
        dcm_first = util.read_dicom(os.path.join(series.dcm_dir, dcm_list[0]))
        dcm_second = util.read_dicom(os.path.join(series.dcm_dir, dcm_list[1]))

        affine_x = -dcm_first.PixelSpacing[0] * dcm_first.ImageOrientationPatient[0]
        affine_y = -dcm_first.PixelSpacing[1] * dcm_first.ImageOrientationPatient[4]
        affine_z = dcm_second.ImagePositionPatient[2] - dcm_first.ImagePositionPatient[2]

        # Define affine matrix
        affine = [[affine_x, 0., 0., -dcm_first.ImagePositionPatient[0]],
                  [0., affine_y, 0., -dcm_first.ImagePositionPatient[1]],
                  [0., 0., affine_z, dcm_first.ImagePositionPatient[2]],
                  [0., 0., 0., 1.]]
        affine = np.array(affine, dtype=np.float_)

        # Fix dimension of mask to match dicoms and threshold probabilities
        output = np.transpose(output, (2, 1, 0))
        if not series.is_bottom_up:
            output = np.flip(output, axis=2)
        output = (output >= args.threshold).astype(np.float_)

        # Create NIfTI file and save
        nifti_out = nib.Nifti1Image(output, affine=affine)
        filename = '{}_{}_output.nii.gz'.format(num, series.study_name)
        nib.save(nifti_out, os.path.join(save_dir, filename))


def overlay_mask(args):
    """Overlay ground truth mask on model output binary mask."""
    mask_list = [m for m in os.listdir(args.mask_dir) if m.endswith('.npy')]
    dset_nums, file_suffix = zip(*[m.split('_') for m in mask_list])
    save_dir = os.path.join(args.save_dir, 'overlays_threshold_{}'.format(args.threshold))
    os.makedirs(save_dir, exist_ok=True)

    with open(args.pkl_path, 'rb') as pkl_file:
        series_list = pickle.load(pkl_file)

    for num, suffix in tqdm(zip(dset_nums, file_suffix)):
        output = np.load(os.path.join(args.mask_dir, num + '_' + suffix))
        series = None
        for s in series_list:
            if s.dset_path == '/series/' + num:
                series = s
        assert series is not None, 'Could not find series {} in series list pickle file.'.format(num)
        dcm_list = sorted([d for d in os.listdir(series.dcm_dir) if d.endswith('.dcm')])
        assert len(dcm_list) == output.shape[0], \
            '{} dcm files in directory, but mask dimension is {}.'.format(len(dcm_list), output.shape)
        dcm_first = util.read_dicom(os.path.join(series.dcm_dir, dcm_list[0]))
        dcm_second = util.read_dicom(os.path.join(series.dcm_dir, dcm_list[1]))

        affine_x = -dcm_first.PixelSpacing[0] * dcm_first.ImageOrientationPatient[0]
        affine_y = -dcm_first.PixelSpacing[1] * dcm_first.ImageOrientationPatient[4]
        affine_z = dcm_second.ImagePositionPatient[2] - dcm_first.ImagePositionPatient[2]

        # Define affine matrix
        affine = [[affine_x, 0., 0., -dcm_first.ImagePositionPatient[0]],
                  [0., affine_y, 0., -dcm_first.ImagePositionPatient[1]],
                  [0., 0., affine_z, dcm_first.ImagePositionPatient[2]],
                  [0., 0., 0., 1.]]
        affine = np.array(affine, dtype=np.float_)

        # Fix dimension of mask to match dicoms and threshold probabilities
        output = np.transpose(output, (2, 1, 0))
        output = (output >= args.threshold).astype(np.float_)
        if series.is_aneurysm:
            try:
                true_mask = np.load(os.path.join(series.dcm_dir, 'aneurysm_mask.npy'))
                true_mask = np.transpose(true_mask, (1, 0, 2))
            except:
                # TODO: Remove after drawing missing masks
                continue
        else:
            true_mask = np.zeros(output.shape)
        if not series.is_bottom_up:
            output = np.flip(output, axis=2)
            true_mask = np.flip(true_mask, axis=2)
        false_positives = np.logical_and(output, np.logical_not(true_mask)).astype(np.float_)
        true_positives = np.logical_and(output, true_mask).astype(np.float_) * 2.
        false_negatives = np.logical_and(np.logical_not(output), true_mask).astype(np.float_) * 3.
        overlay = false_positives + true_positives + false_negatives

        # Create NIfTI file and save
        nifti_out = nib.Nifti1Image(overlay, affine=affine)
        filename = '{}_{}_overlay.nii.gz'.format(num, series.study_name)
        nib.save(nifti_out, os.path.join(save_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask_dir', type=str, default='/pidg/chute/hsct/results/chute_vbt',
                        help='Directory with output mask npy files.')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory to save converted NIfTI files.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold probability for binary mask.')
    parser.add_argument('--pkl_path', type=str, default='/pidg/cta/all/series_list.pkl',
                        help='Path to pickle file.')
    parser.add_argument('--output', type=str, choices=('model_output', 'overlay'),
                        help='Type of output NIfTI file. Either model output or output overlaid with true mask.')

    args_ = parser.parse_args()

    if args_.output == 'model_output':
        generate_nifti(args_)
    elif args_.output == 'overlay':
        overlay_mask(args_)
