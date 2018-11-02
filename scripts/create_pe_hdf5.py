"""Create HDF5 file for PE project."""
import argparse
import h5py
import numpy as np
import os
import pickle
import re
import util

from ct import CTPE, CONTRAST_HU_MIN, CONTRAST_HU_MAX
from tqdm import tqdm


def main(args):
    study_re = re.compile(r'(\d+)_(\d\.\d\d)')
    study_paths = []
    ctpes = []

    # Read slice-wise labels
    with open(args.slice_list, 'r') as slice_fh:
        slice_lines = [l.strip() for l in slice_fh.readlines() if l.strip()]
    name2slices = {}
    for slice_line in slice_lines:
        k, v = slice_line.split(':')
        v = v.strip()
        if v: 
            name2slices[k] = [int(n) for n in v.split(',')]
        else:
            name2slices[k] = []
    # Collect list of paths to studies to convert
    voxel_means = []
    voxel_stds = []
    for base_path, _, file_names in os.walk(args.data_dir):
        npy_names = [f for f in file_names if f.endswith('.npy')]
        for name in npy_names:
            match = study_re.match(name)
            if match and (name in name2slices):
                study_num, slice_thickness = int(match.group(1)), float(match.group(2))
                if slice_thickness in args.use_thicknesses:
                    # Add to list of studies
                    study_paths.append(os.path.join(base_path, name))
                    pe_slice_nums = name2slices.get(name, [])
                    ctpes.append(CTPE(study_num, slice_thickness, pe_slice_nums))

    # Create h5py file and load all studies into it
    hdf5_fh = h5py.File(os.path.join(args.output_dir, 'data.hdf5'), 'w')
    for ctpe, study_path in tqdm(zip(ctpes, study_paths), total=len(ctpes)):
        volume = np.load(study_path) - args.hu_intercept
        ctpe.num_slices = volume.shape[0]
        hdf5_fh.create_dataset(str(ctpe.study_num), data=volume, chunks=True)
        get_mean_std(volume, voxel_means, voxel_stds)

    with open(os.path.join(args.output_dir, 'data.pkl'), 'wb') as pkl_fh:
        pickle.dump(ctpes, pkl_fh)
    hdf5_fh.close()

    print('Wrote {} studies'.format(len(study_paths)))
    print('Mean {}'.format(np.mean(voxel_means)))
    print('Std {}'.format(np.mean(voxel_stds)))


def get_mean_std(scan, means, stds):
    scan = scan.astype(np.float32)
    scan = (scan - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
    scan = np.clip(scan, 0., 1.)
    means.append(np.mean(scan))
    stds.append(np.std(scan))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create HDF5 file for PE')

    parser.add_argument('--data_dir', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/data-final/images',
                        help='Base directory for loading 3D volumes.')
    parser.add_argument('--slice_list', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/1JuneSliceData/segmental_slice_list.txt')
    parser.add_argument('--use_thicknesses', default='1.25', type=str,
                        help='Comma-separated list of thicknesses to use.')
    parser.add_argument('--hu_intercept', type=float, required=True,
                        help='Intercept for converting from original numpy files to HDF5 (probably -1024).')
    parser.add_argument('--output_dir', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/segmental_cc_data',
                        help='Output directory for HDF5 file and pickle file.')

    args_ = parser.parse_args()
    args_.use_thicknesses = util.args_to_list(args_.use_thicknesses, arg_type=float, allow_empty=False)

    main(args_)
