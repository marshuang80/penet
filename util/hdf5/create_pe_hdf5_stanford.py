"""Create HDF5 file for PE project."""
import argparse
import numpy as np
import os
import pickle
import re
import util
import pandas as pd
import h5py

from ct import CTPE, CONTRAST_HU_MIN, CONTRAST_HU_MAX
from tqdm import tqdm


def main(args):
    study_re = re.compile(r'(\d+)_(\d\.\d)')
    study_paths = []
    ctpes = []

    idx2acc = pickle.load(open("/data4/PE_stanford/idx_2_acc_stanford.pkl","rb"))
    
    df = pd.read_csv(args.csv_path)
    df = df.set_index("acc")

    df = df[~df.index.duplicated(keep='first')]


    # Read slice-wise labels
    with open(args.slice_list, 'r') as slice_fh:
        slice_lines = [l.strip() for l in slice_fh.readlines() if l.strip()]
    name2slices = {}
    name2info = {}
    print(len(slice_lines))
    for slice_line in slice_lines:
        try:
            info, slices = slice_line.split(':')
        except:
            print("exception")
            print(slice_line)
            continue
        slices = slices.strip()
        info = info.split(',')
        studynum, thicc, label, num_slices, phase, dataset = int(info[0]), float(info[1]), int(info[2]), int(info[3]), info[4], info[5]
        #if phase != "test": 
        #    continue

        name2info[studynum] = [thicc, label, num_slices, phase, dataset]
        if slices: 
            name2slices[studynum] = [int(n) for n in slices.split(',')]
        else:
            name2slices[studynum] = []
        if not slices and label == "1": print("EXCEPTION")
    # Collect list of paths to studies to convert
    voxel_means = []
    voxel_stds = []
    for study_num in name2slices.keys():
        study_paths.append(study_num)
        pe_slice_nums = name2slices.get(study_num, [])
        thicc, label, num_slices, phase, dataset = name2info[study_num]
        print(study_num, label, phase, num_slices)
        acc = idx2acc[str(study_num)]
        age = df.loc[acc].current_age_yrs
        is_smoker = df.loc[acc].SMOKER
        race = df.loc[acc].CANONICAL_RACE
        gender = df.loc[acc].GENDER

        if type(gender) != str: print(gender)

        slice_thickness = args.use_thicknesses[0]
        ctpes.append(CTPE(study_num, slice_thickness, pe_slice_nums, num_slices, dataset, is_positive=label, phase=phase,
                          age=age, is_smoker=is_smoker, race=race, gender=gender))

    with open(os.path.join(args.output_dir, 'series_list.pkl'), 'wb') as pkl_fh:
        pickle.dump(ctpes, pkl_fh)

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
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/tanay_data_12_4_clean/slice_list_12_4_clean.txt')
    parser.add_argument('--use_thicknesses', default='1.25', type=str,
                        help='Comma-separated list of thicknesses to use.')
    parser.add_argument('--hu_intercept', type=float, default=-1024,
                        help='Intercept for converting from original numpy files to HDF5 (probably -1024).')
    parser.add_argument('--output_dir', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/tanay_data_12_4_clean',
                        help='Output directory for HDF5 file and pickle file.')
    parser.add_argument('--csv_path', type=str)

    args_ = parser.parse_args()
    args_.use_thicknesses = util.args_to_list(args_.use_thicknesses, arg_type=float, allow_empty=False)

    main(args_)
