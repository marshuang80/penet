"""Convert raw dicoms to inputs for PENet"""

import argparse
import pandas as pd
import pickle
import numpy as np
import h5py
import pydicom
import os
import sys
sys.path.append(os.getcwd())
import util

from ct import CTPE 
from tqdm import tqdm
from rsna.constants import *


def main(args):

    # create output dir
    path_dict = SPLIT_PATHS[args.split]
    path_dict['output_dir'].mkdir(parents=True, exist_ok=True)

    # create hdf5 file
    hdf5_fh = h5py.File(path_dict['hdf5'], "a")

    # read annotations
    df = pd.read_csv(path_dict['csv'])
    all_series = set(df.SeriesInstanceUID)
    
    if args.split == 'train':
        all_studies = list(set(df['StudyInstanceUID'].tolist()))
        num_train = int(len(all_studies) * TRAIN_PERCENT)
        train_studies = all_studies[:num_train]
        df['Phase'] = 'val'
        df.loc[df['StudyInstanceUID'].isin(train_studies), 'Phase'] = 'train'
        print(df['Phase'].value_counts())
    else:
        df['Phase'] = 'test'

    # series list to be dump as pickle
    series_list = []

    for series_uid in tqdm(all_series):

        # group series 
        series_df = df[df.SeriesInstanceUID == series_uid]
        study_uid = series_df['StudyInstanceUID'].head(1).item()
        phase = series_df['Phase'].head(1).item()
        if args.split == 'train':
            series_df = series_df[
                ['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID', 'pe_present_on_image']
            ]
        else: 
            series_df = series_df[['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID']]

        # get instance paths
        get_dicom_path = lambda row: path_dict['dicoms'] \
            / study_uid / row['SeriesInstanceUID'] / (row['SOPInstanceUID'] + '.dcm')
        series_df['Path'] = series_df.apply(get_dicom_path, axis=1)

        # aggregate to list
        series_df = series_df.groupby(['SeriesInstanceUID']).aggregate(list)
        dcm_paths = series_df['Path'].tolist()[0]
        if args.split == "train":
            dcm_labels = series_df['pe_present_on_image'].tolist()[0]
        else: 
            dcm_labels = [0] * len(dcm_paths)

        # read dicom slices
        dcm_slices = []
        for path in dcm_paths:
            try:
                dcm_slice = pydicom.dcmread(path)
                pixel_array = dcm_slice.pixel_array
            except:
                print(f"error reading dicom from {path}")
                continue
            dcm_slices.append(dcm_slice)

        # check for empty dir
        if len(dcm_slices) == 0:
            print(f"{series_uid} has no slices")
            continue

        # sort slices by InstanceNumber
        dcm_labels_sorted = [label for _, label in sorted(zip(dcm_slices, dcm_labels), key=lambda pair: pair[0].ImagePositionPatient[-1])]
        dcm_slices_sorted = sorted(dcm_slices, key=lambda dcm: int(dcm.ImagePositionPatient[-1]))

        # read pixel data
        try: 
            npy_volume = np.array([
                util.dcm_to_raw(dcm) for dcm in dcm_slices_sorted
            ])
        except: 
            print(f"error extracting pixel data from {series_uid}")
            continue

        # get relevant information
        dcm = dcm_slices[0]
        is_positive = 1 if 1 in dcm_labels_sorted else 0 
        pe_slice_idx = [i for i,l in enumerate(dcm_labels_sorted) if l == 1]
        num_slices = len(dcm_labels_sorted)
        slice_thickness = dcm.SliceThickness

        # reverse positiion
        #if dcm.PatientPosition == "FFS":	
        #    npy_volume = npy_volume[::-1]

        # save parsed data
        hdf5_fh.create_dataset(str(series_uid), data=npy_volume, dtype=np.int16, chunks=True)

        try:
            series = CTPE(
                study_num=series_uid,
                slice_thickness=slice_thickness,
                pe_slice_nums=pe_slice_idx,
                num_slices=num_slices,
                dataset='rsna',
                is_positive=is_positive,
                phase=phase
            )
        except RuntimeError as e:
            print(f'Error at {series_uid}: {e}')
            continue

        series_list.append(series)

    # Write summary file for all series
    with open(path_dict['series_list'], 'wb') as pkl_file:
        pickle.dump(series_list, pkl_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=('train', 'test'),
                        help="Indicate if parsing train or test dataset")
    args = parser.parse_args()

    main(args)

