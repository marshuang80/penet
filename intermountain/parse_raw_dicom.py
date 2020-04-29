"""
(1) Remove all non-axial DICOM files
(2) Convert DICOM files to .npy : one per study
(3) Create slice_list.txt to store slice list information for each study 
(4) Create HDF5 file for this study

TODO: 
    - create masks based on Hounsfield Units (HU)
    - contrast adjustment
    - remove out of range slices
    - auto phase/dataset label instead of hard code
    - move HDF5 creation to new scrip to include contrast adjustment and segmentation
"""

import argparse
import numpy as np
import pydicom
import os
import pandas as pd
import h5py

def main(args):

    # create directory for outpots
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    npy_path = os.path.join(args.output_dir,"images")
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    # create hdf5 file
    hdf5_fh = h5py.File(os.path.join(args.output_dir, "data.hdf5"), "a")

    slice_list = []
    df_acc = pd.read_csv(args.csv_path)
    acc_no = set(df_acc.ACCESSION_NO)

    df_acc.set_index("ACCESSION_NO", inplace=True)
    # loop over directories of all studies

    for subdir, dirs, files in os.walk(args.input_dir):
        dcm_slices = []
        study_name = subdir.split("/")[-1]

        # filter out unwanted files
        try:
            study_name = int(study_name)
        except:
            continue
        if not int(study_name) in acc_no: continue

        subsegmental = df_acc.loc[int(study_name)].SUBSEGMENTAL
        start = df_acc.loc[int(study_name)].START
        end = df_acc.loc[int(study_name)].END

        if np.isnan(start) or np.isnan(end):
            continue 
        # only include axiel slices
        for filename in files:
            file_path = os.path.join(subdir, filename)
            try:
                dcm = pydicom.dcmread(file_path)
                arr = dcm.pixel_array
            except:
                print("error reading dicom")
                continue
            if dcm.SeriesDescription != args.series_description:
                continue
            dcm_slices.append(dcm)
        
        # check for empty dir
        if len(dcm_slices) == 0:
            print("*",study_name)
            continue

        # sort slices by InstanceNumber
        dcm_slices_sorted = sorted(dcm_slices, key=lambda dcm: int(dcm.InstanceNumber))

        # gather study information
        dcm = dcm_slices_sorted[0]  # get example dcm

        thicc = dcm.SliceThickness
        label = 1 if df_acc.loc[int(study_name)].POSITIVE_CTPA == 1 else 0
        num_slices = int(end-start)

        phase = "test"
        # TODO 
        dataset = "segmental" if df_acc.loc[int(study_name)].SUBSEGMENTAL == 1 else "central"
        #slice_info = f"{study_name},{thicc},{label},{num_slices},{phase},{dataset}:"
        slice_info = str(study_name) + "," + str(thicc) + "," + str(label) + "," + str(num_slices) + "," + str(phase) + "," + str(dataset) + ":"

        # add slice numbers to slice info
        for idx in range(len(dcm_slices_sorted)):
            slice_info += str(idx)
            slice_info += ","

        slice_info = slice_info[:-1]
        slice_info += "\n"
        slice_list.append(slice_info)
        
        # create np array of slices and save to output_dir/images directory
        npy_volume = np.array([dcm.pixel_array for dcm in dcm_slices_sorted])
        print(npy_volume.shape)
        
        try:
            npy_volume = npy_volume[int(start): int(end)+1]
        except:
            npy_volume = npy_volume
        print("* " + str(study_name) + " : " + str(label))
        
        np.save(npy_path + "/" + str(study_name) + "_" + str(thicc), npy_volume)

        # reverse volumne
        if dcm.PatientPosition == "FFS":	
            npy_volume = npy_volume[::-1]
        
        # write to hdf5 
        try:
            hdf5_fh.create_dataset(str(study_name), data=npy_volume, dtype=np.int16, chunks=True)
            #i += 1
        except:
            print(npy_volume.dtype)
            print("*",study_name)
        

    hdf5_fh.close()
    with open(os.path.join(args.output_dir,"slice_list.txt"),"w") as f:
        f.writelines(slice_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default="/mnt/bbq/intermountain/CTPA_RANDOM_DICOM/",
                        help="Root directories holding CTA studies")
    parser.add_argument('--output_dir', type=str, default="/mnt/bbq/intermountain/CTPA_RANDOM_NPY/",
                        help="Output directory for .npy files and slice_list.txt")
    parser.add_argument('--csv_path', type=str, default="/mnt/bbq/intermountain/CTPA.csv",
                        help="Location for csv file")
    parser.add_argument('--series_description', type=str, default="CTA 2.0 CTA/PULM CE",
                        help="Descriptions used to indicate axiel view")

    args = parser.parse_args()
    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    main(args)

