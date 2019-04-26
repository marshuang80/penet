import argparse
import numpy as np
import pydicom
import os
import pandas as pd
import h5py

def main(args):

    slice_counts = []
    count = 0

    for subdir, dirs, files in os.walk(args.input_dir):
        if count % 10 == 0: print("\t" + str(count) + "/" + "3527")
        count += 1
        dcm_slices = []
        study_name = subdir.split("/")[-1]

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

        slice_counts.append(len(dcm_slices))

    print(np.array(slice_counts).mean())

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

