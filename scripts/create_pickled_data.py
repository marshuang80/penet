import argparse
import csv
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
import util

from collections import defaultdict




def get_paths_and_labels(phase, series_type, data_dir):
    """ Generates a list of all image paths and associated labels. 
    
    Args:
        phase: one of 'train', 'val', test'
        series_type: one of 'sagittal', 'axial'
        data_dir: directory where csvs with 'train', 'val', 'test' splits are stored
    """
    img_dir = os.path.join(data_dir, series_type)
    csv_path = os.path.join(data_dir, '{}_{}.csv'.format(phase,series_type))

    df = pd.read_csv(csv_path, header=None, names=['Path', 'Label'])
    query = 'Fracture'
    df['Label'] = df['Label'].apply(lambda x: 1 if x == query else 0)

    # converting paths to full paths for easier loading later
    df['Path'] = df['Path'].apply(lambda x: Path(img_dir) / (x + '.npy'))
    df['Label'] = df['Label'].astype(int)

    # remove img_paths that don't actually exist
    df = df[df['Path'].apply(os.path.exists)]
    df.reset_index(inplace=True)

    img_paths = df['Path'].tolist()
    labels = df['Label'].tolist()
    
    return img_paths, labels


def get_location_annotations(location_csv_path):
    """ Loads the location annotations file for training.
    
    Args:
        data_dir: directory where annotations are stored
    """  
    fracture_locations_df = pd.read_csv(location_csv_path, index_col='Series_Ref')
    fracture_locations = defaultdict(list)
    for series_ref, row in fracture_locations_df.iterrows():
        fracture_locations[(series_ref, row['SliceNum'])].extend(np.array(str(row['GridLocation']).split(",")).astype(np.int) - 1)
    return fracture_locations

def get_slice_annotations(data_dir, series_type):
    """ Loads the slice annotations file for training.
    
    Args:
        data_dir: directory where annotations are stored
    """    
    fracture_slice_ranges = defaultdict(list)
    fracture_annotations_file_path = os.path.join(data_dir, '{}_sliceNumber_annotations.csv'.format(series_type))
    with open(fracture_annotations_file_path) as f:
        reader = csv.DictReader(f)
        for line in reader:
            ref_id = str(line['Series_Ref'])
            slice_start = int(line['StartSlice']) 
            slice_end = int(line['EndSlice']) 
            assert (slice_start > 0 and slice_end > 0 and slice_end >= slice_start)
            fracture_slice_ranges[ref_id].append((slice_start,slice_end))
    return fracture_slice_ranges


def get_series_ref_from_img_path(img_path):
    """ Parses the series reference (name of file without extension) from the full image path.

    Args:
        img_path: full path to the image file
    """
    reg_ex = '([0-9_])+(?=\.npy)'
    match = re.search(reg_ex, img_path)
    if not match:
        util.print_err('Invalid file found: {}'.format(img_path))
    return match.group(0)


def create_windows(img_paths, labels, data_dir, series_type, location_csv_path, img_format, num_slices, phase):
    """ Creates and returns the windows list with each entry structured as:
    
    (start slice, end slice (not inclusive), path to img, series idx, label for the window)
    
    Args:
        img_paths: list of all paths to series
        labels: list of series -> fracture / no fracture; matches with img_paths
        data_dir: directory where annotations are stored
        series_type: one of 'sagittal', 'axial'
    """
    
    fracture_slice_ranges = get_slice_annotations(data_dir, series_type)
    if series_type == 'sagittal':
        fracture_locations = get_location_annotations(location_csv_path)
    else:
        fracture_locations = {}
        
    windows = []
    count = 0

    # loop through every series
    for series_idx, path in enumerate(img_paths):
        slice_ranges = []
        series = np.load(path)
        
        if img_format == 'raw':
            expected_num_channels = 4
        elif img_format == 'png':
            expected_num_channels = 3
            
        
        if len(series.shape) != expected_num_channels:
            print("Bad series: %s", path)
            continue

        # get all slices that have fractures in them
        ref_id = get_series_ref_from_img_path(str(path))
        if ref_id in fracture_slice_ranges:
            slice_ranges = fracture_slice_ranges[ref_id]
            slice_ranges = [list(range(x, y)) for x, y in slice_ranges]
            
        stride = 1 if num_slices == 1 else 4
            
        # iterate through all possible windows
        for i in range(1, len(series) - num_slices, stride):
            label = labels[series_idx] if phase == 'test' else 0
            count += 1
            fracture_squares = []
                
            # check whether the current range of slices contains any fracture
            for slice_range in slice_ranges:
                if len(set(range(i, i + num_slices)).intersection(slice_range)) > 0:
                    label = 1
                    if series_type == 'sagittal':
                        for idx in range(i, i + num_slices):
                            if (ref_id, idx) in fracture_locations:
                                fracture_squares.extend(fracture_locations[(ref_id, idx)])

            fracture_set = set(fracture_squares)
            # start slice, end slice (not inclusive), path to img, series idx, label, fracture squares for the window
            windows.append((i, i + num_slices, path, series_idx, label, fracture_set))
            
    # sanity check to make sure nothing weird is going on
    assert count == len(windows)
    
    return windows

# creates and saves the img_paths, labels, and windows list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_type', type=str, choices=('sagittal', 'axial'), 
                        help='Series type to use')
    parser.add_argument('--data_dir', type=str, 
                        default="/data/CT-CSPINE/processed-studies/data_20180524_161757/", 
                        help='data directory where csvs and annotations are stored')
    parser.add_argument('--location_csv', type=str,
                        default="/data/CT-CSPINE/processed-studies/sagittal_fracture_location_annotations.csv",
                        help='path to csv with the location-specific fracture annotations')
    parser.add_argument('--ext', type=str,
                        default="", 
                        help='extension for the file names')
    parser.add_argument('--img_format', type=str, default='raw', choices=('raw', 'png'),
help='Format for input images: "raw" means raw Hounsfield Units, "png" means PNG.')
    parser.add_argument('--num_slices', type=int, default=1, choices=(1,8),
help='Number of slices to include per window')
    args = parser.parse_args()
    
    for phase in ('test', 'val', 'train'):
        print("Creating img paths and labels for {}, phase {}".format(args.series_type, phase))
        img_paths, labels = get_paths_and_labels(phase, args.series_type, args.data_dir)
        print("Creating windows for {}, phase {}".format(args.series_type, phase))
        windows = create_windows(img_paths, labels, args.data_dir, args.series_type, args.location_csv, args.img_format, args.num_slices, phase)
        print("Saving")

        with open(os.path.join(args.data_dir, '{}_{}_img_paths{}_{}.pkl'.format(args.series_type, phase, args.ext, args.num_slices)), 'wb') as f:
            pickle.dump(img_paths, f)

        with open(os.path.join(args.data_dir, '{}_{}_labels{}_{}.pkl'.format(args.series_type, phase, args.ext, args.num_slices)), 'wb') as f:
            pickle.dump(labels, f)

        with open(os.path.join(args.data_dir, '{}_{}_windows{}_{}.pkl'.format(args.series_type, phase, args.ext, args.num_slices)), 'wb') as f:
            pickle.dump(windows, f)
