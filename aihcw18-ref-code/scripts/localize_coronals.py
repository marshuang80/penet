import os
import pickle 
import numpy as np
import csv 

def localize(file, thicc):
    with open(file, 'r') as f:
        slices = pickle.load(f)
    cropped = []
    
    crop_size = 384
    top = 256 - crop_size / 2 + 32 
    bottom = top + crop_size
    left = 256 - crop_size  / 2
    right = left + crop_size
   
    num_slices = 70 / thicc
    center_slice = len(slices) / 2 

    start, end = int(center_slice - num_slices / 2), int(center_slice + num_slices / 2)
    start = max(0, start)
    end = min(end, len(slices))
    for slice in slices[start:end]:
        if min(slice.shape) < 512:
            continue
        cropped.append(slice[top:bottom, left:right])
    return np.array(cropped)

with open('coronal_info.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    info = dict(reader)

rootdir = 'DataPEcor/images/'
storedir = 'DataPEcor/localized-new/images'

for file, thicc in info.items():
    localized = localize(os.path.join(rootdir, file + '.pickle'), float(thicc))
    if localized.shape[0] > 10:
        np.save(os.path.join(storedir, file), localized)
