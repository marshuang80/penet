import numpy as np 
import pydicom 
import cv2
import torch
import util
import os
import glob
from ct.ct_pe_constants import W_CENTER_DEFAULT, W_WIDTH_DEFAULT, CONTRAST_HU_MEAN, \
                               CONTRAST_HU_MIN, CONTRAST_HU_MAX



def dicom_2_npy(input_study, series_description): 
    dcm_slices = []

    # get all dicom files in path
    files = glob.glob(os.path.join(input_study,"*dcm"))
    if len(files) == 0:
        raise Exception("No dicom files in directory")
        return

    # read in all dcm slices 
    for dicom in files:
        try: 
            dcm = pydicom.dcmread(dicom)
            arr = dcm.pixel_array
        except:
            print("error reading dicom")
            continue
        # skip dicom types that we don't want
        if dcm.SeriesDescription != series_description: 
            continue
        dcm_slices.append(dcm)

    # check if dicoms are succesfully retrived
    if len(dcm_slices) == 0:
        raise Exception("no dicom files retrived")
        return
        
    # sort slices
    # test using image patient location instead
    #dcm_slices_sorted = sorted(dcm_slices, key=lambda dcm: int(dcm.InstanceNumber))
    dcm_slices_sorted = sorted(dcm_slices, key=lambda dcm: int(dcm.ImagePositionPatient[-1]))
    # save as npy_volume
    npy_volume = np.array([dcm.pixel_array for dcm in dcm_slices_sorted])

    # reverse volumne if patient position defer from standard 
    # test using image patient location instead
    #if dcm.PatientPosition == "FFS":	
    #    npy_volume = npy_volume[::-1]

    return npy_volume


def normalize(img):
    
    img = img.astype(np.float32)
    img = (img - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN) 
    img = np.clip(img, 0., 1.) - CONTRAST_HU_MEAN
    return img


def format_img(img):
    """reshape, normalize image and convert to tensor"""

    num_slices = img.shape[0]
    num_windows = num_slices - 24 + 1

    # rescale
    interpolation=cv2.INTER_AREA
    img = util.resize_slice_wise(img, (208,208), interpolation)
    
    # crop
    row = (img.shape[-2] - 192) // 2
    col = (img.shape[-1] - 192) // 2
    img = img[:,row : row + 192, col : col + 192]
    
    # noramlize Hounsfield Units 
    img_normalized = normalize(img)

    # expand dimention for tensor
    img_split = np.array([img_normalized[i:i+24] for i in range(num_windows)])
    img_expand = [np.expand_dims(np.expand_dims(split, axis=0), axis=0) for split in img_split]

    # create torch tensor
    study_windows = [torch.from_numpy(np.array(window)) for window in img_expand]

    return study_windows
