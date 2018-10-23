import os
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
import argparse
import json
import time
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET
import pydicom as dicom
import util

'''

    rootdir: This should be a (relative or absolute) path to the data directory.
    hounsfield_clamp: In case the image is a CT-scan, the pixel array is in Hounsfield units and 
                      in some cases, it can help to ignore some points in the data which might 
                      be irrelevant.
    usage:
        python spine_preprocess.py -r '/data3/CT-CSPINE' -o '/data3/CT-CSPINE/processed-studies' -s 'axial' -i '/data3/CT-CSPINE/processed-studies/data_20180524_161757'
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--rootdir", default = '/data3/CT-CSPINE', type=str)
    parser.add_argument('-o', "--outputdir", default = '/data3/CT-CSPINE/processed-studies', type=str)
    parser.add_argument('-c', "--hounsfield_clamp", default='-500 1300', type=str)
    parser.add_argument('-i', "--images_dir_existing", default=None, type=str)
    parser.add_argument('-s', "--series_type", default=None, type=str)
    return parser

def get_hounsfield_clamp():
    """ If creating the images as PNG, apply a clamp to the houndsfield units at creation time.
        Expected argument format is "[Window min] [Window max]
    """ 
    hounsfield_clamp = None
    try: 
        if args.hounsfield_clamp:
            hounsfield_clamp = [int(x) for x in args.hounsfield_clamp.split(' ')]  
    finally:
        return hounsfield_clamp

def extract_png_from_dcm(dcm_file_path):
    """ Function to convert from a DICOM image to png and save it
    Args:
        dcm_file_path: An file path to write the png data
    """
    dcm_file = open(dcm_file_path, 'rb')
    dcm = dicom.dcmread(dcm_file)
    pa = dcm.pixel_array
    shape = pa.shape

    hounsfield_clamp = get_hounsfield_clamp()
    if hounsfield_clamp and len(hounsfield_clamp) == 2:
        low = hounsfield_clamp[0]
        high = hounsfield_clamp[1]
        pa[pa >= high] = high
        pa[pa <= low] = low

    # Rescaling grey scale between 0-255
    max_val = np.max(pa).astype(np.float32)

    if max_val == 0:
        return None

    image_2d_scaled = (pa.astype(np.float32) / max_val * 255.).astype(np.uint8)
    
    if len(shape) == 3:
        pa = np.max(pa, axis=0)
        shape = pa.shape

    im = Image.fromarray(image_2d_scaled)  
    return im

def get_series_label(dcm_file_path):
    """ Assign a series label (sagittal, axial, coronal, or unknown) based on the dcm tags 
    Args: 
        dcm_file_path - path to the dcm file
    Returns:
        string with series label
    """
    
    dcm_file = open(dcm_file_path, 'rb')
    dcm = dicom.dcmread(dcm_file)
    series_desc = dcm.SeriesDescription.lower()
    try:
        image_type_list  = [image_type.lower() for image_type in dcm.ImageType]
        if 'axial' in image_type_list:
            return 'axial'
        elif 'sag' in series_desc or 'sag' in image_type_list:
            return 'sagittal'
        elif 'cor' in series_desc or 'cor' in image_type_list:
            return 'coronal'
    except Exception:
        print('Error reading %s' % dcm_file_path)
    return 'unknown'
    
def validate_spine_ct_dcm(file_path):
    """ Validate a dcm for this task, ruling out invalid files based on tags 
    Args: 
        file_path - path to the dcm file
    Returns: true if the dcm is valid, false otherwise
    """
    # Check if the dicom is valid
    try:
        dcm_file = open(file_path, 'rb')
        plan = dicom.dcmread(dcm_file)
        desc = plan.SeriesDescription.lower()
        if desc == 'dose report':
            return False
        if 'head' in desc:
            return False
        for neg in ['LOCALIZER','SCREEN SAVER', 'SCREEN SAVE', 'OTHER']:
            if neg in plan.ImageType:
                return False
        if len(plan.dir('ConvolutionKernel')) > 0:
            for neg in ['SOFT']:
                if neg in plan.ConvolutionKernel:
                    return False
    except AttributeError:
        return False  
    
    return True

def parse_study_xml(filename):
    """ Parses the xml for a DCM study, creating python objects mapping series to labels, studies, and trial acc numbers 
    """
    series_dict = dict()
    refID_to_trial = dict()
    study_dict = defaultdict(list)
    study = ET.parse(filename)
    accession_num = filename.split('/')[3]
    study_dir = '/'.join(filename.split('/')[:-1])
    series_list = set([series.text for series in study.findall('.//series')])
    for series in series_list:
        files = [file.text for file in study.findall(".//*[series='%s']/file" % series)]
        ## If the series has less than 10 images, don't include it
        if len(files) < 10:
            continue
        else:
            dcm_file_path = study_dir + '/' + files[0]
            is_valid_dcm  = validate_spine_ct_dcm(dcm_file_path)
            if not is_valid_dcm:
                continue
            else:
                series_label = get_series_label(dcm_file_path)
                trial_ref = "%s_%03d" % (accession_num, int(series))
                series_dict[trial_ref] = series_label
                refID_to_trial[trial_ref] = accession_num
                study_dict[study_dir].append(trial_ref)
    return (study_dict, series_dict, refID_to_trial) 

def preprocess_series(images_dir, series_label, study_tuple):
    """ Converts a series to a npy block of raw houndsfield units
        Args:
            images_dir/series_label = the directory where the images will be saved
            study_tuple = info about where the raw dcm inputs are stored
    """
    study_dir, series = study_tuple
    series_nr = str(int(series.split('_')[1]))
    study = ET.parse(study_dir + '/__index.xml')
    instances = set([int(instance.text) for instance in study.findall(".//*[series='%s']/instance"% series_nr)])
    files = [study.findtext(".//*[series='%s']/[instance='%d']/file" % (series_nr, instance)) for instance in sorted(instances)]
    series_len = len(files)

    processed_slices = []
    for file in files:
        try:
            dcm_file_path = study_dir + '/' + file
            dcm_file = open(dcm_file_path, 'rb')
            plan = dicom.dcmread(dcm_file)
            img_slice = util.dcm_to_raw(plan)
            if img_slice is not None:
                processed_slices.append(img_slice)
            block = np.array(processed_slices)
            if 0 not in block.shape:
                npy_file_path = os.path.join(os.path.join(images_dir, series_label), series)
                np.save(npy_file_path, block)
        except Exception as e:
            print (e) 
            
def preprocess_wrapper():
    """ Main wrapper for preprocessing all studies in the rootdir
    """
    series_labels = ['sagittal', 'unknown','axial','coronal']
    if not args.images_dir_existing:
        images_dir = os.path.join(args.outputdir, 'data_'+time.strftime("%Y%m%d_%H%M%S"))

        ## First, make required directories
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print (time.strftime("%H:%M:%S"), " Creating directory %s" % images_dir)

        for series_label in series_labels:
            series_dir = os.path.join(images_dir, series_label)
            if not os.path.exists(series_dir):
                os.makedirs(series_dir)
    else:
        images_dir = args.images_dir_existing

    ## Parse all study xml files
    xml_files = glob.glob(args.rootdir + '/[0-9]*/ST-[0-9]*/__index.xml')

    with Pool(mp.cpu_count()) as p:
        series_result = p.map(parse_study_xml, tqdm(xml_files))

    series_dict, study_dict, refID_to_trial = {}, {}, {}
    [(study_dict.update(x[0]), series_dict.update(x[1]), refID_to_trial.update(x[2])) for x in series_result]

    ## Load our overrides mapping from trials to series and apply it
    series_overrides = json.load(open(os.path.join(args.outputdir,'series_overrides.json'),'rb'))
    for k,v in series_dict.items():
        if k in series_overrides:
            series_dict[k] = series_overrides[k]
    
    series_processed = {}
    refID_to_trial_processed = {}
    
    if args.images_dir_existing:
        series_processed = json.load(open(os.path.join(images_dir, 'series.json'),'rb'))
        refID_to_trial_processed = json.load(open(os.path.join(images_dir, 'ref_to_trial.json'),'rb'))
    
    with open(os.path.join(images_dir, 'series.json'), 'w') as fp:
        json.dump(series_dict, fp, sort_keys=True, indent=4)
    
    with open(os.path.join(images_dir, 'ref_to_trial.json'), 'w') as fp:
        json.dump(refID_to_trial, fp, sort_keys=True, indent=4)
    
    study_to_labeled_series = defaultdict(list)
    for label in series_labels:
        for study, series_list in study_dict.items():
            for series in series_list:
                if series_dict[series] == label:
                    if series in series_processed and refID_to_trial[series] == refID_to_trial_processed[series]:
                        continue
                    study_to_labeled_series[label].append((study,series))
    
    series_labels_to_process = [args.series_type] if args.series_type else series_labels
    
    for label in series_labels_to_process:
        t1 = time.time()
        print(time.strftime("%H:%M:%S"), 'Number of series for view %s: %d' % (label, len(study_to_labeled_series[label])))
        for study_tuple in tqdm(study_to_labeled_series[label], total=len(study_to_labeled_series[label]), unit=' series'):
            preprocess_series(images_dir, label, study_tuple)
        t2 = time.time()
        print(time.strftime("%H:%M:%S"), 'Completed processing of %s, Elapsed time: %d s' % (label,t2 - t1))       

    print (time.strftime("%H:%M:%S"), " done processing all")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    preprocess_wrapper()
