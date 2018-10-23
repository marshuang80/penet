'''
Tool for visualizing CT scans
INSTRUCTIONS (Mac):
    -To set up the remote display:
        -Install XQuartz on your local machine
        -Enable remote connections by running this command on your local machine: "sudo cp ~/.Xauthority ~root/" [IMPORTANT: this should be done from your user account, not as root]
        -ssh into your remote machine with the "-X" flag, e.g. "ssh -X <username>@<example.com>"
        -run script:
            python remote_display.py --hu_datadir <xyz> --filename <xyz> --label_info <xyz>
        -default args are for chest CT scans    

Adapted from stackoverflow answer: https://stackoverflow.com/questions/5501192/how-to-display-picture-and-get-mouse-click-coordinate-on-it
'''

import sys
import os
import numpy as np
import re
from collections import defaultdict
import csv
from PIL import Image
import argparse

import pickle
import re
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
from skimage.feature import hog 
from skimage import data, exposure
curr_pos = 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/DataPE/images')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--label_dict_file', type=str, default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/New4ChestData/label_dict') 
    return parser

# def get_plots(info_dict):
#     global file_indexes
#     plots = []
#     for img_slice in orig_scan:
#         if 'npy' not in args.filename:
#             continue
#         file_indexes.append(args.filename)
#        # print (filename)
#         info = info_dict[args.filename]
#         print(args.filename, type(info))
#         y = info['lung_percentage_details']
#       #  print (y)
#         x = list(range(len(y)))
#         plots.append((x,y))

#     return plots

def key_event(e, scan,localized_range):

    #print("key event:", e.key)
    global curr_pos

    if e.key == "right":
        #print("key is right")
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return

    #print("Curr pos b4 mod: ", curr_pos)
    curr_pos = curr_pos % len(scan)
    #print("# localized: ", len(scan), " # Original: ", len(orig_scan))
    #print(len(scan))
    #print("Curr pos after: ", curr_pos)
    ax.cla()
    #ax.axhline(y=8, linewidth=2, color='r', label="lower_threshold")
    ax.set_title("Image: {},label: {}, slice: {}".format(args.filename, 1, localized_range[curr_pos]))#'''label_dict[args.filename[:-4]]''', )
    ax.imshow(scan[curr_pos], cmap='gray')#
    fig.canvas.draw()


def visualize_lung_percentage_cover(filename):
    pass


def crop_center(img,cropx,cropy):
    num_slices, y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[0:num_slices,starty:starty+cropy,startx:startx+cropx]

def windowing(scan):
    np_image_array1 = np.clip(scan, 750, 1450)
    np_image_array1 = np_image_array1 - 750
    np_image_array1 = np_image_array1 * 255.0 / 700.0
    np_image_array1 = np_image_array1.astype(np.uint8, copy=False)
    np_image_array1 = crop_center(np_image_array1,300,300)
    return np_image_array1

def HOG_descriptor(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

    print("HOG Feature descriptor shape: ", fd.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    '''
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    '''
    return fd

if __name__ == "__main__":
    unlocalized_datadir = '/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/DataPE/images'
    args = get_parser().parse_args()
    #label dict entry example: file = 997_2_1.npy label=1 --> key , val = 997_2_1 , 1
    #Should create this dict from relevant csv files and pickle it before running script
    with open(args.label_dict_file, 'rb') as fp:
        label_dict = pickle.load(fp)
        
    unlocalized_filepath = unlocalized_datadir + '/' + args.filename
    localized_filepath = args.datadir + '/' + args.filename
    print(localized_filepath)
    orig_scan = windowing(np.load(unlocalized_filepath))
    scan = windowing(np.load(localized_filepath))
    print(len(orig_scan), len(scan))
    if 'localized' in args.datadir:
        range_info_file = args.datadir[:args.datadir.rfind('/')] + '/range_info'
        with open(range_info_file, 'rb') as fp:
            range_info= pickle.load(fp)
        print("Range info: ",range_info[args.filename[:-4]])
        localized_range = list(range_info[args.filename[:-4]])
    else:
        localized_range = list(range(len(scan)))

    #HOG_descriptor(scan[0])
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', lambda event: key_event(event, scan, localized_range))
    ax = fig.add_subplot(111)
    img_slice = scan[args.start]
    curr_pos = args.start
    ax.imshow(img_slice, cmap='gray')
    plt.show()

