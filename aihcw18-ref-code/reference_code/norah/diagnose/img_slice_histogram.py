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
curr_pos = 0


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--datadir', type=str, default='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/localized_New4ChestData/strict/HU_info_lt_-800_ut_-600_p_3.0%_ns_60/images')
	parser.add_argument('--filename', type=str, required=True)
	parser.add_argument('--start', type=int, default=0)
	parser.add_argument('--label_dict_file', type=str, default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/New4ChestData/label_dict') 
	return parser

def store_histogram_images(datadir,scan_name):
	for slice_index, img_slice in enumerate(scan):
		fig, ax = plt.subplots()
		val = '{}_{}'.format(scan_name, slice_index)
		img_slice = img_slice.flatten()
		ax.hist(img_slice, range=(lower_threshold,upper_threshold))
		ax.set_title(val)
		fig.canvas.draw()


def key_event(e, scan):

	#print("key event:", e.key)
	global curr_pos
	global args
	global lower_threshold
	global upper_threshold

	if e.key == "right":
		#print("key is right")
		curr_pos = curr_pos + 1
	elif e.key == "left":
	    curr_pos = curr_pos - 1
	else:
	    return

	ax.cla()
	print("Curr pos b4 mod: ", curr_pos)
	curr_pos = curr_pos % len(scan)
	img_slice = scan[curr_pos]
	val = '{}_{}'.format(args.filename, curr_pos)
	img_slice = img_slice.flatten()
	ax.hist(img_slice, range=(lower_threshold,upper_threshold))
	ax.set_title(val)
	fig.canvas.draw()

def visualize_lung_percentage_cover(filename):
    pass



if __name__ == "__main__":
	args = get_parser().parse_args()
    #label dict entry example: file = 997_2_1.npy label=1 --> key , val = 997_2_1 , 1
    #Should create this dict from relevant csv files and pickle it before running script
	with open(args.label_dict_file, 'rb') as fp:
		label_dict = pickle.load(fp)

	
	lower_threshold = 150
	upper_threshold = 1500
	scan_filepath = args.datadir  + '/' + args.filename + '.npy'
	scan = np.load(scan_filepath)

	fig = plt.figure()
	fig.canvas.mpl_connect('key_press_event', lambda event: key_event(event, scan))
	ax = fig.add_subplot(111)
	curr_pos = args.start
	img_slice = scan[args.start]
	val = '{}_{}'.format(args.filename, args.start)
	img_slice = img_slice.flatten()
	ax.hist(img_slice, range=(lower_threshold,upper_threshold))
	ax.set_title(val)
	fig.canvas.draw()
	plt.show()

