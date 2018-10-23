

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



def create_spectrogram(e):
	global index
	global ids
	global range_info
	global X
	global fig
	global ax
	if e.key == "right":
		print("key is right")
		scan_id = ids[index]
		scan_name = scan_id[:scan_id.rfind('_')]
		num_slices = len(range_info[scan_name + '.npy'])
		print(scan_name, num_slices)
		data = X[index:index+num_slices]
		#data = data[:,20:]
		data = data.transpose()
		print(data.shape)
		x,y = data.shape
		ax.cla()
		ax.set_title(scan_name)
		ax.imshow(data, cmap='hot', interpolation='nearest', origin='lower', aspect='auto', extent=[0,y,0,x*10])	
		ax.set_xlabel("slice")
		ax.set_ylabel("HU units")
		index = index + num_slices
		fig.canvas.draw()

if __name__ == '__main__':
	datadir ='/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/localized_New4ChestData/strict/HU_info_lt_-800_ut_-600_p_3.0%_ns_60/images/'
	range_info_file = '/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/localization_info_updatedNew4ChestData/range_info_New4ChestData_part_1_lt_-800'
	cluster_info_dir = '/deep/group/aihc-bootcamp-winter2018/nborus/ct_chest_pe/New4ChestData/kmeans_histogram_cluster_slices/combined/range_0_-1/localized/'
	







