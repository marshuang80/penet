"""
Interactive tool to review annotations and convert to usable inputs for the model. 

Expected format of annotations csv is Series_Ref, StartSlice, EndSlice, Comments:
    * Series_Ref is a unique identifier to the stored series npy block
    * StartSlice is a 0-based index to the start slice of the label within the block
    * EndSlice is a 0-based index to the end slice of the label within the block (inclusive)
    * Comments are the annotated comments w.r.t to the label's location in the slice
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import json, os
from collections import defaultdict
import numpy as np
from PIL import Image
import csv
from IPython.display import clear_output
import util

## Load the relevant info
images_dir = '/data/CT-CSPINE/processed-studies/data_20180523_233650/'
slice_number_annotations = pd.read_csv('/data/CT-CSPINE/processed-studies/data_20180512_203649/sagittal_sliceNumber_annotations.csv',index_col='Series_Ref')
slice_number_annotations = slice_number_annotations
slice_number_annotations = slice_number_annotations.reset_index()

## Helper function for displaying image with grid
def display_gridded_image(np_slice):
    my_dpi=200.
    img = Image.fromarray(util.apply_window(np_slice, 400., 1800.))

    fig=plt.figure(figsize=(float(img.size[0])/my_dpi,float(img.size[1])/my_dpi),dpi=my_dpi)
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Set the gridding interval: here we use the major tick interval
    myInterval=170.
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-',color='b')

    # Add the image
    ax.imshow(img)

    # Find number of gridsquares in x and y direction
    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(myInterval)))
    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(myInterval)))

    # Add some labels to the gridsquares
    for j in range(ny):
        y=myInterval/2+j*myInterval
        for i in range(nx):
            x=myInterval/2.+float(i)*myInterval
            ax.text(x,y,'{:d}'.format(i+j*nx + 1),color='b',ha='center',va='center')
    
    plt.show(fig)

##################################################################################################
## CONTROLS
label = 'sagittal'
reviewer_name = 'Robin'
fname = os.path.join(images_dir, label + '_annotated_slices_' + reviewer_name + '.csv')
#################################################################################################

completed_series = defaultdict(list)

if not os.path.isfile(fname):
    with open(fname, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Series_Ref','SliceNum','GridLocation'])
        
def review_images():
    """ Iterate through annotated images to localize fractures.
    """
    
    for idx in slice_number_annotations.index:
        series_ref = slice_number_annotations.loc[idx, 'Series_Ref']
        npy_path = images_dir + label + '/' + series_ref + '.npy'
        start_slice = slice_number_annotations.loc[idx,'StartSlice']
        end_slice = slice_number_annotations.loc[idx,'EndSlice']
        if (series_ref + "_{}_{}".format(start_slice, end_slice)) not in completed_series[label]:
            comments = slice_number_annotations.loc[idx,'Comments']
            block = np.load(npy_path)
            grid_locations = None
            for sliver in range(start_slice, end_slice):
                np_slice = block[sliver]
                display_gridded_image(np_slice)
                while True:
                    print("Reviewing series %s slice %d: " % (series_ref, sliver))
                    print("Comments: %s" % (comments))
                    if grid_locations is not None:
                        print("Previous locations: {}".format(grid_locations))
                        print("Enter the grid squares (comma separated list):")
                        print("Hit enter (empty string) to use previous locations")
                        new_locs = input()
                        if new_locs:
                            grid_locations = new_locs
                    else:
                        print("Enter the grid squares (comma separated list):")
                        grid_locations = input()
                    made_mistake_label = input('Make a mistake? Y or N ').upper()
                    if made_mistake_label != 'Y':
                        break
                with open(fname,'a') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow([series_ref, sliver, str(grid_locations)])
            completed_series[label].append(series_ref + "_{}_{}".format(start_slice, end_slice))
            clear_output()