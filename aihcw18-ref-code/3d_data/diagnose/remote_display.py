'''
Tool for visualizing HU data on CT scans
INSTRUCTIONS (Mac):
	-To set up the remote display:
		-Install XQuartz on your local machine
		-Enable remote connections by running this command on your local machine: "sudo cp ~/.Xauthority ~root/" [IMPORTANT: this should be done from your user account, not as root]
		-ssh into your remote machine with the "-X" flag, e.g. "ssh -X <username>@<example.com>"
		-run script:
			python remote_display.py --hu_datadir <xyz> --grayscale_datadir <xyz> --slice_index <xyz>
		-default args are for chest CT scans 	

Adapted from stackoverflow answer: https://stackoverflow.com/questions/5501192/how-to-display-picture-and-get-mouse-click-coordinate-on-it

'''


from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import argparse, time, json
import os

def get_parser():
	parser = argparse.ArgumentParser()
    # Experiment Parameters
	parser.add_argument('--hu_datadir', type=str, default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/New4ChestData/images/')  
	parser.add_argument('--grayscale_datadir', type=str, default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/New4ChestData/images/')
	parser.add_argument('--slice_index', type=int, default=0)
	parser.add_argument('--filename', type=str, required=True)
	parser.add_argument('--image_len', type=int, default=256)
	return parser

def set_up_tkinter(args):	
	#setting up a tkinter canvas with scrollbars
	canvas_height, canvas_width = args.image_len, args.image_len
	frame = Frame(root, bd=2, relief=SUNKEN)
	frame.grid_rowconfigure(0, weight=1)
	frame.grid_columnconfigure(0, weight=1)
	xscroll = Scrollbar(frame, orient=HORIZONTAL)
	xscroll.grid(row=1, column=0, sticky=E+W)
	yscroll = Scrollbar(frame)
	yscroll.grid(row=0, column=1, sticky=N+S)
	canvas = Canvas(frame, bd=0, width=canvas_width, height=canvas_height, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
	canvas.grid(row=0, column=0, sticky=N+S+E+W)
	xscroll.config(command=canvas.xview)
	yscroll.config(command=canvas.yview)
	frame.pack(fill=BOTH,expand=1)
	return frame, xscroll, yscroll, canvas

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def create_grayscale_image(gs_array_filename, args):
	#Create image from grayscale-normalized data	
	gs_array_filepath = args.grayscale_datadir + gs_array_filename
	gs_ct_scan = np.load(gs_array_filepath)
	gs_slice = gs_ct_scan[args.slice_index] 
	np_image_array1 = np.clip(gs_slice, 750, 1450)
	np_image_array1 = np_image_array1 - 750
	np_image_array1 = np_image_array1 * 255.0 / 700.0
	np_image_array1 = np_image_array1.astype(np.uint8, copy=False)
	#np_image_array1 = crop_center(np_image_array1,256,256)
	img = Image.fromarray(np_image_array1)
	img = ImageTk.PhotoImage(img)
	return img

if __name__ == '__main__':
	args = get_parser().parse_args()	
	root = Tk()		
	frame, xscroll, yscroll, canvas = set_up_tkinter(args)
	hu_array_filepath = args.hu_datadir + args.filename
	'''
	#Get the file containing HU data, & index of relevant slice
	hu_array_filepath = filedialog.askopenfilename(initialdir=args.hu_datadir, title='Choose numpy array file representing CT scan:')
	'''
	hu_ct_scan = np.load(hu_array_filepath)
	hu_slice = hu_ct_scan[args.slice_index]

	
	gs_array_filename = os.path.basename(hu_array_filepath)
	img = create_grayscale_image(gs_array_filename, args)

	canvas.create_image(0,0,image=img,anchor="nw")
	canvas.config(scrollregion=canvas.bbox(ALL))

	def print_HU_val(row,col):
	    print("HU val at ({},{}): {}".format(row,col, hu_slice[row][col]))

	#function to be called when mouse is clicked
	def printcoords(event):
	    #outputting x and y coords to console
	    print_HU_val(event.y ,event.x )

	#mouseclick event
	canvas.bind("<Button 1>",printcoords)
	root.mainloop()