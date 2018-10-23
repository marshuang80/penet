import os
import pickle 
import numpy as np
import csv 

dirr = 'DataPEcor/localized/images'

for file in os.listdir(dirr):
    x = np.load(os.path.join(dirr, file))
    print (file, x.shape)
