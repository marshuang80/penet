import csv
import numpy as np

file_to_thicc = dict()

with open('data-final/final_info.txt', 'r') as csv_file:
    reader = csv.reader(csv_file)
    info = list(reader)
    for file, acc, ser, thicc in info:
        file_to_thicc['data-final/images/' + file] = float(thicc)

for file, thicc in file_to_thicc.items():
    x = np.load(file)
    num_slices = x.shape[0]
    reqd_slices = int(num_slices * thicc / 5.0)
    reqd_range = np.linspace(0, num_slices - 1, reqd_slices, dtype=np.int32)
    new_x = x[reqd_range, :, :]
    new_file_path = file.replace('data-final', '30AprData')
    np.save(new_file_path, new_x)
