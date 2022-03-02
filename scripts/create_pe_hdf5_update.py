import h5py
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


def npy2hdf5(data_dir, output_dir):
    hdf5_fh = h5py.File(os.path.join(output_dir, 'data.hdf5'), 'a')

    for npy in tqdm(data_dir.glob("*.npy"),total=len(list(data_dir.glob("*.npy")))):
        study = np.load(npy)
        filename = str(npy).split("/")[-1].split(".")[0]
        print(filename)
        hdf5_fh.create_dataset(filename, data=study, dtype='i2', chunks=True)
    hdf5_fh.close()


if __name__ == "__main__":
    data_dir = Path("")
    output_dir = ""
    npy2hdf5(data_dir, output_dir)
