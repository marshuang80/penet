import cv2
import h5py
import numpy as np
import os
import pickle
import random
import torch
import util

from .base_ct_dataset import BaseCTDataset
from ct.ct_pe_constants import *
from scipy.ndimage.interpolation import rotate


class CTPEDataset3d(BaseCTDataset):
    def __init__(self, args, phase, is_training_set=True):
        """
        Args:
            args: Command line arguments.
            phase: one of 'train','val','test'
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTPEDataset3d, self).__init__(args.data_dir, args.img_format, is_training_set=is_training_set)
        self.phase = phase
        self.resize_shape = args.resize_shape
        self.is_test_mode = not args.is_training
        self.pe_types = args.pe_types
        self.meta_features = args.features.split(",")

        # Augmentation
        self.crop_shape = args.crop_shape
        self.do_hflip = self.is_training_set and args.do_hflip
        self.do_vflip = self.is_training_set and args.do_vflip
        self.do_rotate = self.is_training_set and args.do_rotate
        self.do_jitter = self.is_training_set and args.do_jitter
        self.do_center_abnormality = self.is_training_set and args.do_center_pe

        self.threshold_size = args.threshold_size
        self.pixel_dict = {
            'min_val': CONTRAST_HU_MIN,
            'max_val': CONTRAST_HU_MAX,
            'avg_val': CONTRAST_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT
        }

        # Load info for the CTPE series in this dataset
        with open(args.pkl_path, 'rb') as pkl_file:
            all_ctpes = pickle.load(pkl_file)

        # TODO 
        self.ctpe_list = [ctpe for ctpe in all_ctpes if self._include_ctpe(ctpe)]
        #self.ctpe_list = [ctpe for ctpe in all_ctpes]

        self.positive_idxs = [i for i in range(len(self.ctpe_list)) if self.ctpe_list[i].is_positive]
        self.min_pe_slices = args.min_abnormal_slices
        self.num_slices = args.num_slices
        self.abnormal_prob = args.abnormal_prob if self.is_training_set else None
        self.use_hem = args.use_hem if self.is_training_set else None

        # Map from windows to series indices, and from series indices to windows
        self.window_to_series_idx = []  # Maps window indices to series indices
        self.series_to_window_idx = []  # Maps series indices to base window index for that series
        window_start = 0
        for i, s in enumerate(self.ctpe_list):
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0)
            self.window_to_series_idx += num_windows * [i]
            self.series_to_window_idx.append(window_start)
            window_start += num_windows

        if args.toy:
            self.window_to_series_idx = np.random.choice(self.window_to_series_idx, args.toy_size, replace=False)

        if self.use_hem:
            # Initialize a HardExampleMiner with IDs formatted like (series_idx, start_idx)
            example_ids = []
            for window_idx in range(len(self)):
                series_idx = self.window_to_series_idx[window_idx]
                series = self.ctpe_list[series_idx]
                if not series.is_positive:
                    # Only include negative examples in the HardExampleMiner
                    start_idx = (window_idx - self.series_to_window_idx[series_idx]) * self.num_slices
                    example_ids.append((series_idx, start_idx))
            self.hard_example_miner = util.HardExampleMiner(example_ids)

    def _include_ctpe(self, pe):
        """Predicate for whether to include a series in this dataset."""
        if pe.phase != self.phase and self.phase != 'all':
            return False
        
        if pe.is_positive and pe.type not in self.pe_types:
            return False

        return True

    def _parse_metadata(self, ctpe):

        race_dict = {'Asian':0,
                'Black':1,
                'Native American':2,
                'Other':3,
                'Pacific Islander':4,
                'Unknown':5,
                'White':6}


        age = [ctpe.age / 100.0]
        try:
            sex = [1] if ctpe.sex == "Male" else [0]
        except:
            print(ctpe.sex)
        is_smoker = [1] if ctpe.is_smoker == "Y" else [0]
        race_raw = ctpe.race 
        race = [0] * 7
        race[race_dict[race_raw]] = 1
        
        meta_dict = {"age":age,
                     "is_smoker": is_smoker,
                     "race": race,
                     "sex", sex}

        meta = []
        for feature in self.meta_features:
            meta += feature

        return np.array(meta)

    def __len__(self):
        return len(self.window_to_series_idx)

    def __getitem__(self, idx):
        # Choose ctpe and window within ctpe
        ctpe_idx = self.window_to_series_idx[idx]
        ctpe = self.ctpe_list[ctpe_idx]

        if self.abnormal_prob is not None and random.random() < self.abnormal_prob:
            # Force aneurysm window with probability `abnormal_prob`.
            if not ctpe.is_positive:
                ctpe_idx = random.choice(self.positive_idxs)
                ctpe = self.ctpe_list[ctpe_idx]
            start_idx = self._get_abnormal_start_idx(ctpe, do_center=self.do_center_abnormality)
        elif self.use_hem:
            # Draw from distribution that weights hard negatives more heavily than easy examples
            ctpe_idx, start_idx = self.hard_example_miner.sample()
            ctpe = self.ctpe_list[ctpe_idx]
        else:
            # Get sequential windows through the whole series
            # TODO
            start_idx = (idx - self.series_to_window_idx[ctpe_idx]) * self.num_slices

        if self.do_jitter:
            # Randomly jitter start offset by num_slices / 2
            start_idx += random.randint(-self.num_slices // 2, self.num_slices // 2)
            start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)

        volume = self._load_volume(ctpe, start_idx)
        volume = self._transform(volume)

        is_abnormal = torch.tensor([self._is_abnormal(ctpe, start_idx)], dtype=torch.float32)

        # TODO 

        meta = self._parse_metadata(ctpe)

        meta = torch.from_numpy(meta)

        # metadata dictionary
        meta_dict = {"age": age,
                "is_smoker": is_smoker,
                "race":race, 
                "sex": sex
                }

        # Pass series info to combine window-level predictions
        target = {'is_abnormal': is_abnormal,
                  'study_num': ctpe.study_num,
                  'dset_path': str(ctpe.study_num),
                  'slice_idx': start_idx,
                  'series_idx': ctpe_idx}

        return volume, target, meta
        #return volume, target

    def get_series_label(self, series_idx):
        """Get a floating point label for a series at given index."""
        return float(self.ctpe_list[series_idx].is_positive)

    def get_series(self, study_num):
        """Get a series with specified study number."""
        for ctpe in self.ctpe_list:
            if ctpe.study_num == study_num:
                return ctpe
        return None

    def update_hard_example_miner(self, example_ids, losses):
        """Update HardExampleMiner with set of example_ids and corresponding losses.

        This should be called at the end of every epoch.

        Args:
            example_ids: List of example IDs which were used during training.
            losses: List of losses for each example ID (must be parallel to example_ids).
        """
        example_ids = [(series_idx, start_idx) for series_idx, start_idx in example_ids
                       if series_idx not in self.positive_idxs]
        if self.use_hem:
            self.hard_example_miner.update_distribution(example_ids, losses)

    def _get_abnormal_start_idx(self, ctpe, do_center=True):
        """Get an abnormal start index for num_slices from a series.

        Args:
            ctpe: CTPE series to sample from.
            do_center: If true, center the window on the abnormality.

        Returns:
            Randomly sampled start index into series.
        """
        abnormal_bounds = (min(ctpe.pe_idxs), max(ctpe.pe_idxs))

        # Get actual slice number
        if do_center:
            # Take a window from center of abnormal region
            center_idx = sum(abnormal_bounds) // 2
            start_idx = max(0, center_idx - self.num_slices // 2)
        else:
            # Randomly sample num_slices from the abnormality (taking at least min_pe_slices).
            start_idx = random.randint(abnormal_bounds[0] - self.num_slices + self.min_pe_slices,
                                       abnormal_bounds[1] - self.min_pe_slices + 1)

        return start_idx

    def _load_volume(self, ctpe, start_idx):
        """Load num_slices slices from a CTPE series, starting at start_idx.

        Args:
            ctpe: The CTPE series to load slices from.
            start_idx: Index of first slice to load.

        Returns:
            volume: 3D NumPy arrays for the series volume.
        """
        if self.img_format == 'png':
            raise NotImplementedError('No support for PNGs in our HDF5 files.')

        with h5py.File(os.path.join(self.data_dir, 'data.hdf5'), 'r') as hdf5_fh:
            volume = hdf5_fh[str(ctpe.study_num)][start_idx:start_idx + self.num_slices]

        return volume

    def _is_abnormal(self, ctpe, start_idx):
        """Check whether a window from `ctpe` starting at start_idx includes an abnormality.

        Args:
            ctpe: CTPE object to check for any abnormality.

        Returns:
            True iff (1) ctpe contains an aneurysm and (2) abnormality is big enough.
        """
        if ctpe.is_positive:
            abnormal_slices = [i for i in ctpe.pe_idxs if start_idx <= i < start_idx + self.num_slices]
            is_abnormal = len(abnormal_slices) >= self.min_pe_slices
        else:
            is_abnormal = False

        return is_abnormal

    def _crop(self, volume, x1, y1, x2, y2):
        """Crop a 3D volume (before channel dimension has been added)."""
        volume = volume[:, y1: y2, x1: x2]

        return volume

    def _rescale(self, volume, interpolation=cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)

    def _pad(self, volume):
        """Pad a volume to make sure it has the expected number of slices.
        Pad the volume with slices of air.

        Args:
            volume: 3D NumPy array, where slices are along depth dimension (un-normalized raw HU).

        Returns:
            volume: 3D NumPy array padded/cropped to have the expected number of slices.
        """

        def add_padding(volume_, pad_value=AIR_HU_VAL):
            """Pad 3D volume with air on both ends to desired number of slices.
            Args:
                volume_: 3D NumPy ndarray, where slices are along depth dimension (un-normalized raw HU).
                pad_value: Constant value to use for padding.
            Returns:
                Padded volume with depth args.num_slices. Extra padding voxels have pad_value.
            """
            num_pad = self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode='constant', constant_values=pad_value)

            return volume_

        volume_num_slices = volume.shape[0]

        if volume_num_slices < self.num_slices:
            volume = add_padding(volume, pad_value=AIR_HU_VAL)
        elif volume_num_slices > self.num_slices:
            # Choose center slices
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]

        return volume

    def _transform(self, inputs):
        """Transform slices: resize, random crop, normalize, and convert to Torch Tensor.

        Args:
            inputs: 2D/3D NumPy array (un-normalized raw HU), shape (height, width).

        Returns:
            volume: Transformed volume, shape (num_channels, num_slices, height, width).
        """
        if self.img_format != 'raw':
            raise NotImplementedError('Unsupported img_format: {}'.format(self.img_format))

        # Pad or crop to expected number of slices
        inputs = self._pad(inputs)

        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation=cv2.INTER_AREA)

        if self.crop_shape is not None:
            row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])
            col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
            # Random crop during training, center crop during test inference
            row = random.randint(0, row_margin) if self.is_training_set else row_margin // 2
            col = random.randint(0, col_margin) if self.is_training_set else col_margin // 2
            inputs = self._crop(inputs, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])

        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-2)

        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-1)

        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL)

        # Normalize raw Hounsfield Units
        inputs = self._normalize_raw(inputs)

        inputs = np.expand_dims(inputs, axis=0)  # Add channel dimension
        inputs = torch.from_numpy(inputs)

        return inputs

