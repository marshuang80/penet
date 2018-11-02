import cv2
import numpy as np
import pickle
import random
import torch
import util

from .base_ct_dataset import BaseCTDataset
from ct.ct_head_constants import *
from scipy.ndimage.interpolation import rotate


class CTHeadDataset(BaseCTDataset):
    """Base dataset for loading a CTA dataset."""

    def __init__(self, args, phase, is_training_set=True):
        """
        Args:
            args: Command line arguments.
            phase: one of 'train', 'val', 'test'.
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTHeadDataset, self).__init__(args.data_dir, args.img_format, is_training_set)
        self.img_format = args.img_format
        self.phase = phase
        self.use_contrast = args.use_contrast
        self.resize_shape = args.resize_shape
        self.is_test_mode = not args.is_training
        self.include_normals = args.include_normals if args.is_training else None

        # Augmentation
        self.crop_shape = args.crop_shape
        self.elastic_transform = self.is_training_set and args.elastic_transform
        self.do_hflip = self.is_training_set and args.do_hflip
        self.do_vflip = self.is_training_set and args.do_vflip
        self.do_rotate = self.is_training_set and args.do_rotate

        self.threshold_size = args.threshold_size
        self.pixel_dict = {
            'min_val': CONTRAST_HU_MIN if self.use_contrast else NON_CON_HU_MIN,
            'max_val': CONTRAST_HU_MAX if self.use_contrast else NON_CON_HU_MAX,
            'avg_val': CONTRAST_HU_MEAN if self.use_contrast else NON_CON_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT
        }

        # Load info for the CT series in this dataset, in descending order of sequence length.
        series_list = []
        lengths = []
        with open(args.pkl_path, 'rb') as pkl_file:
            all_series = pickle.load(pkl_file)
            for s in all_series:
                if self._include_series(s):
                    series_list.append(s)
                    lengths.append(len(s))

            self.series_list = [series for _, series in sorted(zip(lengths, series_list),
                                                               reverse=True, key=lambda x: x[0])]

        if args.toy:
            self.series_list = np.random.choice(self.series_list, args.toy_size, replace=False)

    def _include_series(self, s):
        """Predicate for whether to include a series in this dataset."""
        raise NotImplementedError('Subclass of CTHeadDataset must implement _include_series.')

    def __len__(self):
        raise NotImplementedError('Subclass of CTHeadDataset must implement __len__.')

    def __getitem__(self, idx):
        raise NotImplementedError('Subclass of CTHeadDataset must implement __getitem__.')

    def _crop(self, inputs, mask, x1, y1, x2, y2):
        raise NotImplementedError('Subclass of CTHeadDataset must implement _crop.')

    def _rescale(self, inputs, interpolation=cv2.INTER_AREA):
        raise NotImplementedError('Subclass of CTHeadDataset must implement _rescale.')

    def _pad(self, inputs, mask):
        raise NotImplementedError('Subclass of CTHeadDataset must implement _pad.')

    def _handle_bottom_up(self, inputs, mask, is_bottom_up):
        raise NotImplementedError('Subclass of CTHeadDataset must implement _handle_bottom_up.')

    def _transform(self, inputs, mask, brain_bbox, is_bottom_up):
        """Transform slices: resize, random crop, normalize, and convert to Torch Tensor.

        Args:
            inputs: 2D/3D NumPy array (un-normalized raw HU), shape (height, width).
            mask: 2D/3D NumPy array of the corresponding aneurysm mask. May be None.
            brain_bbox: Bounding box annotation around the brain.
            is_bottom_up: If True, the series goes from bottom to top of patient's head.
                Otherwise goes top to bottom.

        Returns:
            volume: Transformed volume, shape (num_channels, num_slices, height, width).
            mask: Transformed mask, shape (num_slices, height, width).
        """
        if self.img_format != 'raw':
            raise NotImplementedError('Unsupported img_format: {}'.format(self.img_format))

        # Pad or crop to expected number of slices
        inputs, mask = self._pad(inputs, mask)

        # Crop tightly around the brain
        if brain_bbox is not None:
            x1, y1, side_length = util.get_crop(brain_bbox)
            inputs, mask = self._crop(inputs, mask, x1, y1, x1 + side_length, y1 + side_length)

        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = self._rescale(mask, interpolation=cv2.INTER_LINEAR)

        if self.crop_shape is not None:
            # Crop volume and mask with the same random crop
            row = random.randint(0, inputs.shape[-2] - self.crop_shape[-2] - 1)
            col = random.randint(0, inputs.shape[-1] - self.crop_shape[-1] - 1)
            inputs, mask = self._crop(inputs, mask, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])

        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-2)
            if mask is not None:
                mask = np.flip(mask, axis=-2)

        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-1)
            if mask is not None:
                mask = np.flip(mask, axis=-1)

        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL)
            if mask is not None:
                mask = rotate(mask, angle, (-2, -1), reshape=False, cval=0)

        # Normalize raw Hounsfield Units
        inputs = self._normalize_raw(inputs)

        inputs, mask = self._handle_bottom_up(inputs, mask, is_bottom_up)

        inputs = np.expand_dims(inputs, axis=0)  # Add channel dimension
        inputs = torch.from_numpy(inputs)

        if mask is not None:
            mask = (mask > 0.5).astype(np.float32)
            mask = torch.from_numpy(mask)

        return inputs, mask
