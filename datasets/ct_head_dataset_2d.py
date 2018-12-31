import cv2
import h5py
import numpy as np
import os

from .ct_head_dataset import CTHeadDataset


class CTHeadDataset2d(CTHeadDataset):
    """Dataset for loading 2d data (individual slices) from CTA."""

    def __init__(self, args, phase, is_training_set=True):
        """
        Args:
            args: Command line arguments.
            phase: one of 'train','val','test'
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTHeadDataset2d, self).__init__(args, phase, is_training_set=is_training_set)

        # Map from slices to series indices
        self.slice_to_series_idx = []  # Maps window/slice indices to series indices
        self.series_to_slice_idx = []  # Maps series indices to base window/slice index for that series
        self.num_slices = 0
        for i, s in enumerate(self.series_list):
            self.slice_to_series_idx += len(s) * [i]
            self.series_to_slice_idx.append(self.num_slices)
            self.num_slices += len(s)

    def _include_series(self, s):
        """Predicate for whether to include a series in this dataset."""
        if self.phase == 'all':
            return True

        if s.phase != self.phase:
            return False
        if s.mode != ('contrast' if self.use_contrast else 'non_contrast'):
            return False
        if s.is_aneurysm and (s.aneurysm_ranges is None or s.aneurysm_bounds is None):
            return False
        if not self.is_test_mode and s.aneurysm_mask_path is None:
            return False

        return True

    def __len__(self):
        return len(self.slice_to_series_idx)

    def __getitem__(self, idx):
        series_idx = self.slice_to_series_idx[idx]
        series = self.series_list[series_idx]
        # Get slice index within the series
        slice_idx = idx - self.series_to_slice_idx[series_idx]

        slice_, target = self._load_slice(series, slice_idx)
        slice_, target = self._transform(slice_, target, series.brain_bbox, series.is_bottom_up)
        if target is None:
            # Pass the dset_path as a unique ID for the corresponding series
            target = {'dset_path': series.dset_path, 'slice_idx': slice_idx}

        return slice_, target

    def _load_slice(self, series, slice_idx):
        """Load num_slices slices from a series, starting at start_idx.

        Args:
            series: The series to load slices from.
            slice_idx: Index of first slice to load.

        Returns:
            slice, mask: 2D NumPy arrays for the slice and corresponding mask.
        """
        if self.img_format == 'png':
            raise NotImplementedError('No support for PNGs in our HDF5 files.')

        mask = None
        with h5py.File(os.path.join(self.data_dir, 'data.hdf5'), 'r') as hdf5_fh:
            slice_ = hdf5_fh[series.dset_path][slice_idx]
            if not self.is_test_mode:
                mask = hdf5_fh[self._get_mask_path(series)][slice_idx]
                mask = mask.astype(np.float32)

        return slice_, mask

    def _crop(self, slice_, mask, x1, y1, x2, y2):
        """Crop a 2D slice and its corresponding mask (before channel dimension has been added)."""
        slice_ = slice_[y1: y2, x1: x2]

        if mask is not None:
            mask = mask[y1: y2, x1: x2]

        return slice_, mask

    def _rescale(self, slice_, interpolation=cv2.INTER_AREA):
        slice_ = cv2.resize(slice_, tuple(self.resize_shape), interpolation=interpolation)
        return np.array(slice_)

    def _handle_bottom_up(self, inputs, mask, is_bottom_up):
        """Do nothing different for bottom-up vs. bottom-down in 2D case."""
        pass

    def _pad(self, inputs, mask):
        return inputs, mask
