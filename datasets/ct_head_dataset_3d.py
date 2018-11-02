import cv2
import h5py
import numpy as np
import os
import random
import torch
import util

from .ct_head_dataset import CTHeadDataset
from ct.ct_head_constants import AIR_HU_VAL


class CTHeadDataset3d(CTHeadDataset):
    def __init__(self, args, phase, is_training_set=True):
        """
        Args:
            args: Command line arguments.
            phase: one of 'train','val','test'
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTHeadDataset3d, self).__init__(args, phase, is_training_set=is_training_set)
        self.aneurysm_idxs = [i for i in range(len(self.series_list)) if self.series_list[i].is_aneurysm]
        self.min_aneurysm_slices = args.min_abnormal_slices
        self.num_slices = args.num_slices
        self.abnormal_prob = args.abnormal_prob if self.is_training_set else None
        self.use_hem = args.use_hem if self.is_training_set else None

        # Map from windows to series indices, and from series indices to windows
        self.window_to_series_idx = []  # Maps window indices to series indices
        self.series_to_window_idx = []  # Maps series indices to base window index for that series
        window_start = 0
        for i, s in enumerate(self.series_list):
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0)
            self.window_to_series_idx += num_windows * [i]
            self.series_to_window_idx.append(window_start)
            window_start += num_windows

        if self.use_hem:
            # Initialize a HardExampleMiner with IDs formatted like (series_idx, start_idx)
            example_ids = []
            for window_idx in range(len(self)):
                series_idx = self.window_to_series_idx[window_idx]
                series = self.series_list[series_idx]
                if not series.is_aneurysm:
                    # Only include negative examples in the HardExampleMiner
                    slice_offset = (window_idx - self.series_to_window_idx[series_idx]) * self.num_slices
                    start_idx = series.slice_num_to_idx(slice_offset + 1)
                    example_ids.append((series_idx, start_idx))
            self.hard_example_miner = util.HardExampleMiner(example_ids)

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
        if not self.is_test_mode and s.is_aneurysm and s.aneurysm_mask_path is None:
            return False
        if not s.is_aneurysm and not self.include_normals:
            return False

        return True

    def __len__(self):
        return len(self.window_to_series_idx)

    def __getitem__(self, idx):
        # Choose series and window within series
        series_idx = self.window_to_series_idx[idx]
        series = self.series_list[series_idx]

        if self.abnormal_prob is not None and random.random() < self.abnormal_prob:
            # Force aneurysm window with probability `abnormal_prob`.
            if not series.is_aneurysm:
                series_idx = random.choice(self.aneurysm_idxs)
                series = self.series_list[series_idx]
            start_idx = self._get_start_idx(series, force_aneurysm=True, randomize=self.is_training_set)
        elif self.use_hem:
            # Draw from distribution that weights hard negatives more heavily than easy examples
            series_idx, start_idx = self.hard_example_miner.sample()
            series = self.series_list[series_idx]
        else:
            # Get sequential windows through the whole series
            slice_offset = (idx - self.series_to_window_idx[series_idx]) * self.num_slices
            start_idx = series.slice_num_to_idx(slice_offset + 1)

        volume, mask = self._load_volume(series, start_idx)
        volume, mask = self._transform(volume, mask, series.brain_bbox, series.is_bottom_up)

        is_abnormal = torch.tensor([self._is_abnormal(mask)], dtype=torch.float32)

        # Pass series info to combine window-level predictions
        target = {'mask': mask,
                  'is_abnormal': is_abnormal,
                  'dset_path': series.dset_path,
                  'slice_idx': start_idx,
                  'series_idx': series_idx}
        # 'brain_bbox': np.array(series.brain_bbox)}

        return volume, target

    def get_series_label(self, series_idx):
        """Get a floating point label for a series at given index."""
        return float(self.series_list[series_idx].is_aneurysm)

    def update_hard_example_miner(self, example_ids, losses):
        """Update HardExampleMiner with set of example_ids and corresponding losses.

        This should be called at the end of every epoch.

        Args:
            example_ids: List of example IDs which were used during training.
            losses: List of losses for each example ID (must be parallel to example_ids).
        """
        example_ids = [(series_idx, start_idx) for series_idx, start_idx in example_ids
                       if series_idx not in self.aneurysm_idxs]
        if self.use_hem:
            self.hard_example_miner.update_distribution(example_ids, losses)

    def _get_start_idx(self, series, force_aneurysm, randomize=True):
        """Get a random start index for num_slices from a series.

        Args:
            series: Series to sample from.
            force_aneurysm: If true, force the range to contain the brain.
                Else sample the start index uniformly at random.
            randomize: If true, randomize the position of the aneurysm within the slices.
                Else put the aneurysm in the middle of the window.

        Returns:
            Randomly sampled start index into series.
        """
        lo_slice = 1
        hi_slice = len(series) - self.num_slices + 1

        # Get actual slice number
        if force_aneurysm and randomize:
            # Take a random window containing at least min_visible_slices slices of aneurysm(s)
            min_start = max(lo_slice, series.aneurysm_bounds[0] + self.min_aneurysm_slices - self.num_slices)
            max_start = max(min_start, min(hi_slice, series.aneurysm_bounds[1] - self.min_aneurysm_slices + 1))
            start_num = random.randint(min_start, max_start)
        elif force_aneurysm:
            # Center aneurysm(s) in the window
            aneurysm_center = (series.aneurysm_bounds[0] + series.aneurysm_bounds[1]) // 2
            start_num = max(1, aneurysm_center - self.num_slices // 2 + 1)
        else:
            if not randomize:
                raise RuntimeError('Cannot center on a non-existent aneurysm.')
            # Randomly sample num_slices from the study
            start_num = random.randint(lo_slice, hi_slice + 1)

        # Convert to slice index in volume
        start_idx = start_num - 1

        return start_idx

    def _load_volume(self, series, start_idx):
        """Load num_slices slices from a series, starting at start_idx. Also load the corresponding mask.

        Args:
            series: The series to load slices from.
            start_idx: Index of first slice to load.

        Returns:
            volume, mask: 3D NumPy arrays for the series volume and corresponding mask
            (mask will be None if task_type is not segmentation).
        """
        if self.img_format == 'png':
            raise NotImplementedError('No support for PNGs in our HDF5 files.')

        with h5py.File(os.path.join(self.data_dir, 'data.hdf5'), 'r') as hdf5_fh:
            volume = hdf5_fh[series.dset_path][start_idx:start_idx + self.num_slices]
            if series.is_aneurysm:
                mask = hdf5_fh[series.aneurysm_mask_path][start_idx:start_idx + self.num_slices]
                mask = mask.astype(np.float32)
            else:
                mask = np.zeros_like(volume, dtype=np.float32)

        return volume, mask

    def _is_abnormal(self, mask):
        """Check whether `mask` includes an abnormality.

        Args:
            mask: Window mask to check for any abnormal regions.

        Returns:
            True iff (1) mask contains an aneurysm and (2) mask is >20% of slices or not on the edge.
        """
        abnormal_range = util.get_range(mask, axis=0)

        if abnormal_range is None:
            # No aneurysm
            return False

        if ((abnormal_range[0] == 0 and abnormal_range[1] < self.min_aneurysm_slices - 1)
           or (abnormal_range[1] == self.num_slices - 1
               and abnormal_range[0] > self.num_slices - self.min_aneurysm_slices)):
            # Not enough overlap
            return False

        return True

    def _crop(self, volume, mask, x1, y1, x2, y2):
        """Crop a 3D volume and its corresponding mask (before channel dimension has been added)."""
        volume = volume[:, y1: y2, x1: x2]

        if mask is not None:
            mask = mask[:, y1: y2, x1: x2]

        return volume, mask

    def _rescale(self, volume, interpolation=cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)

    def _handle_bottom_up(self, volume, mask, is_bottom_up):
        """Flip a window if not bottom-up."""
        if not is_bottom_up:
            volume = np.flip(volume, axis=0).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()

        return volume, mask

    def _pad(self, volume, mask):
        """Pad a volume to make sure it has the expected number of slices.
        Pad the volume with slices of air, and pad its corresponding mask with zeros.

        Args:
            volume: 3D NumPy ndarray, where slices are along depth dimension (un-normalized raw HU).
            mask: Constant value to use for padding.

        Returns:
            volume, mask: Both padded/cropped to have the expected number of slices.
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
            if mask is not None:
                mask = add_padding(mask, pad_value=0)
        elif volume_num_slices > self.num_slices:
            # Choose center slices
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]
            if mask is not None:
                mask = mask[start_slice:start_slice + self.num_slices, :, :]

        return volume, mask

    def sample_aneurysm(self):
        """Sample an aneurysm series, and get a random window containing the aneurysm."""
        series = self.series_list[random.choice(self.aneurysm_idxs)]
        start_idx = self._get_start_idx(series, force_aneurysm=True)
        slices, _ = self._load_volume(series, start_idx)
        volume = self._transform(slices, None, series.brain_bbox, series.is_bottom_up)
        volume = volume.unsqueeze(0)  # Add batch dimension
        return volume
