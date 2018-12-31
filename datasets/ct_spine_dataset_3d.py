import cv2
import numpy as np
import os
import pickle
import random
import torch
import torchvision.transforms as transforms

from .base_ct_dataset import BaseCTDataset
from ct.ct_spine_constants import *
from PIL import Image


class CTSpineDataset3d(BaseCTDataset):
    def __init__(self, args, phase, is_training_set):
        """
        Args:
            args: Command line arguments. See args/ folder for details
            phase: One of 'train', 'val', 'test' to load
            is_training_set: If true, load dataset for training. Otherwise, load for test inference.
        """
        super(CTSpineDataset3d, self).__init__(args.data_dir, args.img_format, is_training_set)
        self.resize_shape = args.resize_shape
        self.num_slices = args.num_slices
        self.vstep_size = args.vstep_size
        self.series = args.series
        self.hide_probability = args.hide_probability
        self.img_format = args.img_format
        self.is_training = is_training
        self.hide_level = args.hide_level
        self.pixel_dict = {}
        if self.img_format == 'raw':
            self.pixel_dict['avg_val'] = HU_AXIAL_MEAN if self.series == 'axial' else HU_SAGITTAL_MEAN
        else:
            self.pixel_dict['avg_val'] = PNG_AXIAL_MEAN if self.series == 'axial' else PNG_SAGITTAL_MEAN

        self.pixel_dict['min_val'] = HU_MIN
        self.pixel_dict['max_val'] = HU_MAX
        self.pixel_dict['w_center'] = W_CENTER_DEFAULT
        self.pixel_dict['w_width'] = W_WIDTH_DEFAULT
        
        self.img_paths, self.labels, self.windows = self._load_data(phase)
        
        # maintain two separate lists of fracture v. no fracture for ease
        # of random sampling in training -- window[4] is the label for the window
        self.fractures = [window for window in self.windows if window[4] == 1]
        self.no_fractures = [window for window in self.windows if window[4] == 0]
        
        if args.toy:
            fracture_mask = np.array(random.sample(range(len(self.fractures)), args.toy_size // 2))
            self.fractures = [window for i, window in enumerate(self.fractures) if i in fracture_mask]
            
            fracture_mask = np.array(random.sample(range(len(self.no_fractures)), args.toy_size // 2))
            self.no_fractures = [window for i, window in enumerate(self.no_fractures) if i in no_fracture_mask]
            self.windows = self.fractures + self.no_fractures
    
    def __len__(self):
        return len(self.windows) if not self.is_training else 5000 
    
    def __getitem__(self, index):
        """ Returns window -> label (window contains a slice with fracture) """ 
        if self.is_training_set:
            choose_fracture = random.random() < .5
            num_idxs = len(self.fractures) if choose_fracture else len(self.no_fractures)
            index = random.randint(0,num_idxs - 1)
            start_slice, end_slice, series_path, series_idx, label, fracture_squares = self.fractures[index] \
                if choose_fracture else self.no_fractures[index]
                
            window = np.load(str(self.img_paths[series_idx]), mmap_mode='r')[start_slice:end_slice]      
            label = torch.tensor([label], dtype=torch.float32)
            return self._transform_fn(window, fracture_squares), label
        else:
            start_slice, end_slice, series_path, series_idx, label, fracture_squares = self.windows[index]
            label = torch.tensor([label], dtype=torch.float32)
            window = np.load(self.img_paths[series_idx])[start_slice:end_slice]
            series_idx = torch.tensor([series_idx], dtype=torch.float32)
            return self._transform_fn(window, fracture_squares), series_idx, label
    
    def get_series_label(self, series_idx):
        """ Returns the label associated with `series_idx` as a float. """
        return float(self.labels[int(series_idx)])

    def _load_data(self, phase):
        """Loads the data from {self.series}_{phase} pickle files.
        
        Args:
            phase: One of 'train', 'val', 'test'.
            
        Returns: 
            A list of the img_paths to each series, a list of the labels for
            each series, and a list `windows` containing windows for each 
            series organized as follows:
            
            (start_slice, end_slice, series_path, series_idx, label for window)
        """
        with open(os.path.join(self.data_dir, '{}_{}_img_paths_{}.pkl'.format(self.series, phase, self.num_slices)), 'rb') as f:
            img_paths = pickle.load(f)
        with open(os.path.join(self.data_dir, '{}_{}_labels_{}.pkl'.format(self.series, phase, self.num_slices)), 'rb') as f:
            labels = pickle.load(f)
        with open(os.path.join(self.data_dir, '{}_{}_windows_{}.pkl'.format(self.series, phase, self.num_slices)), 'rb') as f:
            windows = pickle.load(f)  


        assert len(img_paths) == len(labels), \
               "Different number of examples in inputs and labels"
        
        return img_paths, labels, windows
    
    def _transform_fn(self, window, fracture_squares):
        """ Resizes, normalizes, and converts a window of a series to Torch Tensor.
        
        Args:
            series: The series to extract slices from.
            
        Returns:
            Torch Tensor, shape (num_channels, num_slices, height, width).
        """
        H, W = self.resize_shape
        
        N = len(window)
        window = window.astype(np.float32)
        resized_window = np.ones((self.num_slices, H, W)) * self.pixel_dict['avg_val']
        
        for i in range(N):
            if self.img_format == 'png':
                transforms_list = [transforms.Resize((H, W)), transforms.ToTensor(), transforms.Lambda(lambda x: 255 * x), transforms.Normalize([self.pixel_dict['avg_val']], [1])]
                transform = transforms.Compose([t for t in transforms_list if t])
                resized_window[i] = transform(Image.fromarray(window[i]))
            else:
                window[i] = self._normalize_raw(window[i]) 
                resized_window[i] = cv2.resize(np.squeeze(window[i]), tuple(self.resize_shape), interpolation=cv2.INTER_AREA)
            
            if self.hide_probability != 0 and self.hide_level == 'image' and self.is_training:
                resized_window[i] = self._hide_and_seek_transform(resized_window[i], fracture_squares,
                                                                  self.pixel_dict['avg_val'],
                                                                  ANNOTATION_GRID_DIM_X, ANNOTATION_GRID_DIM_Y,
                                                                  self.hide_probability, self.hide_level)
        
        if self.hide_probability != 0 and self.hide_level == 'window' and self.is_training:
            resized_window = self._hide_and_seek_transform(resized_window, fracture_squares, self.pixel_dict['avg_val'],
                                                           ANNOTATION_GRID_DIM_X, ANNOTATION_GRID_DIM_Y,
                                                           self.hide_probability, self.hide_level)

        img_vol = torch.from_numpy(resized_window).float()
        img_vol = torch.unsqueeze(img_vol, 0)
        return img_vol
