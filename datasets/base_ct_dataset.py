import numpy as np
import random

from PIL import Image
from torch.utils.data import Dataset


class BaseCTDataset(Dataset):
    """Base dataset for CT studies."""

    def __init__(self, data_dir, img_format, is_training_set=True):
        """
        Args:
            data_dir: Root data directory.
            img_format: Format of input image files. Options are 'raw' (Hounsfield Units) or 'png'.
            is_training_set: If training, shuffle pairs and define len as max of len(src_paths) and len(tgt_paths).
            If not training, take pairs in order and define len as len(src_paths).
            appear with the same set of tgt images.
        """
        if img_format != 'png' and img_format != 'raw':
            raise ValueError('Unsupported image format: {}'.format(img_format))

        self.data_dir = data_dir
        self.img_format = img_format
        self.is_training_set = is_training_set
        self.pixel_dict = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def _get_img(self, img_path):
        """Load an image from `img_path`. Use format `self.img_format`."""
        return np.load(img_path) if self.img_format == 'raw' else Image.open(img_path).convert('L')

    @staticmethod
    def _hide_and_seek_transform(np_block, to_show_set, mean, grid_dim_x, grid_dim_y, hide_prob, hide_level):
        """ Replaces grid squares in the image with the pixel mean with probability `hide_prob`
            , excepting those that are passed in through the 'to_show_set'

        Args:
            np_block: 2 or 3D npy block of pixel values (based on self.hide_level)
            to_show_set: set object containing the grid squares that should remain unhidden (contain the labeled feature)

        Returns:
            2 or 3D npy block of pixel values
        """
        # Blank out each square with P(hide_probability)
        grid_size = grid_dim_x * grid_dim_y
        to_hide_list = np.argwhere(np.array([random.random() for i in range(grid_size)]) < hide_prob)
        to_hide = np.array([])
        if len(to_hide_list) > 1:
            to_hide = np.concatenate(to_hide_list)

        # If there are no forced inclusions and we chose to hide everything in the image
        # Show at least one square
        if len(to_show_set) == 0 and len(to_hide_list) == grid_size:
            to_show_set = set([random.randint(0, grid_size - 1)])

        to_hide_set = set(to_hide.ravel())
        to_hide_set = to_hide_set - to_show_set
        
        if hide_level == 'image':
            image_dim = np_block.shape[0]
        else:
            image_dim = np_block.shape[1]
            
        for square in to_hide_set:
            row = square // grid_dim_x
            col = square % grid_dim_y
            width = image_dim // grid_dim_y
            remainder = width * grid_dim_y % image_dim
            
            pad_horizontal = remainder if row == grid_dim_x - 1 else 0
            pad_vertical = remainder if col == grid_dim_y - 1 else 0 
            if hide_level == 'image':
                np_block[row*width:(row+1)*width + pad_horizontal, col*width:(col+1)*width + pad_vertical] = mean 
            else:
                for idx in range(np_block.shape[0]):
                    np_block[idx, row*width:(row+1)*width + pad_horizontal, col*width:(col+1)*width + pad_vertical] = mean 
        
        return np_block

    def _normalize_raw(self, pixels):
        """Normalize an ndarray of raw Hounsfield Units to [-1, 1].

        Clips the values to [min, max] and scales into [0, 1],
        then subtracts the mean pixel (min, max, mean are defined in constants.py).

        Args:
            pixels: NumPy ndarray to convert. Any shape.

        Returns:
            NumPy ndarray with normalized pixels in [-1, 1]. Same shape as input.
        """
            
        #print(pixels.min(), pixels.max())
        # TODO normalize to range 
        #pixels = np.interp(pixels, (pixels.min(), pixels.max()), (-3024, 3071))

        pixels = pixels.astype(np.float32)
        pixels = (pixels - self.pixel_dict['min_val']) / (self.pixel_dict['max_val'] - self.pixel_dict['min_val'])
        pixels = np.clip(pixels, 0., 1.) - self.pixel_dict['avg_val']

        return pixels
