import cv2
from PIL import Image
import numpy as np
import util
import torch
from io import BytesIO
from ct.ct_pe_constants import (
    W_CENTER_DEFAULT,
    W_WIDTH_DEFAULT,
    CONTRAST_HU_MEAN,
    CONTRAST_HU_MIN,
    CONTRAST_HU_MAX,
)


def resize_array(np_array, slice_shape=(208, 208), interpolation=cv2.INTER_AREA):
    return cv2.resize(np_array, slice_shape, interpolation=interpolation)


def preprocess(x_arrays):
    x_arrays = sorted(x_arrays, key=lambda dcm: int(dcm.ImagePositionPatient[-1]))
    x_arrays = np.array([resize_array(dcm.pixel_array) for dcm in x_arrays])
    # rescale
    interpolation = cv2.INTER_AREA
    x_arrays = util.resize_slice_wise(x_arrays, (208, 208), interpolation)

    # crop
    row = (x_arrays.shape[-2] - 192) // 2
    col = (x_arrays.shape[-1] - 192) // 2
    x_arrays = x_arrays[:, row : row + 192, col : col + 192]
    return x_arrays


def get_windows(x_stacked, input_slice_number):
    num_slices = x_stacked.shape[0]
    num_windows = num_slices - input_slice_number + 1

    x_stacked = util.normalize(x_stacked)
    for i in range(num_windows):
        img_split = np.array(x_stacked[i : i + input_slice_number])
        img_expand = np.expand_dims(np.expand_dims(img_split, axis=0), axis=0)
        yield torch.from_numpy(np.array(img_expand))


def get_best_window(x_stacked, input_slice_number, best_window):

    x_unnorm_best = np.array(x_stacked[best_window : best_window + input_slice_number])
    # noramlize Hounsfield Units
    x_stacked = util.normalize(x_stacked)
    x_best = np.array(x_stacked[best_window : best_window + input_slice_number])
    x_unnorm_best = np.expand_dims(np.expand_dims(x_unnorm_best, axis=0), axis=0)
    x_best = np.expand_dims(np.expand_dims(x_best, axis=0), axis=0)
    return torch.from_numpy(np.array(x_best)), x_unnorm_best


def compute_gradcam_gif(cam, x, x_un_normalized):
    gradcam_output_buffer = BytesIO()

    new_cam = util.resize(cam, x[0])
    input_np = np.transpose(x_un_normalized[0], (1, 2, 3, 0))
    input_normed = np.float32(input_np) / 255

    cam_frames = list(util.add_heat_map(input_normed, new_cam))
    cam_frames = [Image.fromarray(frame) for frame in cam_frames]
    cam_frames[0].save(
        gradcam_output_buffer,
        save_all=True,
        append_images=cam_frames[1:] if len(cam_frames) > 1 else [],
        format="GIF",
        loop=0,
        optimize=True,
    )

    return gradcam_output_buffer
