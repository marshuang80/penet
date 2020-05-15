import cv2
from PIL import Image
import numpy as np
import util
import torch
from io import BytesIO
from ct.ct_pe_constants import W_CENTER_DEFAULT, W_WIDTH_DEFAULT


def format_img(img, input_slice_number, normalize_images=False):
    num_slices = img.shape[0]
    num_windows = num_slices - input_slice_number + 1

    # crop
    row = (img.shape[-2] - 192) // 2
    col = (img.shape[-1] - 192) // 2
    img = img[:, row : row + 192, col : col + 192]
    img = img.astype(np.float32)

    # noramlize Hounsfield Units
    if normalize_images:
        img = util.normalize(img)

    # expand dimention for tensor
    img_split = np.array([img[i : i + input_slice_number] for i in range(num_windows)])
    img_expand = [
        np.expand_dims(np.expand_dims(split, axis=0), axis=0) for split in img_split
    ]
    return img_expand


def resize_array(np_array, slice_shape=(208, 208), interpolation=cv2.INTER_AREA):
    return cv2.resize(np_array, slice_shape, interpolation=interpolation)


def preprocess(x_arrays, input_slice_number):
    x_arrays = sorted(x_arrays, key=lambda dcm: int(dcm.ImagePositionPatient[-1]))
    x_arrays = [resize_array(dcm.pixel_array) for dcm in x_arrays]

    x_stacked = np.stack(x_arrays, 0)
    x_stacked = util.apply_window(x_stacked, W_CENTER_DEFAULT, W_WIDTH_DEFAULT)

    x = format_img(x_stacked, input_slice_number, normalize_images=True)
    x = [torch.from_numpy(np.array(window)) for window in x]

    x_un_normalized = format_img(x_stacked, input_slice_number, normalize_images=False)

    return x, x_un_normalized


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
    )

    return gradcam_output_buffer
