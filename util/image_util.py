import cv2
import matplotlib.pyplot as plt
import numpy as np
#import SimpleITK as sitk
import scipy.ndimage.interpolation as interpolation

from scipy import interpolate
#import sklearn.metrics as metrics, average_precision_score


def apply_window(img, w_center, w_width):
    """Window a NumPy array of raw Hounsfield Units.

    Args:
        img: Image to apply the window to. NumPy array of any shape.
        w_center: Center of window.
        w_width: Width of window.

    Returns:
        img_np: NumPy array of after windowing. Values in range [y_min, y_max].
    """
    # Convert to float
    img = np.copy(img).astype(np.float64)

    # Clip to min and max values
    w_max = w_center + w_width / 2
    w_min = w_center - w_width / 2
    img = np.clip(img, w_min, w_max)

    # Normalize to uint8
    img -= w_min
    img /= w_width
    img *= np.iinfo(np.uint8).max
    img = img.astype(np.uint8)

    return img


def get_crop(bbox):
    """Get crop coordinates and side length given a bounding box.
    Force the crop to be square.

    Args:
        bbox: x1, y1, x2, y2; coordinates for bounding box.

    Returns:
        x1, y1, side_length:
    """
    # Get side length to keep square aspect ratio
    x1, y1, x2, y2 = bbox
    side_length = max(x2 - x1, y2 - y1) + 1

    # Center the skull in the cropped region
    x1 = max(0, x1 - (side_length - (x2 - x1 + 1)) // 2)
    y1 = max(0, y1 - (side_length - (y2 - y1 + 1)) // 2)

    return x1, y1, side_length


def pad_to_shape(array, output_shape, offsets, dtype=np.float32):
    """Pad an array with zeros to the desired output shape.
    Args:
        array: Array to be padded.
        output_shape: The desired shape for the output.
        offsets: List of offsets (will be prepended with zeros
            if fewer dimensions than output_shape).
        dtype: Data type for output array.
    Returns:
        array padded to the given `output_shape`.
    """

    # Create a list of slices from offset to offset + shape in each dimension
    if len(offsets) < len(output_shape):
        offsets = [0] * (len(output_shape) - len(offsets)) + offsets

    side_lengths = [min(output_shape[dim], offsets[dim] + array.shape[dim]) - offsets[dim]
                    for dim in range(array.ndim)]
    tgt_idx = [slice(offsets[dim], offsets[dim] + side_lengths[dim])
               for dim in range(array.ndim)]
    src_idx = [slice(0, side_lengths[dim])
               for dim in range(array.ndim)]

    # Create an array of zeros, may be larger than output shape
    result = np.zeros(output_shape, dtype=dtype)

    # Insert the array in the result at the specified offsets
    result[tgt_idx] = array[src_idx]

    # Trim down to output_shape
    result = result[[range(0, d) for d in output_shape]]

    return result


def un_normalize(tensor, img_format, pixel_dict):
    """Un-normalize a PyTorch Tensor seen by the model into a NumPy array of
    pixels fit for visualization. If using raw Hounsfield Units, window the input.

    Args:
        tensor: Tensor with pixel values in range (-1, 1).
            If video, shape (batch_size, num_channels, num_frames, height, width).
            If image, shape (batch_size, num_channels, height, width).
        img_format: Input image format to the network. Options are 'raw' or 'png'.
        pixel_dict: Dictionary containing min, max, avg of pixel data; window center, width.

    Returns:
        pixels_np: Numpy ndarray with entries of type np.uint8.
    """
    pixels_np = tensor.cpu().float().numpy()

    # Reverse pre-processing steps for visualization
    if img_format == 'png':
        pixels_np = 0.5 * (pixels_np + 1.) * 255.
    else:
        pixels_np = (pixels_np + pixel_dict['avg_val']) * (pixel_dict['max_val'] - pixel_dict['min_val']) \
                    + pixel_dict['min_val']
        pixels_np = pixels_np.astype(np.int16)  # 16-bit int for Hounsfield Units
        pixels_np = apply_window(pixels_np, pixel_dict['w_center'], pixel_dict['w_width'])
        
    return pixels_np


def mask_to_bbox(mask):
    """Convert a mask to bounding box coordinates.

    Args:
        mask: NumPy ndarray of any type, where 0 or false is treated as background.

    Returns:
        x1, y1, x2, y2: Coordinates of corresponding bounding box.
    """
    is_3d = len(mask.shape) == 3

    reduce_axes = (0, 1) if is_3d else (0,)
    cols_any = np.any(mask, axis=reduce_axes)
    cols_where = np.where(cols_any)[0]
    if cols_where.shape[0] == 0:
        return None
    x1, x2 = cols_where[[0, -1]]

    reduce_axes = (0, 2) if is_3d else (1,)
    rows_any = np.any(mask, axis=reduce_axes)
    rows_where = np.where(rows_any)[0]
    if rows_where.shape[0] == 0:
        return None
    y1, y2 = rows_where[[0, -1]]

    return x1, y1, x2, y2


def get_range(mask, axis=0):
    """Get the range of the foreground label along an axis.
    Args:
        mask: NumPy with mask values, where 0 is treated as background.
        axis: The axis to get the min/max range for.
    Returns:
        z_range: List with two elements, the min and max z-axis indices containing foreground.
    """
    if len(mask.shape) != 3:
        raise ValueError('Unexpected shape in get_z_range: Needs to be a 3D tensor.')

    reduction_axes = [0, 1, 2]
    reduction_axes.pop(axis)

    axis_any = np.any(mask, axis=tuple(reduction_axes))
    axis_where = np.where(axis_any)[0]
    if axis_where.shape[0] == 0:
        return None
    axis_min, axis_max = axis_where[[0, -1]]

    return [axis_min, axis_max]


def resize_slice_wise(volume, slice_shape, interpolation_method=cv2.INTER_AREA):
    """Resize a volume slice-by-slice.

    Args:
        volume: Volume to resize.
        slice_shape: Shape for a single slice.
        interpolation_method: Interpolation method to pass to `cv2.resize`.

    Returns:
        Volume after reshaping every slice.
    """
    slices = list(volume)
    for i in range(len(slices)):
        slices[i] = cv2.resize(slices[i], slice_shape, interpolation=interpolation_method)
    return np.array(slices)


def _make_rgb(image):
    """Tile a NumPy array to make sure it has 3 channels."""
    if image.shape[-1] != 3:
        tiling_shape = [1] * (len(image.shape) - 1) + [3]
        return np.tile(image, tiling_shape)
    else:
        return image


def concat_images(images, spacing=10):
    """Concatenate a list of images to form a single row image.

    Args:
        images: Iterable of numpy arrays, each holding an image.
        Must have same height, num_channels, and have dtype np.uint8.
        spacing: Number of pixels between each image.

    Returns: Numpy array. Result of concatenating the images in images into a single row.
    """
    images = [_make_rgb(image) for image in images]
    # Make array of all white pixels with enough space for all concatenated images
    assert spacing >= 0, 'Invalid argument: spacing {} is not non-negative'.format(spacing)
    assert len(images) > 0, 'Invalid argument: images must be non-empty'
    num_rows, _, num_channels = images[0].shape
    assert all([img.shape[0] == num_rows and img.shape[2] == num_channels for img in images]),\
        'Invalid image shapes: images must have same num_channels and height'
    num_cols = sum([img.shape[1] for img in images]) + spacing * (len(images) - 1)
    concatenated_images = np.full((num_rows, num_cols, num_channels), fill_value=255, dtype=np.uint8)

    # Paste each image into position
    col = 0
    for img in images:
        num_cols = img.shape[1]
        concatenated_images[:, col:col + num_cols, :] = img
        col += num_cols + spacing

    return concatenated_images


def stack_videos(img_list):
    """Stacks a sequence of image numpy arrays of shape (num_images x w x h x c) to display side-by-side."""
    # If not RGB, stack to make num_channels consistent
    img_list = [_make_rgb(img) for img in img_list]
    stacked_array = np.concatenate(img_list, axis=2)
    return stacked_array


def add_heat_map(pixels_np, intensities_np, alpha_img=0.33, color_map='magma', normalize=True):
    """Add a CAM heat map as an overlay on a PNG image.

    Args:
        pixels_np: Pixels to add the heat map on top of. Must be in range (0, 1).
        intensities_np: Intensity values for the heat map. Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map. Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range exactly from 0 to 1.

    Returns:
        Original pixels with heat map overlaid.
    """
    assert(np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0)
    color_map_fn = plt.get_cmap(color_map)
    if normalize:
        intensities_np = normalize_to_image(intensities_np)
    else:
        intensities_np *= 255
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = alpha_img * pixels_np.astype(np.float32) + (1. - alpha_img) * heat_map.astype(np.float32)
    new_img = np.uint8(normalize_to_image(new_img))

    return new_img


def dcm_to_png(dcm, w_center=None, w_width=None):
    """Convert a DICOM object to a windowed PNG-format Numpy array.
    Add the given shift to each pixel, clip to the given window, then
    scale pixels to range implied by dtype (e.g., [0, 255] for `uint8`).
    Return ndarray of type `dtype`.

    Args:
        dcm: DICOM object.
        w_center: Window center for windowing conversion.
        w_width: Window width for windowing conversion.

    See Also:
        https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281050
    """
    pixels = dcm.pixel_array
    shift = dcm.RescaleIntercept
    if w_center is None:
        w_center = dcm.WindowCenter
    if w_width is None:
        w_width = dcm.WindowWidth

    img = np.copy(pixels).astype(np.float64) + shift
    img = apply_window(img, w_center, w_width)

    return img


def dcm_to_raw(dcm, dtype=np.int16):
    """Convert a DICOM object to a Numpy array of raw Hounsfield Units.

    Scale by the RescaleSlope, then add the RescaleIntercept (both DICOM header fields).

    Args:
        dcm: DICOM object.
        dtype: Type of elements in output array.

    Returns:
        ndarray of shape (height, width). Pixels are `int16` raw Hounsfield Units.

    See Also:
        https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    """
    img_np = dcm.pixel_array
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    img_np = img_np.astype(dtype)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    img_np[img_np == -2000] = 0

    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope

    if slope != 1:
        img_np = slope * img_np.astype(np.float64)
        img_np = img_np.astype(dtype)

    img_np += int(intercept)
    img_np = img_np.astype(np.int16)

    return img_np


def random_elastic_transform(volume, mask, default_pixel, num_control_points=2, std_dev=5):
    """Apply a random elastic deformation to a volume and corresponding mask.

    Adapted from:
        https://github.com/faustomilletari/VNet/blob/master/utilities.py

    Args:
        volume: Volume to deform.
        mask: Mask to deform in the same way.
        default_pixel: Default pixel value for filling.
        num_control_points: Number of control points to use throughout the volume.
        std_dev: Standard deviation of the location of the control points.

    Returns:
        Deformed target, deformed mask. NumPy ndarrays.
    """
    volume_sitk = sitk.GetImageFromArray(volume, isVector=False)
    mask_sitk = sitk.GetImageFromArray(mask, isVector=False)
    transform_mesh_size = [num_control_points] * volume_sitk.GetDimension()

    tx = sitk.BSplineTransformInitializer(volume_sitk, transform_mesh_size)

    params = tx.GetParameters()
    params_np = np.asarray(params, dtype=np.float32)
    params_np = params_np + np.random.randn(len(params)) * std_dev

    # No deformation along the z-axis
    params_np[int(len(params) / 3) * 2:] = 0

    params = tuple(params_np)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(default_pixel)
    deformed_volume_sitk = resampler.Execute(volume_sitk)
    resampler.SetDefaultPixelValue(0)
    deformed_mask_sitk = resampler.Execute(mask_sitk)

    deformed_volume = sitk.GetArrayFromImage(deformed_volume_sitk)
    deformed_volume = deformed_volume.astype(dtype=np.float32)

    deformed_mask = sitk.GetArrayFromImage(deformed_mask_sitk)
    deformed_mask = (deformed_mask > 0.5).astype(dtype=np.float32)

    return deformed_volume, deformed_mask


def random_translate(volume, mask):
    """Apply a random translation to a volume and corresponding mask.

    Adapted from:
        https://github.com/faustomilletari/VNet/blob/master/utilities.py

    Args:
        volume: Volume to translate.
        mask: Mask to translate in the same way.

    Returns:
        Translated volume, translated mask. Numpy ndarrays.
    """
    volume_sitk = sitk.GetImageFromArray(volume, isVector=False)
    mask_sitk = sitk.GetImageFromArray(mask, isVector=False)

    idx = np.where(mask > 0)
    translation = (0, np.random.randint(int(-np.min(idx[1]) / 2), (volume.shape[1] - np.max(idx[1])) / 2),
                   np.random.randint(int(-np.min(idx[0]) / 2), (volume.shape[0] - np.max(idx[0])) / 2))
    translation = sitk.TranslationTransform(3, translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(translation)

    translated_volume_sitk = resampler.Execute(volume_sitk)
    translated_label_sitk = resampler.Execute(mask_sitk)

    translated_volume = sitk.GetArrayFromImage(translated_volume_sitk)
    translated_volume = translated_volume.astype(dtype=float)

    translated_label = sitk.GetArrayFromImage(translated_label_sitk) > 0
    translated_label = translated_label.astype(dtype=float)

    return translated_volume, translated_label


def resample(volume, slice_thickness, pixel_spacing, output_scale=(1., 1., 1.)):
    """Resample a volume to a new scale.

    Args:
        volume: NumPy ndarray to resample.
        slice_thickness: Input slice thickness reported by dcm.SliceThickness.
        pixel_spacing: Input pixel spacing as reported by dcm.PixelSpacing.
        output_scale: Amount of 3D space occupied by a single voxel after interpolation (in mm).

    Adapted from:
        https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

    Returns:
        Interpolated volume, actual scale (in mm) after interpolation.
    """
    input_scale = np.array([slice_thickness] + list(pixel_spacing), dtype=np.float32)
    resize_factor = input_scale / output_scale

    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape
    real_scale = input_scale / real_resize_factor
    volume = interpolation.zoom(volume, real_resize_factor, mode='nearest')

    return volume, real_scale


def get_skull_bbox(img):
    """Get a minimal bounding box around the skull.

    Args:
        img: Numpy array of uint8's, after windowing.

    Returns:
        start_x, start_y, end_x, end_y: Coordinates of top-left, bottom-right corners
        for minimal bounding box around the skull.
    """
    _, thresh_img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    image, contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    skull_bbox = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(c)
        extent = area / float(w * h)
        if extent < 0.2:
            continue
        skull_bbox = get_min_bbox(skull_bbox, (x, y, x + w, y + h))

    return skull_bbox


def get_min_bbox(box_1, box_2):
    """Get the minimal bounding box around two boxes.

    Args:
        box_1: First box of coordinates (x1, y1, x2, y2). May be None.
        box_2: Second box of coordinates (x1, y1, x2, y2). May be None.
    """
    if box_1 is None:
        return box_2
    if box_2 is None:
        return box_1

    b1_x1, b1_y1, b1_x2, b1_y2 = box_1
    b2_x1, b2_y1, b2_x2, b2_y2 = box_2

    x1 = min(b1_x1, b2_x1)
    y1 = min(b1_y1, b2_y1)
    x2 = max(b1_x2, b2_x2)
    y2 = max(b1_y2, b2_y2)

    return x1, y1, x2, y2


def get_plot(title, curve):
    """Get a NumPy array for the given curve.
    Args:
        title: Name of curve.
        curve: NumPy array of x and y coordinates.
    Returns:
        NumPy array to be used as a PNG image.
    """
    fig = plt.figure()
    ax = plt.gca()

    plot_type = title.split('_')[-1]
    ax.set_title(plot_type)
    if plot_type == 'PRC':
        precision, recall, _ = curve
        ax.step(recall, precision, color='b', alpha=0.2, where='post')
        ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
    elif plot_type == 'ROC':
        false_positive_rate, true_positive_rate, _ = curve
        #roc_auc = metrics.auc(fpr, tpr)
        ax.plot(false_positive_rate, true_positive_rate, color='b')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc = 'lower right')
    else:
        ax.plot(curve[0], curve[1], color='b')

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

    fig.canvas.draw()

    curve_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    curve_img = curve_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return curve_img


def resize(cam, input_img, interpolation='linear'):
    """Resizes a volume using factorized bilinear interpolation"""
    temp_cam = np.zeros((cam.shape[0], input_img.size(2), input_img.size(3)))
    for dim in range(temp_cam.shape[0]):
        temp_cam[dim, :, :] = cv2.resize(cam[dim, :, :], dsize=(temp_cam.shape[1], temp_cam.shape[2]))

    if temp_cam.shape[0] == 1:
        new_cam = np.tile(temp_cam, (input_img.size(1), 1, 1))
    else:
        new_cam = np.zeros((input_img.size(1), temp_cam.shape[1], temp_cam.shape[2]))
        for i in range(temp_cam.shape[1] * temp_cam.shape[2]):
            y = i % temp_cam.shape[2]
            x = (i // temp_cam.shape[2])
            compressed = temp_cam[:, x, y]
            labels = np.arange(compressed.shape[0], step=1)
            new_labels = np.linspace(0, compressed.shape[0] - 1, new_cam.shape[0])
            f = interpolate.interp1d(labels, compressed, kind=interpolation)
            expanded = f(new_labels)
            new_cam[:, x, y] = expanded

    return new_cam


def normalize_to_image(img):
    """Normalizes img to be in the range 0-255."""
    img -= np.amin(img)
    img /= (np.amax(img) + 1e-7)
    img *= 255
    return img


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
