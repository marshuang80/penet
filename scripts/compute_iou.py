"""Compute mean IoU between output masks and ground-truth masks."""
import argparse
import h5py
import numpy as np
import os
import util

from tqdm import tqdm


def main(args):
    hdf5_fh = h5py.File(args.hdf5_path, 'r')

    ious = []
    mask_names = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    for mask_name in tqdm(mask_names):
        mask_path = os.path.join(args.input_dir, mask_name)
        hdf5_path = '/aneurysm_masks/{}'.format(mask_name[:5])

        mask_pred = np.load(mask_path) > args.prob_threshold
        mask_pred = mask_pred.astype(np.uint8)
        if hdf5_path in hdf5_fh:
            mask_gold = hdf5_fh[hdf5_path][...]
            mask_gold = mask_gold.astype(np.uint8)

            iou = get_iou(mask_pred, mask_gold)
            ious.append(iou)

    util.print_err('IOUs: {}'.format(ious))
    util.print_err('Mean IOU: {}'.format(np.mean(ious)))

    hdf5_fh.close()


def get_iou(v, w=None):
    """Get intersection over union between two masks

    Args:
        v: First volume.
        w: Second volume.

    Returns:
        Intersection-over-union of two masks.
    """
    assert v.dtype == np.uint8 and np.max(v) <= 1, "Mask v is not a binary mask of uint8's"
    assert w is None or (w.dtype == np.uint8 and np.max(w) <= 1), "Mask w is not a binary mask of uint8's"

    if w is None:
        intersection = 0.
        union = np.sum(v)
    else:
        v = v[:w.shape[0]]
        intersection = np.sum(v * w)
        union = np.sum(v) + np.sum(w) - intersection

    assert union > 0, "IoU written for positive examples only."

    return intersection / union


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hdf5_path', default='/data/head-ct/data.hdf5')
    parser.add_argument('--input_dir', default='/data/head-ct/results/chute_xnet0810_seg_val3/')
    parser.add_argument('--prob_threshold', default=0.5)

    main(parser.parse_args())
