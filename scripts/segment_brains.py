import numpy as np
import os
import pickle
import skimage.morphology as morph
import torch
import torch.nn.functional as F
import util

from args import TestArgParser
from data_loader import CTDataLoader
from logger import TestLogger
from saver import ModelSaver
from tqdm import tqdm


def segment_brains(args, series_list, unscaled_side_length=512):
    """Generate segmentation masks for brains."""
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    model = model.to(args.device)
    model.eval()

    # Get logger, data loader
    data_loader = CTDataLoader(args, phase=args.phase, is_training=False)
    logger = TestLogger(args, len(data_loader.dataset), data_loader.dataset.pixel_dict)

    # Get segmentation outputs
    dset2series = {s.dset_path: s for s in series_list}
    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (inputs, info_dict) in enumerate(data_loader):
            with torch.no_grad():
                logits = model.forward(inputs.to(args.device))
                probs = F.sigmoid(logits)

            logger.visualize(inputs, logits, targets_dict=None, phase=args.phase, unique_id=i)

            # Get bounding boxes for the brain
            preds = (probs.to('cpu').numpy() > args.prob_threshold)
            for dset_path, slice_idx, pred in zip(info_dict['dset_path'], info_dict['slice_idx'], list(preds)):
                series = dset2series[dset_path]
                if args.get_bbox:
                    # Keep only the largest connected component (should be the brain)
                    mask = np.squeeze(pred)
                    labels = morph.label(mask)
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    labels_counts = zip(unique_labels[1:], counts[1:])  # Forget background label 0
                    labels_counts = sorted(labels_counts, key=lambda x: x[1], reverse=True)
                    max_label = labels_counts[0][0]
                    mask = (labels == max_label)

                    brain_bbox = util.mask_to_bbox(mask)
                    if brain_bbox is not None:
                        series.brain_bbox = util.get_min_bbox(series.brain_bbox, brain_bbox)
                elif pred[0]:
                    # Convert predictions to range
                    if series.is_bottom_up:
                        first_slice = series.absolute_range[0] + int(slice_idx)
                    else:
                        first_slice = series.absolute_range[1] - int(slice_idx) - args.num_slices + 1
                    last_slice = first_slice + args.num_slices - 1
                    if series.brain_range is None:
                        series.brain_range = [first_slice, last_slice]
                    else:
                        series.brain_range[0] = min(series.brain_range[0], first_slice)
                        series.brain_range[1] = max(series.brain_range[1], last_slice)
                        print('Updated brain range: {}'.format(series.brain_range))

            progress_bar.update(inputs.size(0))

        # Scale bounding boxes back to input size
        rescale_factor = unscaled_side_length / args.resize_shape[0]
        for s in series_list:
            if s.brain_bbox is not None:
                s.brain_bbox = [int(rescale_factor * x) for x in s.brain_bbox]

    pkl_path = os.path.join(args.pkl_output_dir, 'series_list.pkl')
    util.print_err('Dumping pickle file to {}...'.format(pkl_path))
    with open(pkl_path, 'wb') as pkl_fh:
        pickle.dump(series_list, pkl_fh)


if __name__ == '__main__':
    parser = TestArgParser()
    parser.parser.add_argument('--prob_threshold', type=float, default=0.5,
                               help='Threshold probability for including a voxel in the brain mask.')
    parser.parser.add_argument('--pkl_output_dir', type=str, default='.',
                               help='Output path for updated series list.')
    parser.parser.add_argument('--min_mask_ccs', type=int, default=1e4,
                               help='Minimum mask connected component size (voxels) to keep in post-processing.')
    parser.parser.add_argument('--get_bbox', type=util.str_to_bool, default=False,
                               help='If true, get bounding boxes around the brain.')
    args_ = parser.parse_args()

    if args_.model not in ('UNet', 'VNet', 'R2Plus1D'):
        raise ValueError('Invalid model for segmenting brains: {}.'.format(args_.model))
    if not args_.ckpt_path:
        raise ValueError('Must specify a checkpoint for segmenting brains.')

    args_.phase = 'all'
    args_.only_topmost_window = args_.get_bbox

    with open(args_.pkl_path, 'rb') as pkl_fh_:
        series_list = pickle.load(pkl_fh_)

    segment_brains(args_, series_list)
