import cv2
import h5py
import json
import math
import numpy as np
import os
import pandas as pd
import re
import sklearn.metrics as sk_metrics
import torch
import util

from args import TestArgParser
from ct import MAX_HEAD_HEIGHT_MM
from data_loader import CTDataLoader
from collections import defaultdict
from logger import TestLogger
from PIL import Image
from saver import ModelSaver
from time import time
from tqdm import tqdm


def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    data_loader = CTDataLoader(args, phase=args.phase, is_training=False)
    series2slices = defaultdict(list)
    series2slicelabels = defaultdict(list)
    series2probs = defaultdict(list)
    series2start = defaultdict(list)
    series2end = defaultdict(list)
    series2labels = {}
    logger = TestLogger(args, len(data_loader.dataset), data_loader.dataset.pixel_dict)

    window_shape = (args.num_slices, 512, 512)
    resize_shape = (args.num_slices,) + tuple(args.resize_shape)

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(args.results_dir))
    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (inputs, targets_dict) in enumerate(data_loader):
            with torch.no_grad():
                cls_logits, seg_logits = model.forward(inputs.to(args.device))
                cls_probs = torch.sigmoid(cls_logits)
                seg_probs = torch.sigmoid(seg_logits)

            if args.visualize_all:
                logger.visualize(inputs, cls_logits, seg_logits, targets_dict=None, phase=args.phase, unique_id=i)

            # TODO: Choose threshold and set masks to zero if cls_probs < threshold
            masks = seg_probs.to('cpu').numpy()
            max_probs = cls_probs.to('cpu').numpy()
            for dset_path, slice_idx, brain_bbox, mask, is_abnormal, prob \
                    in zip(targets_dict['dset_path'], targets_dict['slice_idx'], targets_dict['brain_bbox'],
                           list(masks), targets_dict['is_abnormal'], list(max_probs)):
                # Keep track of time
                series2start[dset_path].append(time())

                # Convert to standard Python data types
                brain_bbox = [x.item() for x in brain_bbox]
                slice_idx = int(slice_idx)

                # Save series num for aggregation
                series_num = int(re.match(r'/series/(\d+)', dset_path).group(1))
                series2slices[series_num].append(slice_idx)
                series2slicelabels[series_num].append(is_abnormal.item())
                series2probs[series_num].append(prob.item())

                series = data_loader.get_series(dset_path)
                if series_num not in series2labels:
                    series2labels[series_num] = int(series.is_aneurysm)
                if args.save_segmentation:
                    if prob > args.cls_threshold:
                        # Place mask into its position within the original volume
                        if not series.is_bottom_up:
                            mask = np.flip(mask, axis=0)

                        # Reverse crop: Pad out to resize shape
                        y1 = max(0, args.resize_shape[-2] - args.crop_shape[-2]) // 2
                        x1 = max(0, args.resize_shape[-1] - args.crop_shape[-1]) // 2
                        mask = util.pad_to_shape(mask, resize_shape, offsets=[y1, x1])

                        # Reverse resize: Resize out to brain bbox shape
                        x1, y1, side_length = util.get_crop(brain_bbox)
                        mask = util.resize_slice_wise(mask, (side_length, side_length),
                                                      interpolation_method=cv2.INTER_LINEAR)

                        # Reverse crop: Pad out to original shape
                        mask = util.pad_to_shape(mask, window_shape, offsets=[y1, x1])
                    else:
                        mask = np.zeros(window_shape, dtype=np.float32)

                    # Write predicted mask to disk
                    output_name = '{:05d}_{}.npy'.format(series_num, slice_idx)
                    output_path = os.path.join(args.results_dir, output_name)
                    np.save(output_path, mask)

                series2end[dset_path].append(time())

            progress_bar.update(inputs.size(0))

    # Save CSV of outputs
    df_rows = []
    for series_num in series2slices:
        for slice_idx, prob, label \
                in zip(series2slices[series_num], series2probs[series_num], series2slicelabels[series_num]):
            df_rows.append({'series_num': series_num,
                            'slice_idx': slice_idx,
                            'prob': prob,
                            'label': int(label)})
    df = pd.DataFrame(df_rows)
    columns = ['series_num', 'slice_idx', 'prob', 'label']
    df[columns].to_csv(os.path.join(args.results_dir, 'slice_wise.csv'), index=False)

    # Combine masks
    util.print_err('Combining masks...')
    max_probs = []
    labels = []
    for series_num in tqdm(series2slices):

        # Sort by slice index and get max probability
        slice_list, prob_list = (list(t) for t in zip(*sorted(zip(series2slices[series_num], series2probs[series_num]),
                                                              key=lambda slice_and_prob: slice_and_prob[0])))
        series2slices[series_num] = slice_list
        series2probs[series_num] = prob_list
        max_prob = max(prob_list)
        max_probs.append(max_prob)
        label = series2labels[series_num]
        labels.append(label)
        series = data_loader.get_series('/series/{:05d}'.format(series_num))

        if args.save_segmentation:
            # Create mask of 0s with same size as volume
            dset_path = '/series/{:05d}'.format(series_num)
            with h5py.File(os.path.join(args.data_dir, 'data.hdf5')) as hdf5_fh:
                volume_shape = hdf5_fh[dset_path].shape
            mask_all = np.zeros(volume_shape, dtype=np.float32)

            for slice_idx in series2slices[series_num]:
                # Fill the mask window-by-window
                mask_path = os.path.join(args.results_dir, '{:05d}_{}.npy'.format(series_num, slice_idx))
                mask_window = np.load(mask_path)
                num_slices = min(args.num_slices, volume_shape[0] - slice_idx)
                mask_all[slice_idx: slice_idx + num_slices, :, :] = mask_window[:num_slices, :, :]
                os.unlink(mask_path)

            # Flip if top-down, and write to disk
            if not series.is_bottom_up:
                mask_all = np.flipud(mask_all)

            if mask_all.shape != (len(series), 512, 512):
                util.print_err('Expected shape {}, got {}'.format((len(series), 512, 512), mask_all.shape))

            output_path = os.path.join(args.results_dir, '{:05d}_prob_{:.4f}.npy'.format(series_num, max_prob))
            np.save(output_path, mask_all)

    if args.outputs_for_xgb:
        # Write features for XGBoost
        save_for_xgb(args.results_dir, data_loader, series2probs, series2labels, series2slices,
                     args.num_slices, args.eval_stride, args.do_truncate)
        # Write the slice indices used for the features
        with open(os.path.join(args.results_dir, 'xgb', 'series2slices.json'), 'w') as json_fh:
            json.dump(series2slices, json_fh, sort_keys=True, indent=4)

    # Compute times, write to files
    df_rows = []
    for dset_path in series2start:
        start, end = min(series2start[dset_path]), max(series2end[dset_path])
        time_taken = end - start
        df_rows.append({'series': dset_path, 'time_in_seconds': time_taken})
    time_df = pd.DataFrame(df_rows)
    time_df.to_csv(os.path.join(args.results_dir, 'times.csv'))

    # Compute AUROC and AUPRC using max aggregation, write to files
    max_probs, labels = np.array(max_probs), np.array(labels)
    np.save(os.path.join(args.results_dir, 'predictions.npy'), np.array(max_probs))
    np.save(os.path.join(args.results_dir, 'labels.npy'), np.array(labels))
    metrics = {
        args.phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, max_probs),
        args.phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, max_probs),
    }
    with open(os.path.join(args.results_dir, 'metrics.txt'), 'w') as metrics_fh:
        for k, v in metrics.items():
            metrics_fh.write('{}: {:.5f}\n'.format(k, v))

    curves = {
        args.phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, max_probs),
        args.phase + '_' + 'ROC': sk_metrics.roc_curve(labels, max_probs)
    }
    for name, curve in curves.items():
        curve_np = util.get_plot(name, curve)
        curve_img = Image.fromarray(curve_np)
        curve_img.save(os.path.join(args.results_dir, '{}.png'.format(name)))


def save_for_xgb(results_dir, data_loader, series2probs, series2labels, series2slices,
                 num_slices, eval_stride, do_truncate):
    """Write window-level and series-level features to train an XGBoost classifier.

    Args:
        results_dir: Path to results directory for writing outputs.
        data_loader: DataLoader for mapping series numbers to actual series.
        series2probs: Dict mapping series numbers to probabilities.
        series2labels: Dict mapping series numbers to labels.
        series2slices: Dict mapping series numbers to slice indices.
        num_slices: Number of slices in each window.
        eval_stride: Number of slices between each window.
        do_truncate: If true, truncate to just the head.
    """
    for series_num in series2probs:
        series = data_loader.get_series('/series/{:05d}'.format(series_num))
        if do_truncate:
            # Determine how many windows correspond to head region (assuming 99th percentile tall head)
            window_thickness = num_slices * series.spacing_between_slices
            stride_thickness = eval_stride * series.spacing_between_slices
            num_windows = 1 + math.ceil(max(0, MAX_HEAD_HEIGHT_MM - window_thickness) / stride_thickness)
        else:
            num_windows = len(series2probs[series_num])

        # Take topmost num_windows windows and write them top-down (so padding goes at the end for short scans)
        if series.is_bottom_up:
            series2probs[series_num] = [prob for prob in reversed(series2probs[series_num][-num_windows:])]
            series2slices[series_num] = [slice_idx for slice_idx in reversed(series2slices[series_num][-num_windows:])]
        else:
            series2probs[series_num] = [prob for prob in series2probs[series_num][:num_windows]]
            series2slices[series_num] = [slice_idx for slice_idx in series2slices[series_num][:num_windows]]

    # Build features from list of probs
    series2features = get_xgb_features(series2probs)

    # Convert to numpy
    xgb_inputs = np.zeros([len(series2features), max(len(ft) for ft in series2features.values())])
    xgb_labels = np.zeros(len(series2labels))
    for i, (series_num, features) in enumerate(series2features.items()):
        xgb_inputs[i, :len(features)] = np.array(features).ravel()
        xgb_labels[i] = series2labels[series_num]

    # Write to disk
    os.makedirs(os.path.join(results_dir, 'xgb'), exist_ok=True)
    xgb_inputs_path = os.path.join(results_dir, 'xgb', 'inputs.npy')
    xgb_labels_path = os.path.join(results_dir, 'xgb', 'labels.npy')
    np.save(xgb_inputs_path, xgb_inputs)
    np.save(xgb_labels_path, xgb_labels)


def get_xgb_features(series2probs, num_buckets=10):
    """Get features for XGBoost from a list of probabilities for each series.

    Args:
        series2probs: Dict mapping series numbers to list of top-down probabilities.
        num_buckets: Number of buckets to split each list of probabilities into.

    Returns:
        Dict mapping series numbers to list of features for that series.
    """
    # series2features = {}
    # for series_num, probs in series2probs.items():
    #     features = [-1] * (num_buckets * 3)  # Max, mean, min for each bucket
    #     buckets = np.array_split(probs, num_buckets)
    #     i = 0
    #     for bucket in buckets:
    #         if len(bucket) == 0:
    #             break
    #         features[i] = np.max(bucket).item()
    #         features[i + 1] = np.mean(bucket).item()
    #         features[i + 2] = np.min(bucket).item()
    #         i += 3
    #     series2features[series_num] = features
    #
    # return series2features
    return series2probs


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    args_ = parser.parse_args()
    test(args_)
