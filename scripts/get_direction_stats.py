import argparse
import json
import moviepy.editor as mpy
import numpy as np
import os
import pandas as pd
import util

from collections import defaultdict
from ct.ct_head_constants import *
from data_loader import CTDataLoader
from pathlib import Path

pixel_dict = {
            'min_val': CONTRAST_HU_MIN,
            'max_val': CONTRAST_HU_MAX,
            'avg_val': CONTRAST_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT
        }

load_args = {
    "batch_size": 1,
    "crop_shape": "224,224",
    "data_dir": "/home/chute/synth/data/seg",
    "elastic_transform": False,
    "flip": False,
    "img_format": "raw",
    "model": "VNet",
    "num_channels": 1,
    "num_slices": 32,
    "num_workers": 8,
    "resize_shape": "256,256",
    "rotate": False,
    "threshold_size": 0,
    "toy": False,
    "use_contrast": True,
    "phase": "train",
    "loader": "window",
    "dataset": "CTHeadDataset3d",
    "task_type": "segmentation",
    "name": "mask",
}

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def overlay_mask(args, pixel_dict, load_args):
    """Overlays mask onto input video and save it as a gif."""
    # Set up resize and crop
    load_args['resize_shape'] = util.args_to_list(load_args['resize_shape'], allow_empty=False, arg_type=int,
                                             allow_negative=False)
    load_args['crop_shape'] = util.args_to_list(load_args['crop_shape'], allow_empty=False, arg_type=int, allow_negative=False)
    loader_args = Struct(**load_args)
    data_loader = CTDataLoader(loader_args, phase=loader_args.phase, is_training=False)

    series = data_loader.dataset.series_list[args.index]
    study_name = series.study_name
    is_bottom_up = series.is_bottom_up
    input_, mask = data_loader.dataset.__getitem__(args.index)
    video_np = util.un_normalize(input_, img_format='raw', pixel_dict=pixel_dict)
    mask = mask.numpy()
    video_np_input = np.transpose(video_np, (1, 2, 3, 0))
    video_np = np.float32(video_np_input) / 255.
    mask_np = util.add_heat_map(video_np, mask, color_map='binary', normalize=False)
    input_original = mpy.ImageSequenceClip(list(video_np_input), fps=4)
    input_masked = mpy.ImageSequenceClip(list(mask_np), fps=4)
    clip = mpy.clips_array([[input_original, input_masked]])

    output_path = os.path.join(args.output_dir, 'masks', '{}_{}_mask_sum_{}.gif'\
                               .format(study_name, int(is_bottom_up), int(np.sum(mask))))
    clip.write_gif(output_path, verbose=False)

def check_annotation(args):
    """Check if new ranges in annotation path is correct and report scan direction.

    Args:
        csv_path: Path to the csv file with original annotations.
        annotation_path: Path to the annotation csv file made when drawing masks.
        root_dir: Root directory to DICOM files.
    """

    df_original = pd.read_csv(args.csv_path)
    df_mask = pd.read_csv(args.annotation_path)
    aneurysm_folders = ['aneurysm_0306', 'aneurysm_0713', 'aneurysm_1417']
    for aneurysm_folder in aneurysm_folders:
        for dir_path, _, filenames in os.walk(os.path.join(args.root_dir, aneurysm_folder)):
            if 'mask.npy' not in filenames:
                continue
            path_name = Path(dir_path).parents[1] if os.path.basename(Path(dir_path).parents[0]) == 'study' \
                                                  else Path(dir_path).parents[0]
            study_name = os.path.basename(path_name)
            dcm_files = sorted([f for f in filenames if f.endswith('.dcm')])
            z_size = len(dcm_files) + 1
            for i, r in df_original.iterrows():
                for j, s in df_mask.iterrows():
                    try:
                        origin_acc = str(int(r['Acc']))
                        mask_acc = str(int(s['Acc']))
                    except:
                        # Accession numbers can be NaNs
                        origin_acc = None
                        mask_acc = None
                    if r['AnonID'] == study_name or origin_acc == study_name:
                        if (s['AnonID'] == study_name or mask_acc == study_name) and s['Flip'] == 6.:
                            print('{}: {} ~ {}'.format(study_name,
                                                        z_size-int(r['CTA image # start'])-int(s['My end']),
                                                        z_size-int(r['CTA image # end'])-int(s['My start'])))

def check_direction_stats(args):
    """Check for machine and scan direction correlation."""
    st_change = defaultdict(list)  # List of series that change slice thickness mid-scan
    # {bottom_up: [list of series], top_down: [list of series]}
    scan_direction = defaultdict(list)
    # dir_st_mach = {direction: {slice_thickness: {machine: [list of series]}}}
    dir_st_mach = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    aneurysm_folders = ['aneurysm_0306', 'aneurysm_0713', 'aneurysm_1417']
    for aneurysm_folder in aneurysm_folders:
        for dir_path, _, filenames in os.walk(os.path.join(args.root_dir, aneurysm_folder)):
            path_name = Path(dir_path).parents[1] if os.path.basename(Path(dir_path).parents[0]) == 'study' \
                                                  else Path(dir_path).parents[0]
            study_name = os.path.basename(path_name)
            dcm_files = sorted([f for f in filenames if f.endswith('.dcm')])
            if args.use_mask:
                if 'mask.npy' not in filenames:
                    continue
                else:
                    mask_path = os.path.join(dir_path, 'mask.npy')
            if len(filenames) < 10:
                continue
            try:
                dcm_first = util.read_dicom(os.path.join(dir_path, dcm_files[0]))
                dcm_second = util.read_dicom(os.path.join(dir_path, dcm_files[1]))
                dcm_last = util.read_dicom(os.path.join(dir_path, dcm_files[-1]))
                is_bottom_up = (dcm_second.ImagePositionPatient[2] - dcm_first.ImagePositionPatient[2] > 0)
                slice_thickness = dcm_first.SliceThickness
                end_thickness = dcm_last.SliceThickness
                machine = dcm_first.Manufacturer
            except:
                util.print_err('Could not access required attributes for study {}.'.format(study_name))
                continue
            if args.fix_slice_thickness != []:
                if slice_thickness not in args.fix_slice_thickness or end_thickness not in args.fix_slice_thickness:
                    continue
            if slice_thickness != end_thickness:
                st_change[str(float(slice_thickness))+'-'+str(float(end_thickness))].append(study_name)
            if is_bottom_up:
                scan_direction['bottom_up'].append(study_name)
            else:
                scan_direction['top_down'].append(study_name)
            dir_st_mach[int(is_bottom_up)][float(slice_thickness)][machine].append(study_name)

    util.print_err('Dumping json file in {}...'.format(os.path.join(args.output_dir, 'jsons')))
    with open(os.path.join(args.output_dir, 'jsons', 'direction_stats.json'), 'w') as json_file:
        json.dump(st_change, json_file, sort_keys=True, default=util.json_encoder)
        json.dump(scan_direction, json_file, sort_keys=True, default=util.json_encoder)
        json.dump(dir_st_mach, json_file, sort_keys=True, default=util.json_encoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=('direction', 'annotation', 'overlay'),
                        help='"direction": get machine and slice thickness stats for each scan direction.\
                              "annotation": check if newly annotated slice ranges are correct.\
                              "overlay": save gif with mask overlaid on input.')
    parser.add_argument('--output_dir', type=str, default='/home/ajypark/',
                        help='Path to output for output gifs and json file.')
    parser.add_argument('--root_dir', type=str, default='/data3/CTA/',
                        help='Root directory to aneurysm studies.')
    parser.add_argument('--use_mask', type=util.str_to_bool,
                        help='Report stats only for series with masks.')
    parser.add_argument('--fix_slice_thickness', type=str, default='1.0,1.25',
                        help='Report stats only for series of certain slice thicknesses.')
    parser.add_argument('--csv_path', type=str, default='/data3/CTA/annotations/ann_180530.csv',
                        help='Path to original annotation csv file.')
    parser.add_argument('--annotation_path', type=str, default='/home/ajypark/ann_1mm_updated.csv',
                        help='If true, get stats for contrast data. If false, get stats for non_contrast data.')
    parser.add_argument('--index', type=int,
                        help='Index for input to overlay mask on.')

    args = parser.parse_args()

    args.fix_slice_thickness = util.args_to_list(args.fix_slice_thickness, allow_empty=True, arg_type=float)

    if args.task == 'direction':
        check_direction_stats(args)
    elif args.task == 'annotation':
        check_annotation(args)
    elif args.task == 'overlay':
        overlay_mask(args, pixel_dict, load_args)
