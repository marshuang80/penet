"""Move masks into place alongside the corresponding DICOM files"""

import argparse
import os
import shutil


def move_masks(input_dir, output_dir):
    num_copied = 0
    for base_path, _, file_names in os.walk(input_dir):
        mask_names = [f for f in file_names if not f.startswith('.')]
        base_path_parts = base_path.split(os.path.sep)
        masks_idx = base_path_parts.index('masks')
        for mask_name in mask_names:
            mask_path = os.path.join(base_path, mask_name)
            dest_path = os.path.join(output_dir, *base_path_parts[masks_idx+1:], mask_name)
            shutil.copy2(mask_path, dest_path)
            num_copied += 1

    print('Moved {} masks into place.'.format(num_copied))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing masks.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for JSON and Pickle files.')
    args_ = parser.parse_args()

    move_masks(args_.input_dir, args_.output_dir)
