import argparse
import json
import os
import pandas as pd
import shutil
import util

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def sort_normals(input_dir, slice_thickness, json_path):
    dcms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dir_path, _, filenames in os.walk(input_dir):
        for file in tqdm(filenames):
            try:
                dcm = util.read_dicom(os.path.join(dir_path,file))
                thickness = float(dcm.SliceThickness)
                series_num = str(dcm.SeriesNumber)
                acq_num = str(dcm.AcquisitionNumber)
                inst_num = str(dcm.InstanceNumber)
                folder = os.path.basename(Path(dir_path).parents[0])
            except:
                continue
            if thickness == slice_thickness:
                dcms[folder][series_num + '_' + acq_num][inst_num].append(file)

    with open(json_path, 'w') as json_file:
        json.dump(dcms, json_file, indent=4, sort_keys=True)


def restructure_directory(input_dir, output_dir, json_path):
    with open(json_path, 'r') as json_file:
        dcm_dict = json.load(json_file)

    for acc, series_dict in tqdm(dcm_dict.items()):
        folder = os.path.join(input_dir, acc)
        subfolder = [s for s in os.listdir(folder) if s.startswith('ST')]
        assert len(subfolder) == 1, "Multiple subfolders present in {}.".format(acc)
        subfolder = subfolder[0]

        if len(series_dict) > 1:
            continue

        try:
            for series_acq, inst_num in series_dict.items():

                dcm = util.read_dicom(os.path.join(folder, subfolder, series_dict[series_acq]['1'][0]))
                description = dcm.SeriesDescription.replace('/', ' ')
                Path(os.path.join(output_dir, acc, description)).mkdir(parents=True, exist_ok=True)

                series_num = series_acq.split('_')[0]
                if len(inst_num) < 10:
                    continue
                    
                for i in range(1, len(inst_num) + 1):
                    source_path = os.path.join(folder, subfolder, inst_num[str(i)][0])
                    dest_path = os.path.join(output_dir, acc, description, inst_num[str(i)][0])
                    shutil.copy(source_path, dest_path)
                    os.rename(dest_path,
                              os.path.join(output_dir, acc, description,
                                           'IM-' + series_num.zfill(4) + '-' + str(i).zfill(4) + '.dcm'))
        except:
            util.print_err('Error occurred while copying {}. Skipping...'.format(acc))
            continue


def update_csv(csv_path, output_dir, json_path, out_csv_path):

    df = pd.read_csv(csv_path)

    with open(json_path, 'r') as json_file:
        dcm_dict = json.load(json_file)

    for acc in tqdm(os.listdir(output_dir)):
        if acc in dcm_dict.keys():
            series_acq = list(dcm_dict[acc].keys())
            series_num = series_acq[0].split('_')[0]

            df = df.append({'Acc': acc, 'CTA se': int(series_num)}, ignore_index=True)

    df.to_csv(out_csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, choices=('sort', 'restructure', 'annotate'),
                        help='"sort": sort dicoms and dump into json.\
                        "restructure": restructure directories according to json.\
                        "annotate": update csv file after restructuring directories.')
    parser.add_argument('--csv_path', type=str, default='/data3/CTA/annotations/ann_180529.csv',
                        help='Path to csv file with annotations.')
    parser.add_argument('--out_csv_path', type=str, default='/data3/CTA/annotations/ann_180530.csv',
                        help='Path for output csv file with new files added.')
    parser.add_argument('--input_dir', type=str, default='/data3/CTA/normals',
                        help='Directory to sort.')
    parser.add_argument('--output_dir', type=str, default='/data3/CTA/normal_0314')
    parser.add_argument('--slice_thickness', type=float, default=1.25,
                        help='Restructures series with given slice thickness.')
    parser.add_argument('--json_path', type=str, default='/data3/CTA/notebooks/normal_125.json',
                        help='Path to json file to dump after sort, or to read before restructure.')

    args_ = parser.parse_args()

    if args_.task == 'sort':
        sort_normals(args_.input_dir, args_.slice_thickness, args_.json_path)
    elif args_.task == 'restructure':
        restructure_directory(args_.input_dir, args_.output_dir, args_.json_path)
    elif args_.task == 'annotate':
        update_csv(args_.csv_path, args_.output_dir, args_.json_path, args_.out_csv_path)


