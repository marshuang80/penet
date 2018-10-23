import argparse
import json
import os
import pandas as pd
import pydicom
import util


def get_series_numbers(args):
    with open(os.path.join(args.output_dir, 'dir2type.json'), 'r') as json_fh:
        dir2type = json.load(json_fh)
    df = pd.read_csv(args.input_csv)

    for i, row in df.iterrows():
        series_dir = os.path.join(args.data_dir, str(row['Acc']))
        if os.path.exists(series_dir):
            print('Found at {}'.format(series_dir))
            for subdir in os.listdir(series_dir):
                if subdir not in dir2type:
                    while True:
                        try:
                            input_num = int(input('{} (0=contrast, 1=other)?\n>>> '.format(subdir)))
                            if input_num == 0 or input_num == 1:
                                break
                        except ValueError:
                            continue
                    dir2type[subdir] = 'contrast' if input_num == 0 else 'non_contrast'

                if dir2type[subdir] == 'contrast':
                    print('{} is contrast'.format(subdir))
                    dcm_dir = os.path.join(series_dir, subdir)
                    dcm_names = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]
                    dcm = util.read_dicom(os.path.join(dcm_dir, dcm_names[0]))
                    df.loc[i, 'CTA se'] = int(dcm.SeriesNumber)

    # Write CSV and dir2type mapping
    util.print_err('Dumping CSV file...')
    df.to_csv(os.path.join(args.output_dir, 'updated_annotations.csv'))
    util.print_err('Dumping JSON file...')
    with open(os.path.join(args.output_dir, 'dir2type.json'), 'w') as json_fh:
        json.dump(dir2type, json_fh, indent=4, sort_keys=True, default=util.json_encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/data3/CTA/new-normals/',
                        help='Path to directory with all data folders.')
    parser.add_argument('--input_csv', type=str, default='/data3/CTA/annotations/annotation.csv',
                        help='Path to CSV file with all annotations.')
    parser.add_argument('--output_dir', '-o', type=str, default='.',
                        help='Output directory for updated CSV file and dir2type mapping.')

    get_series_numbers(parser.parse_args())
