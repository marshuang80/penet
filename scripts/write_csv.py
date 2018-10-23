import csv
import json
import argparse

class CSVArgParser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CSV Argument Parser')
        self.parser.add_argument('--csv_path', type=str, required=True, help='Path to log file to write.')
        self.parser.add_argument('--args_path', type=str, required=True, help='Path to args file.')

    def parse_args(self):
        args = self.parser.parse_args()
        return args

def write_hyperparams(csv_path, args_path):
    with open(args_path, 'rb') as f:
        hyperparams = json.load(f)

    fieldnames = ('learning_rate', 'lr_decay_gamma', 'lr_decay_step', 'optimizer', 'crop_shape',
                  'num_slices', 'model_depth', 'weight_decay', 'batch_size')

    with open(csv_path, 'ab') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writerow(hyperparams)

if __name__ == '__main__':
    parser = CSVArgParser()
    args = parser.parse_args()
    write_hyperparams(args.csv_path, args.args_path)