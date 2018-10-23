import argparse
import numpy as np
import os


def main(args):
    """Try different probabilities and print the false negative rate."""

    val_dir = os.path.dirname(args.val_path)
    val_inputs = np.load(args.val_path)
    val_labels = np.load(os.path.join(val_dir, 'labels.npy'))

    # Iterate over thresholds and get the false negative count
    val_maxes = np.max(val_inputs, axis=1)
    thresh2fp = {}
    for threshold in range(100):
        prob_threshold = threshold / 100
        thresh2fp[prob_threshold] = 0
        for i in range(len(val_maxes)):
            did_predict_aneurysm = val_maxes[i] > prob_threshold
            is_aneurysm = bool(val_labels[i])
            if is_aneurysm and not did_predict_aneurysm:
                thresh2fp[prob_threshold] += 1

    for prob_threshold, num_fp in sorted(thresh2fp.items(), key=lambda x: x[0], reverse=True):
        print('{:.2f}: {} false negatives'.format(prob_threshold, num_fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_path', type=str,
                        default='/data/head-ct/results/chute_xnet0816_val3/xgb/inputs.npy',
                        help='Path to the file with val data.')

    main(parser.parse_args())
