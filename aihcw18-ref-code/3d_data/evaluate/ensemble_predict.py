"""
predict.py
    load a model and output predictions on a specified dataset
"""
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np

from utils import compute_probs_from_paths


def get_parser():
    parser = argparse.ArgumentParser()

    # Predict Parameters
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, default="test")

    return parser


def predict(model_paths, split, save=False):

    all_probs = []
    labels = np.array([])
    for i, model_path in enumerate(model_paths):

        probs, labels = compute_probs_from_paths(model_path, split)
        print ("Done with model %d" % i)
        all_probs.append(probs)

    ensemble_probs = np.array(all_probs)

    if save:
        np.save('ensemble_%s_probs.npy' % split, ensemble_probs)
        np.save('ensemble_%s_labels.npy' % split, labels)
    return ensemble_probs


if __name__ == "__main__":

    args = get_parser().parse_args()

    assert args.dataset in ['valid', 'test'], f"{args.dataset} data split not supported"

    predict(args.model_paths, args.dataset, save=True)

